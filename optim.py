"""
Given a 2D matrix G (typically a momentum buffer of a neural network hidden layer), 
the algorithm normalizes the initial matrix and performs 5 steps of a quintic polynomial recurrence relation.
For k = 0, 1, 2, 3, 4:
(X_{k+1} = X_{k} ( aI + bX_{k}^{T} X_{k} + c(X_{k}^{T} X_{k})^{2}))
"""

import torch

def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def update_muon(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1-beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5  # scale for non-square matrices
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] #
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                update = update_muon(p.grad, state['momentum_buffer'], beta=group['momentum'])
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])
        
        return loss


# use 2 optimizers 
class AdamwMuon(torch.optim.Optimizer):
    def __init__(self, params):
        defaults = dict(lr=1e-4, weight_decay=0.01)
        super().__init__(params, defaults)
        
    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = 0.9, 0.999

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            corrected_avg = exp_avg / bias_correction1
            corrected_avg_sq = exp_avg_sq / bias_correction2

            denom = (corrected_avg_sq.sqrt() + 1e-8)

            step_size = group['lr']

            p.data.addcdiv_(corrected_avg, denom, value=-step_size)
            
    def _step_muon(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p)
            update = update_muon(p.grad, state['momentum_buffer'], beta=0.95)
            p.mul_(1 - group['lr'] * group['weight_decay'])
            p.add_(update.reshape(p.shape), alpha=-group['lr'])
            
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")