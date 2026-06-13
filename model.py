"""
The Model file contains the implementation of the BERT model

1. BERT class which is a PyTorch nn.Module that implements the BERT architecture.
 - It consists of an embedding layer, multiple encoder layers, and classifiers for both Masked Language
    Modeling (MLM) and Next Sentence Prediction (NSP).
 - structure:
    - Embedding Layer: Combines token, segment, and position embeddings.
    - Encoder Layers: A stack of transformer encoder layers that process the input embeddings.
    - NSP Classifier: A linear layer that predicts whether the second sentence follows the first.
    - MLM Classifier: A linear layer that predicts the masked tokens in the input sequence.
2. Weight Initialization: The model's weights are initialized following the BERT paper's guidelines (N(0, 0.02)).
3. Optimizer Configuration: The model provides a method to configure two optimizers: Muon for embedding parameters and AdamW for other parameters.
4. Forward Method: The forward method processes the input through the embedding layer, encoder layers, and classifiers, returning the total loss (if labels are provided), MLM logits, and NSP logits.

"""



import torch 
import torch.nn as nn

from encoder import EmbeddingLayer, EncoderLayer
from optim import Muon

class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_length, dropout):
        super(BERT, self).__init__()
        
        self.embedding = EmbeddingLayer(vocab_size, embed_size, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, hidden_size, dropout) for _ in range(num_layers)
        ])
        
        # NSP Classifier
        self.pooler_dense = nn.Linear(embed_size, embed_size) 
        self.pooler_activation = nn.Tanh()
        self.nsp_classifier = nn.Linear(embed_size, 2) 
        
        # MLM Classifier
        self.mlm_transform_dense = nn.Linear(embed_size, embed_size)
        self.mlm_transform_act = nn.GELU()
        self.mlm_transform_norm = nn.LayerNorm(embed_size)
        
        self.mlm_classifier = nn.Linear(embed_size, vocab_size, bias=False)
        
        
        self.mlm_classifier.weight = self.embedding.token_embedding.weight
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize all weights following the BERT paper (N(0, 0.02))
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following the original BERT paper."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def configure_optimizers(self):
        moun_param = []
        adamw_param = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 2 and "embedding" in name:
                moun_param.append(param)
            else:
                adamw_param.append(param)
                
        return Muon(moun_param, lr=0.02, momentum=0.95), torch.optim.AdamW(adamw_param, lr=1e-4, weight_decay=0.01)

    
    def forward(self, input_ids, segment_ids, attention_mask, mlm_labels=None, nsp_labels=None):
        x = self.embedding(input_ids, segment_ids)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask) # (B, T, E)
        
        # --- 3. Process NSP ---
        cls_output = x[:, 0]
        pooled_output = self.pooler_activation(self.pooler_dense(cls_output))
        nsp_logits = self.nsp_classifier(pooled_output)
        
        # --- 4. Process MLM ---
        mlm_hidden = self.mlm_transform_dense(x)
        mlm_hidden = self.mlm_transform_act(mlm_hidden)
        mlm_hidden = self.mlm_transform_norm(mlm_hidden)
        mlm_logits = self.mlm_classifier(mlm_hidden) + self.mlm_bias
        
        total_loss = None
        if mlm_labels is not None and nsp_labels is not None:
            mlm_loss = self.criterion(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            nsp_loss = self.criterion(nsp_logits, nsp_labels)
            total_loss = mlm_loss + nsp_loss
        
        return total_loss, mlm_logits, nsp_logits