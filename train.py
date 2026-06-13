"""
This training script fine-tunes a pre-trained BERT model on a custom text corpus 

the training process includes the following features:
- Mixed precision training (FP16) for faster training and reduced memory usage.
- Gradient accumulation to simulate larger batch sizes without increasing memory usage.
- Gradient clipping to prevent exploding gradients.
- Two optimizers: Muon and AdamW, each with its own learning rate schedule.
- Model checkpointing after each epoch.
- Evaluation on a validation set after each epoch, reporting average loss and NSP accuracy.

The script saves model checkpoints after each epoch, including the model state, optimizer states, scheduler states, and training statistics (loss, validation loss, NSP accuracy).
The script also evaluates the model on a validation set after each epoch, reporting the average loss and NSP accuracy.
"""


import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from data import load_corpus, BERTDataset, collate_fn
from model import BERT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_model(model, dataloader, optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw, 
                scaler, device):
    model.train()
    total_loss = 0.0
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()
    grad_norm = 0.0
    
    for step, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):

            loss, mlm_logits, nsp_logits = model(
                input_ids=input_ids,
                segment_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                nsp_labels=nsp_labels
            )
            current_loss = loss.item()
            total_loss += current_loss
            loss = (loss / GRADIENT_ACCUMULATION_STEPS)

        scaler.scale(loss).backward()
        should_step = ((step + 1)% GRADIENT_ACCUMULATION_STEPS== 0)

        if should_step:
            scaler.unscale_(optimizer_muon)
            scaler.unscale_(optimizer_adamw)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP_VALUE)
            
            scaler.step(optimizer_muon)
            scaler.step(optimizer_adamw)
            
            old_scale = scaler.get_scale()
            scaler.update()
            # Only advance LR schedule if optimizer steps were not skipped due to inf/NaN gradients
            if scaler.get_scale() >= old_scale:
                scheduler_muon.step()
                scheduler_adamw.step()
            
            optimizer_muon.zero_grad()
            optimizer_adamw.zero_grad()

        if step % 50 == 0:
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Step {step}/{len(dataloader)} | "
                f"Loss: {current_loss:.4f} | "
                f"LR_muon: {scheduler_muon.get_last_lr()[0]:.6f} | "
                f"LR_adamw: {scheduler_adamw.get_last_lr()[0]:.6f} | "
                f"Grad Norm: {grad_norm:.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    total_samples = 0
    nsp_correct = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch["mlm_labels"].to(device) # (B, T)
        nsp_labels = batch["nsp_labels"].to(device) # (B,)

        batch_size = input_ids.size(0)
        total_samples += batch_size

        loss, mlm_logits, nsp_logits = model(
            input_ids=input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
            mlm_labels=mlm_labels,
            nsp_labels=nsp_labels
        )
        # mlm_logits (B, T, V), nsp_logits (B, 2)

        val_loss += loss.item() 
        # nsp accuracy
        nsp_pred = torch.argmax(nsp_logits, dim=1) # (B,)
        nsp_correct += (nsp_pred == nsp_labels).sum().item()
    
    avg_loss = val_loss / len(dataloader)
    nsp_accuracy = nsp_correct / total_samples
    
    return avg_loss, nsp_accuracy
        





DATA_PATH = "/kaggle/input/datasets/kaushaltiwari/tiny-shakespeare/tiny-shakespeare.txt"
SPLIT_RATIO = 0.9

VOCAB_SIZE = 30522
EMBED_SIZE = 768
NUM_HEADS = 12
HIDDEN_SIZE = 3072
NUM_LAYERS = 12

MAX_SEQ_LENGTH = 512
DROPOUT = 0.1

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 5e-5

GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CLIP_VALUE = 5.0
SEED = 131

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Seed: {SEED}")

print("Loading corpus...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
corpus = load_corpus(DATA_PATH)

n = len(corpus)
split_idx = int(n * SPLIT_RATIO)
train_corpus = corpus[:split_idx]
val_corpus = corpus[split_idx:]

train_dataset = BERTDataset(corpus=train_corpus, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LENGTH)
val_dataset = BERTDataset(corpus=val_corpus, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LENGTH)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

print(f"Total training batches: {len(train_dataloader)}")
print(f"Total validation batches: {len(val_dataloader)}")

model = BERT(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=DROPOUT).to(device)

n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {n:,}")


optimizer_muon, optimizer_adamw = model.configure_optimizers()
num_training_steps = (len(train_dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler_muon = get_linear_schedule_with_warmup(optimizer_muon, num_warmup_steps, num_training_steps)
scheduler_adamw = get_linear_schedule_with_warmup(optimizer_adamw, num_warmup_steps, num_training_steps)

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    avg_train_loss = train_model(
        model=model,
        dataloader=train_dataloader,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        scheduler_muon=scheduler_muon,
        scheduler_adamw=scheduler_adamw,
        scaler=scaler,
        device=device
    )

    avg_val_loss, nsp_accuracy = evaluate(
        model=model,
        dataloader=val_dataloader,
        device=device
    )
    
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Average Loss: {avg_train_loss:.4f} | "
        f"Validation Loss: {avg_val_loss:.4f} | "
        f"NSP Accuracy: {nsp_accuracy:.4f}"
    )

    checkpoint_path = "bert_model.pt"
    torch.save({
            "epoch": epoch + 1,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict_adamw":optimizer_adamw.state_dict(),
            "optimizer_state_dict_muon":optimizer_muon.state_dict(),
            "scheduler_state_dict_adamw":scheduler_adamw.state_dict(),
            "scheduler_state_dict_muon":scheduler_muon.state_dict(),
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "nsp_accuracy": nsp_accuracy
        }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")

print("Training completed.")


# Epoch 1/30 | Average Loss: 8.8280 | Validation Loss: 7.2684 | NSP Accuracy: 0.6362
# Epoch 5/30 | Average Loss: 6.3856 | Validation Loss: 6.5044 | NSP Accuracy: 0.7427
# Epoch 10/30 | Average Loss: 5.7723 | Validation Loss: 5.9627 | NSP Accuracy: 0.7510
# Epoch 15/30 | Average Loss: 5.2441 | Validation Loss: 5.4624 | NSP Accuracy: 0.7732
# Epoch 20/30 | Average Loss: 4.8296 | Validation Loss: 5.2445 | NSP Accuracy: 0.7828
# Epoch 25/30 | Average Loss: 4.3740 | Validation Loss: 4.9617 | NSP Accuracy: 0.8036
# Epoch 30/30 | Average Loss: 4.1367 | Validation Loss: 4.5710 | NSP Accuracy: 0.8008
