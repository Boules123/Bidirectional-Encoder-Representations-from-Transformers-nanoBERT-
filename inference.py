import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from model import BERT


VOCAB_SIZE = 30522
EMBED_SIZE = 768
NUM_HEADS = 12
HIDDEN_SIZE = 3072
NUM_LAYERS = 12
MAX_SEQ_LENGTH = 512
DROPOUT = 0.0  # No dropout during inference

CHECKPOINT_PATH = "bert_epoch.pt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_model(checkpoint_path):
    """Load a trained BERT model from a checkpoint file."""

    model = BERT(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']} "
        f"(loss: {checkpoint['loss']:.4f})"
    )

    return model


def predict_masked_tokens(model, text, top_k=5):

    tokens = tokenizer.tokenize(text)

    max_tokens = MAX_SEQ_LENGTH - 2
    tokens = tokens[:max_tokens]

    tokens = (
        [tokenizer.cls_token]
        + tokens
        + [tokenizer.sep_token]
    )

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Single sentence → all segment IDs are 0
    segment_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    # Pad to MAX_SEQ_LENGTH
    padding_length = MAX_SEQ_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    # Convert to tensors
    input_ids = torch.tensor(
        [input_ids], dtype=torch.long
    ).to(device)

    segment_ids = torch.tensor(
        [segment_ids], dtype=torch.long
    ).to(device)

    attention_mask = torch.tensor(
        [attention_mask], dtype=torch.long
    ).to(device)

    # Find [MASK] positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (
        (input_ids[0] == mask_token_id)
        .nonzero(as_tuple=True)[0]
    )

    if len(mask_positions) == 0:
        print("No [MASK] token found in the input.")
        return []

    # Forward pass
    with torch.no_grad():
        _, mlm_logits, _ = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask
        )

    # Collect predictions for each [MASK]
    results = []

    for pos in mask_positions:

        logits = mlm_logits[0, pos]
        probs = F.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probs, top_k
        )

        predictions = []

        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.convert_ids_to_tokens(
                [idx.item()]
            )[0]

            predictions.append({
                "token": token,
                "probability": prob.item()
            })

        results.append({
            "position": pos.item(),
            "predictions": predictions
        })

    return results


def predict_next_sentence(model, sentence_a, sentence_b):

    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)

    # Truncate (reserve 3 for [CLS], [SEP], [SEP])
    max_tokens = MAX_SEQ_LENGTH - 3

    while len(tokens_a) + len(tokens_b) > max_tokens:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

    tokens = (
        [tokenizer.cls_token]
        + tokens_a
        + [tokenizer.sep_token]
        + tokens_b
        + [tokenizer.sep_token]
    )

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    segment_ids = (
        [0] * (len(tokens_a) + 2)
        + [1] * (len(tokens_b) + 1)
    )

    attention_mask = [1] * len(input_ids)

    # Pad to MAX_SEQ_LENGTH
    padding_length = MAX_SEQ_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    # Convert to tensors
    input_ids = torch.tensor(
        [input_ids], dtype=torch.long
    ).to(device)

    segment_ids = torch.tensor(
        [segment_ids], dtype=torch.long
    ).to(device)

    attention_mask = torch.tensor(
        [attention_mask], dtype=torch.long
    ).to(device)

    # Forward pass
    with torch.no_grad():
        _, _, nsp_logits = model(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask
        )

    probs = F.softmax(nsp_logits[0], dim=-1)
    predicted_label = torch.argmax(probs).item()
    confidence = probs[predicted_label].item()

    return {
        "label": predicted_label,
        "label_text": "IsNext" if predicted_label == 1 else "NotNext",
        "confidence": confidence,
        "probabilities": {
            "NotNext": probs[0].item(),
            "IsNext": probs[1].item()
        }
    }