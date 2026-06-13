"""
@author: Boules Ashraf 2026

This File contains
1. BertDataset class which is a custom dataset class for BERT pretraining.
 - take a corpus of documents and a tokenizer as input 
 - apply tokenization, masking, and padding to prepare the data for BERT pretraining.
 
2. collate_fn function which is used to collate the data samples into a batch.
3. load_corpus function which is used to load the corpus from a text file.
 - prepare the docs by splitting the text into documents based on empty lines.
 - return a list of documents, where each document is a list of sentences.
 

"""

import random
import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_seq_len=512):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.special_token_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        }

        # Build list of valid documents
        self.valid_docs = [
            doc for doc in corpus if len(doc) > 0
        ]

    def __len__(self):
        return len(self.valid_docs)

    def __getitem__(self, idx):

        document = self.valid_docs[idx]
        create_positive = (len(document) > 1 and random.random() < 0.5)

        if create_positive:
            a_idx = random.randint(0, len(document) - 2)

            sent_A = document[a_idx]
            sent_B = document[a_idx + 1]

            nsp_label = 1

        else:
            sent_A = random.choice(document)

            random_doc_idx = idx

            while random_doc_idx == idx:
                random_doc_idx = random.randint(
                    0,
                    len(self.valid_docs) - 1
                )

            random_doc = self.valid_docs[random_doc_idx]
            sent_B = random.choice(random_doc)

            nsp_label = 0

        tokens_A = self.tokenizer.tokenize(sent_A)
        tokens_B = self.tokenizer.tokenize(sent_B)

        max_tokens = self.max_seq_len - 3

        while len(tokens_A) + len(tokens_B) > max_tokens:
            if len(tokens_A) > len(tokens_B):
                tokens_A.pop()
            else:
                tokens_B.pop()

        tokens = (
            [self.tokenizer.cls_token]
            + tokens_A
            + [self.tokenizer.sep_token]
            + tokens_B
            + [self.tokenizer.sep_token]
        )

        token_type_ids = (
            [0] * (len(tokens_A) + 2)
            + [1] * (len(tokens_B) + 1)
        )

        input_ids, mlm_labels = self._apply_mlm(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = self.max_seq_len - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        mlm_labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "nsp_label": torch.tensor(nsp_label, dtype=torch.long),
        }


    def _apply_mlm(self, tokens):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mlm_labels = [-100] * len(input_ids)
        
        for i, token in enumerate(tokens):
            
            if token in {
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            }:
                continue

            # 15% MLM probability
            if random.random() < 0.15:
                original_token_id = input_ids[i]
                mlm_labels[i] = original_token_id

                prob = random.random()

                if prob < 0.80:
                    # 80% -> [MASK]
                    input_ids[i] = self.tokenizer.mask_token_id

                elif prob < 0.90:
                    # 10% -> random token
                    random_token = random.randint(0, self.tokenizer.vocab_size - 1)
                    while random_token in self.special_token_ids:
                        random_token = random.randint(0, self.tokenizer.vocab_size - 1)
                        
                    input_ids[i] = random_token

                # 10% -> unchanged
        return input_ids, mlm_labels



def collate_fn(batch):

    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "mlm_labels": torch.stack([x["mlm_labels"] for x in batch]),
        "nsp_labels": torch.stack([x["nsp_label"] for x in batch]),
    }


def load_corpus(file_path):
    documents = []
    current_doc = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
            else:
                current_doc.append(line)

    if current_doc:
        documents.append(current_doc)

    return documents
