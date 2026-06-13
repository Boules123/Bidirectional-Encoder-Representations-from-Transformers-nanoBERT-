"""
The Encoder module for BERT, including:
1. EmbeddingLayer: Combines token, segment, and position embeddings.
 - contain 3 embedding layers: token embedding, segment embedding, and position embedding. It also includes layer normalization and dropout for regularization.
 - return the combined embeddings of shape (B, T, E), where B is the batch size, T is the sequence length, and E is the embedding size.
 
2. MHA: Multi-Head Attention mechanism.
 - Multi-Head Attention (MHA) consists of multiple attention heads q, k, v projections, and an output projection. It computes attention scores, applies a mask if provided, and returns the attended output.
 
3. EncoderLayer: A single transformer encoder layer with self-attention and feed-forward network.
 - puts together the MHA and a feed-forward network, with residual connections and layer normalization. It processes the input embeddings and returns the output of shape (B, T, E).
 - use GELU activation function in the feed-forward network, which is a common choice for transformer architectures.
 
The Encoder is designed to process input sequences and produce contextualized embeddings for each token, which can
    then be used for downstream tasks such as Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_length):
        super(EmbeddingLayer, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size) # create token embedding layer
        self.segment_embedding = nn.Embedding(2, embed_size)  # for sentence A and B
        self.position_embedding = nn.Embedding(max_seq_length, embed_size) # give each position a unique embedding 

        self.layer_norm = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        token_embeds = self.token_embedding(input_ids)
        segment_embeds = self.segment_embedding(segment_ids)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = token_embeds + segment_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings #(B, T, E)
    

class MHA(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()

        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size 
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # q, k, v projections
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

        self.out_proj = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, t, e = x.size()
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q,k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).bool()
            # !error: without this line, the code will throw an error when using FP16 precision because -1e9 is not representable in FP16.   
            # Get the minimum safe value for the current precision (FP16 or FP32)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, min_value) 
        
        attn = F.softmax(scores,dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, e)
        
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, n_heads, hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MHA(embed_size, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-attention
        attn_output = self.self_attention(x, mask)
        attn_output = self.dropout(attn_output)
        out1 = self.layer_norm1(x + attn_output)  
        
        # Feed-forward
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.layer_norm2(out1 + ff_output) 
        
        return out2

