
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attention import Attention
import copy


from torch_butterfly import Butterfly

Sparse_Linear = Butterfly

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config, idx):
        super().__init__()

        #if ((config["fabnet_att_layer"]<0) or ((config["num_layers"]-idx) > config["fabnet_att_layer"])):
        # print ("a", config["num_layers"]-(idx+1))
        # print ("b", config["fabnet_att_layer"])
        if ((config["fabnet_att_layer"]<0) or ((config["num_layers"]-(idx+1)) < config["fabnet_att_layer"])):
            self.attn_type = config["attn_type"]
        else:
            self.attn_type = "softmax"

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        att_config = copy.deepcopy(config)
        att_config["attn_type"] = self.attn_type
        print (self.attn_type)
        self.mha = Attention(att_config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])
        
        if config["is_butterfly"] and self.attn_type != "softmax":
            Linear = Sparse_Linear
        else:
            Linear = nn.Linear
        self.mlpblock = nn.Sequential(
            Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )
        
    def forward(self, X, mask):
        if self.attn_type == "fft":
            X = (self.mha(self.norm1(X), mask)) + X
            X = self.mlpblock(self.norm2(X)) + X
        else:
            X = self.dropout1(self.mha(self.norm1(X), mask)) + X
            X = self.mlpblock(self.norm2(X)) + X
        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = Transformer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", Transformer(config, idx))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        X = self.norm(X) * mask[:, :, None]

        return X
