
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from attention_quant import Attention_Quant
from qtorch.quant import Quantizer
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

        self.is_quant = False

        if ((config["fabnet_att_layer"]<0) or ((config["num_layers"]-(idx+1)) < config["fabnet_att_layer"])):
            self.attn_type = config["attn_type"]
        else:
            self.attn_type = "softmax"

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        att_config = copy.deepcopy(config)
        att_config["attn_type"] = self.attn_type
        self.mha = Attention_Quant(att_config)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])
        
        if config["is_butterfly"] and self.attn_type != "softmax":
            # Linear = Sparse_Linear
            self.linear1 = Sparse_Linear(config["transformer_dim"], config["transformer_hidden_dim"], increasing_stride=False)
            self.linear2 = Sparse_Linear(config["transformer_hidden_dim"], config["transformer_dim"], increasing_stride=False)
        else:
            # Linear = nn.Linear
            self.linear1 = nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"], dtype=torch.float32)
            self.linear2 = nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"], dtype=torch.float32)

        self.mha_quantizers = []
        for i in range(2):
            (self.mha_quantizers).append(Quantizer(forward_number=config["quant_num"], backward_number=config["quant_num"],
                                    forward_rounding="nearest", backward_rounding="stochastic"))
        self.mlp_quantizers = []
        for i in range(5):
            (self.mlp_quantizers).append(Quantizer(forward_number=config["quant_num"], backward_number=config["quant_num"],
                                    forward_rounding="nearest", backward_rounding="stochastic"))
        # self.linear1 = Linear(config["transformer_dim"], config["transformer_hidden_dim"]) 
        self.gelu = nn.GELU()
        self.mlp_dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        # self.linear2 = Linear(config["transformer_hidden_dim"], config["transformer_dim"])
        self.mlp_dropout2 = torch.nn.Dropout(p = config["dropout_prob"])
        
        # self.mlpblock = nn.Sequential(
        #     Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
        #     nn.GELU(),
        #     torch.nn.Dropout(p = config["dropout_prob"]),
        #     Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
        #     torch.nn.Dropout(p = config["dropout_prob"])
        # )

    def set_quant(self, is_quant):
        self.is_quant = is_quant
        self.mha.is_quant = is_quant

    def forward(self, X, mask):
        if self.attn_type == "fft":
            Y = self.norm1(X)
            if self.is_quant: Y = (self.mha_quantizers[0])(Y)
            Y = self.mha(Y, mask)
            X = Y + X
            if self.is_quant: X = (self.mha_quantizers[1])(X)
            # X = (self.mha(self.norm1(X), mask)) + X
            # X = self.mlpblock(self.norm2(X)) + X
        else:
            Y = self.norm1(X)
            if self.is_quant: Y = (self.mha_quantizers[0])(Y)
            Y = self.mha(Y, mask)
            Y = self.dropout1(Y)
            X = Y + X
            if self.is_quant: X = (self.mha_quantizers[1])(X)
            # X = self.dropout1(self.mha(self.norm1(X), mask)) + X
            # X = self.mlpblock(self.norm2(X)) + X

        # Perform Norm with quant
        Y = self.norm2(X)
        if self.is_quant:
            Y = (self.mlp_quantizers[0])(Y)
            # print ("====== Quant after normalization ======")
        # else:
        #     print ("====== No Qaunt ======")

        # Perform MLP with quant
        Y = self.linear1(Y)
        if self.is_quant: Y = (self.mlp_quantizers[1])(Y.float())
        Y = self.gelu(Y)
        if self.is_quant: Y = (self.mlp_quantizers[2])(Y)
        Y = self.mlp_dropout1(Y)
        Y = self.linear2(Y)
        if self.is_quant: Y = (self.mlp_quantizers[3])(Y.float())
        Y = self.mlp_dropout2(Y)
        X = Y + X
        if self.is_quant: X = (self.mlp_quantizers[4])(X)

        return X

class Model_Quant(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]
        ###### Quantization ######
        self.is_quant = False
        self.emd_quantizer = Quantizer(forward_number=config["quant_num"], backward_number=config["quant_num"],
                                    forward_rounding="nearest", backward_rounding="stochastic")
        self.last_quantizer = Quantizer(forward_number=config["quant_num"], backward_number=config["quant_num"],
                                    forward_rounding="nearest", backward_rounding="stochastic")
        ########################

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = Transformer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", Transformer(config, idx))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    def set_quant(self, is_quant):
        self.is_quant = is_quant
        for idx in range(self.num_layers):
            getattr(self, f"transformer_{idx}").is_quant = is_quant


    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if self.is_quant:
            X = self.emd_quantizer(X)

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask)

        X = self.norm(X) * mask[:, :, None]

        if self.is_quant: X = self.last_quantizer(X)

        return X
