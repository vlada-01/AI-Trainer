from torch import nn
import torch
import torch.nn.functional as F
import math

# TODO: 
# add BatchNorm2d
# add nn.Embedding, add nn.LayerNorm
# add TransformerDecoder

# internal class
class AddNorm(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = Dropout(dropout)

    def forward(self, x, sub_layer_out):
        x = self.norm(x + self.dropout(sub_layer_out))
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=0, stride=1):
        super().__init__()

        self.h, self.w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.kernels = nn.Linear(in_channels * self.h * self.w, out_channels)
        
    def forward(self, x):
        if self.padding > 0:
            p = self.padding
            x = F.pad(x, [p, p, p, p])
        B, C, _, _ = x.size()

        x = x.unfold(2, self.h, self.stride)
        x = x.unfold(3, self.w, self.stride)
        _, _, H_out, W_out, _, _ = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, H_out, W_out, C * self.h * self.w)

        out = self.kernels(x)
        return out.permute(0, 3, 1, 2)
    
class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=self.training)
    
# internal class
class Encoder(nn.Module):
    def __init__(self, embedding_dim, h, layer_size, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(emb_size=embedding_dim, h=h)
        self.add_norm1 = AddNorm(embedding_dim, dropout=0.1)
        self.ffnn = nn.Sequential(
            Linear(embedding_dim, layer_size),
            ReLU(),
            Dropout(dropout),
            Linear(layer_size, embedding_dim),
            # No second activation
        )
        self.add_norm2 = AddNorm(embedding_dim, dropout=0.1)
        
    def forward(self, x, mask):
        mha_out = self.mha(x, x, x, mask)
        x = self.add_norm1(x, mha_out)
        ffnn_out = self.ffnn(x)
        x = self.add_norm2(x, ffnn_out)
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1, end_dim=-1)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        if self.bias is not None:
            bound = 1 / self.weight.size(1) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        
        h, w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.h = h
        self.w = w
        self.stride = stride

    def forward(self, x):
        B, C, _, _ = x.size()

        x = x.unfold(2, self.h, self.stride).unfold(3, self.w, self.stride)
        _, _, H_out, W_out, _, _ = x.size()
        x = x.reshape(B, C, H_out, W_out, self.h * self.w)
        out, _ = torch.max(x, dim=4)
        return out

# internal class
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, h):
        super().__init__()
        self.h = h
        self.h_size = emb_size // h

        self.W_Q = Linear(emb_size, emb_size, bias=False)
        self.W_K = Linear(emb_size, emb_size, bias=False)
        self.W_V = Linear(emb_size, emb_size, bias=False)

        self.W_O = Linear(emb_size, emb_size)

    #TODO: Mask needs to be added in order to solve the issue with paddings
    def forward(self, q, k, v, mask=None):
        batch, seq_len_tgt, emb_size = q.size()
        seq_len_src = k.size(1)

        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)

        Q = Q.reshape(batch, seq_len_tgt, self.h, self.h_size).transpose(1,2)
        K = K.reshape(batch, seq_len_src, self.h, self.h_size).transpose(1,2)
        V = V.reshape(batch, seq_len_src, self.h, self.h_size).transpose(1,2)
        root = math.sqrt(self.h_size)
        term = torch.matmul(Q, K.transpose(2, 3)) / root
        if mask is not None:
            term = term.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attns = torch.softmax(term, dim=-1)
        weighted_attns = torch.matmul(attns, V)

        final = weighted_attns.transpose(1,2).reshape(batch, seq_len_tgt, emb_size)

        output = self.W_O(final)

        return output
    
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, attn_mask):
        # Positional encoding still carries non zero values, requires attn_mask
        # return x.mean(dim=1)
        tmp = x * attn_mask.unsqueeze(-1)
        masked_sizes = torch.sum(attn_mask, dim=1).clamp_min(1)
        return torch.sum(tmp, dim=1) / masked_sizes.unsqueeze(-1), None
    
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len=1024, padding_idx = 0, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids, attn_mask):
        B, T = input_ids.shape
        embs = self.token_emb(input_ids)                      
        pos_ids = torch.arange(T, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_ids)
        return self.dropout(embs + pos), attn_mask

# TODO: need to update Decoder and TransformerDecoder
# class Decoder(nn.Module):
#     def __init__(self, embedding_dim, h, layer_size, dropout):
#         super().__init__()
#         self.mha1 = MultiHeadAttention(embedding_dim, h)
#         self.add_norm1 = AddNorm(embedding_dim, dropout)
#         self.mha2 = MultiHeadAttention(embedding_dim, h)
#         self.add_norm2 = AddNorm(embedding_dim, dropout)
#         self.ffnn = self.ffnn = nn.Sequential(
#             Linear(embedding_dim, layer_size),
#             ReLU(),
#             Dropout(dropout),
#             Linear(layer_size, embedding_dim),
#             # No second activation
#         )
#         self.add_norm3 = AddNorm(embedding_dim, dropout)

#     def forward(self, x, enc_out, tgt_mask, src_mask):
        
#         self_attn = self.mha1(x, x, x, tgt_mask)
#         x = self.add_norm1(x, self_attn)

#         cross_attn = self.mha2(x, enc_out, enc_out, src_mask)
#         x = self.add_norm2(x, cross_attn)

#         ffnn_out = self.ffnn(x)
#         x = self.add_norm3(x, ffnn_out)

#         return x

# class TransformerDecoder(nn.Module):
#     def __init__(self, num_layers, embedding_dim, h, ffnn_size, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             Decoder(embedding_dim, h, ffnn_size, dropout) for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(embedding_dim)

#     #TODO: might need forward(self, x, mask) later
#     def forward(self, x, enc_out, tgt_mask, src_mask):
#         for layer in self.layers:
#             x = layer(x, enc_out, tgt_mask, src_mask)
#         return self.norm(x)

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(x, min=0)
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, emb_dim, h, ffn_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder(emb_dim, h, ffn_size, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_mask):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x), attn_mask

    
