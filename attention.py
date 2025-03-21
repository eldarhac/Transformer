from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math

STUDENT = {'name': 'Eldar Hacohen',
           'ID': '311587661'}


def create_kqv_matrix(input_vector_dim, n_heads=1):
    d_k = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, d_k * 3)


def kqv(x, linear):
    B, N, D = x.size()

    kqv = linear(x)
    kqv = kqv.view(B, N, -1, 3)
    k, q, v = kqv.split(1, dim=-1)
    k = k.squeeze(-1)
    q = q.squeeze(-1)
    v = v.squeeze(-1)

    return k, q, v


def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    A = torch.matmul(a, b.transpose(-2, -1)) / math.sqrt(D1)
    return A


def create_causal_mask(embed_dim, n_heads, max_context_len):
    mask = torch.ones(max_context_len, max_context_len)
    mask = torch.tril(mask)
    return mask


def self_attention(v, A, mask=None):
    m = mask[-v.size(1):, -v.size(1):]
    if mask is not None:
        A = A.masked_fill(m == 0, float("-inf"))
    sa = torch.matmul(F.softmax(A, dim=-1), v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa, att


def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    sa = None
    attention_weights = []

    for i, kqv_matrix in enumerate(kqv_matrices):
        if sa is None:
            sa, att = self_attention_layer(x, kqv_matrix, mask)
        else:
            sa_head, att = self_attention_layer(x, kqv_matrix, mask)
            sa = torch.cat((sa, sa_head), dim=-1)
        attention_weights.append(att)

    assert sa.size() == x.size()
    return sa, attention_weights


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attention_weights = None

    def forward(self, x):
        sa, self.attention_weights = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa
