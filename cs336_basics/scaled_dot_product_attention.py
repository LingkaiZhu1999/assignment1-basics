import torch
from einops import einsum
from cs336_basics.softmax import softmax
import math

def scaled_dot_product_attention(keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
    scores = einsum(queries, keys, "... n d_k, ... m d_k -> ... n m") / math.sqrt(keys.shape[-1])
    if mask is not None:
        scores = torch.where(mask, scores, torch.tensor(float('-inf')))
    attention_weights = softmax(scores, -1)
    attention = einsum(attention_weights, values, "... n m, ... m d_v -> ... n d_v")
    # mask = torch.where(mask, 0, torch.tensor(float('-inf')))
    return attention


