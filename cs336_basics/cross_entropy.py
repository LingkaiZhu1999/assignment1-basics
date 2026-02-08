import torch
from einops import rearrange
from cs336_basics.softmax import softmax

def cross_entropy(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    log_probs = shifted_logits - log_sum_exp
    if log_probs.dim() == 3 and targets.dim() == 2:
        log_probs = rearrange(log_probs, "batch seq vocab -> (batch seq) vocab")
        targets = rearrange(targets, "batch seq -> (batch seq)")
    return -log_probs[torch.arange(targets.shape[0], device=targets.device), targets].mean()