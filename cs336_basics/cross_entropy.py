import torch
from cs336_basics.softmax import softmax

def cross_entropy(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    log_probs = shifted_logits - log_sum_exp
    return -log_probs[torch.arange(len(targets)), targets].mean()