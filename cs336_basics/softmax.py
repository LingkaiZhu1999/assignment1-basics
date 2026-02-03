import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values # numerical stability
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)