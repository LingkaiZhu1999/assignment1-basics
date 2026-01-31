import torch
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((out_features, in_features), 
            device=device, dtype=dtype)))
        # no bias 



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, 
        "... in_features, out_features in_features -> ... out_features")
