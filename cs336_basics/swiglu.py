import torch
from einops import rearrange, einsum

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, dff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.weight1 = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((dff, d_model), 
            device=device, dtype=dtype)))
        self.weight2 = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((d_model, dff), 
            device=device, dtype=dtype)))
        self.weight3 = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((dff, d_model), 
            device=device, dtype=dtype)))

    def forward(self, x):
        x1 = einsum(x, self.weight1, "... in_features, out_features in_features -> ... out_features")
        x1 = x1 * torch.sigmoid(x1)
        x2 = einsum(x, self.weight3, "... in_features, out_features in_features -> ... out_features")
        x3 = x1 * x2
        x3 = einsum(x3, self.weight2, "... in_features, out_features in_features -> ... out_features")
        return x3
