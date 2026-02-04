import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((d_model), 
            device=device, dtype=dtype)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(1. / self.d_model * torch.sum(torch.square(x), dim=-1, keepdim=True) + self.eps)
        rmsnorm = x / rms * self.weight
        return rmsnorm.to(in_dtype)