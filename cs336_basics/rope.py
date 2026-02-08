import torch
from einops import einsum, rearrange, repeat

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        theta_values = 1.0 / (theta ** ((2 * torch.arange(1, d_k // 2 + 1, 1).float() - 2) / d_k))
        self.register_buffer("theta", theta_values, persistent=False)
        positions = torch.arange(max_seq_len)
        freqs = rearrange(positions, "dim -> dim 1") * rearrange(theta_values, "dim -> 1 dim")
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_vals = self.cos_cached[token_positions]
        sin_vals = self.sin_cached[token_positions]
        if cos_vals.dim() == 2:
            cos_vals = rearrange(cos_vals, "seq d -> 1 1 seq d")
            sin_vals = rearrange(sin_vals, "seq d -> 1 1 seq d")
        elif cos_vals.dim() == 3 and x.dim() == 4:
            cos_vals = rearrange(cos_vals, "batch seq d -> batch 1 seq d")
            sin_vals = rearrange(sin_vals, "batch seq d -> batch 1 seq d")
        cos_vals = repeat(cos_vals, "... seq d -> ... seq (d repeat)", repeat=2)
        sin_vals = repeat(sin_vals, "... seq d -> ... seq (d repeat)", repeat=2)
        cos_parts = x * cos_vals
        x_rotated = x.clone()
        x_rotated[..., 0::2] = -x_rotated[..., 1::2]
        x_rotated[..., 1::2] = x[..., 0::2]
        sin_parts = x_rotated * sin_vals
        return cos_parts + sin_parts
        
