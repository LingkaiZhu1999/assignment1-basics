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
        cos_vals = repeat(cos_vals, "... d -> ... (d repeat)", repeat=2)
        sin_vals = repeat(sin_vals, "... d -> ... (d repeat)", repeat=2)
        cos_parts = x * cos_vals
        x_rotated = x.clone()
        x_rotated[..., 0::2] = -x_rotated[..., 1::2]
        x_rotated[..., 1::2] = x[..., 0::2]
        sin_parts = x_rotated * sin_vals
        return cos_parts + sin_parts
        

if __name__ == "__main__":
    rope = RoPE(theta=10000, d_k=10, max_seq_len=1000)
    rope(torch.randn(1), torch.randn(1))