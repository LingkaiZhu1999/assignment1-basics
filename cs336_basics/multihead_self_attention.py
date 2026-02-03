import torch
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange, einsum
from cs336_basics.rope import RoPE

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: bool = False, theta: float = None, max_seq_len: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.o_proj = Linear(d_model, num_heads * self.d_v)
        self.use_rope = rope
        if rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        # self.d_k 
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        if self.use_rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        seq_len = x.shape[-2]
        # Create causal mask: True values will be kept, False will be masked to -inf
        # We want to mask out future positions, so upper triangle should be False
        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attention = scaled_dot_product_attention(K, Q, V, mask=mask)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        return output


if __name__ == "__main__":
    multihead_self_att = MultiheadSelfAttention(20, 10)
    x = torch.randn((8, 10, 20))
    output = multihead_self_att(x)



