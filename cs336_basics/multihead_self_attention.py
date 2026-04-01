import torch
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import rearrange
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
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K_current = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        if self.use_rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K_current = self.rope(K_current, token_positions)
        V_current = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat((past_k, K_current), dim=-2)
            V = torch.cat((past_v, V_current), dim=-2)
        else:
            K = K_current
            V = V_current

        q_len = Q.shape[-2]
        k_len = K.shape[-2]
        if q_len == 1:
            # A single query token can always attend to all cached past keys.
            mask = None
        elif past_key_value is None:
            mask = ~torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=x.device), diagonal=1)
        else:
            past_len = past_key_value[0].shape[-2]
            key_positions = torch.arange(k_len, device=x.device)
            query_positions = torch.arange(past_len, past_len + q_len, device=x.device)
            mask = rearrange(key_positions, "k -> 1 k") <= rearrange(query_positions, "q -> q 1")

        attention = scaled_dot_product_attention(K, Q, V, mask=mask)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        if use_cache:
            return output, (K, V)
        return output


if __name__ == "__main__":
    multihead_self_att = MultiheadSelfAttention(20, 10)
    x = torch.randn((8, 10, 20))
    output = multihead_self_att(x)


