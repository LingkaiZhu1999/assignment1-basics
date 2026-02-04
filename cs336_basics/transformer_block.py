import torch
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, use_rope=False, theta=None, max_seq_len=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.multihead_self_att = MultiheadSelfAttention(d_model, 
        num_heads, rope=use_rope, theta=theta, max_seq_len=max_seq_len)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions=None):
        x = x + self.multihead_self_att(self.rmsnorm1(x), token_positions)
        x = x + self.swiglu(self.rmsnorm2(x))
        return x