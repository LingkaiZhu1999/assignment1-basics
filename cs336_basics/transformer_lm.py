import torch
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.softmax import softmax
from einops import einsum, repeat

class Transformer_LM(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope_theta=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, 
        d_ff, use_rope=True, theta=rope_theta, max_seq_len=context_length) for i in range(num_layers)]
        )
        self.rmsnorm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        # self.output_embedding = Embedding(vocab_size, d_model)


    def forward(self, x):
        x = self.embedding(x)
        # Generate token positions: shape (batch_size, sequence_length)
        token_positions = repeat(torch.arange(x.shape[1], device=x.device), 'seq -> batch seq', batch=x.shape[0])
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, token_positions)
        x = self.rmsnorm(x)
        x = self.lm_head(x)
        # x = einsum(x, self.embedding.weight, "... seq d_model, num_embedding d_model -> ... seq num_embedding")
        return x
    
if __name__ == "__main__":
    from utils import count_parameters
    transformer = Transformer_LM(1600, 25, 6400, 50257, 1024, 48, 10000)
    num_params = count_parameters(transformer)
    print(num_params)
    