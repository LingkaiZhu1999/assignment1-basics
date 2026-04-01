import torch
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from einops import repeat

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


    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        x = self.embedding(x)
        batch_size, seq_len = x.shape[0], x.shape[1]
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_seq_len = past_key_values[0][0].shape[-2]
        else:
            past_seq_len = 0

        # Generate token positions: shape (batch_size, sequence_length)
        token_positions = repeat(
            torch.arange(past_seq_len, past_seq_len + seq_len, device=x.device),
            "seq -> batch seq",
            batch=batch_size,
        )
        present_key_values = [] if use_cache else None
        for i, transformer_block in enumerate(self.transformer_blocks):
            layer_past = None if past_key_values is None else past_key_values[i]
            if use_cache:
                x, layer_present = transformer_block(
                    x,
                    token_positions,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(layer_present)
            else:
                x = transformer_block(
                    x,
                    token_positions,
                    past_key_value=layer_past,
                    use_cache=False,
                )
        x = self.rmsnorm(x)
        x = self.lm_head(x)
        # x = einsum(x, self.embedding.weight, "... seq d_model, num_embedding d_model -> ... seq num_embedding")
        if use_cache:
            return x, tuple(present_key_values)
        return x
    
if __name__ == "__main__":
    from utils import count_parameters
    transformer = Transformer_LM(1600, 25, 6400, 50257, 1024, 48, 10000)
    num_params = count_parameters(transformer)
    print(num_params)
    
