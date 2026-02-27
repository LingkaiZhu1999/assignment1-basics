import torch
from cs336_basics.transformer_lm import Transformer_LM
from softmax import temperature_scaling_softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.checkpointing import load_checkpoint

def decoding(prompt: torch.Tensor, model: torch.Tensor, temperature: float, top_p: int, max_tokens: int= 1024, end_token: int=256):
    for i in range(max_tokens):
        outputs = model(prompt)
        last_token_logits = outputs[:, -1:, :]

        probs = temperature_scaling_softmax(last_token_logits, -1, temperature=temperature)
        topk = torch.topk(probs, k=top_p, dim=-1)
        masked = torch.zeros_like(probs)
        masked.scatter_(dim=-1, index=topk.indices, src=topk.values)
        probs = masked / torch.sum(masked, dim=-1, keepdim=True)
        # next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
        prompt = torch.cat([prompt, next_token], dim=1)
        if (next_token == end_token).any():
            break
    return prompt


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "TinyStoriesV2-GPT4-train.json",
        "TinyStoriesV2-GPT4-train_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    model = Transformer_LM(512, 16, 1344, 10000, 256, 4, 10000)
    state_dict = torch.load("../final_model.pt")["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    # load_checkpoint("./final_model.pt", model, )
    inputs = torch.tensor([[256]])
    temperature = 1
    top_p = 20
    max_tokens = 256
    end_token = 256
    generated = decoding(inputs, model, temperature, top_p=top_p, max_tokens=max_tokens)
    print(generated.squeeze().tolist())
    # decode
    generated_text = tokenizer.decode(generated.squeeze().tolist())
    print(generated_text)
