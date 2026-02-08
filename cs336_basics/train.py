from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy import cross_entropy

import os
import numpy as np
import wandb

def train(args):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_text.npy")
    data = np.load(data_path, mmap_mode="r")
    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, rope_theta=args.theta).to(args.device)
    model.train()
    optimizer = AdamW(model.parameters(), args.lr)
    for iter in range(args.iterations):
        optimizer.zero_grad()
        inputs, targets = data_loading(data, args.batch_size, args.context_length, device=args.device)
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        print(iter, loss.item())




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="GPT-2",
        description="Assignment 1 of cs336",
        epilog="N/A",
    )

    parser.add_argument("--iterations", type=int, default=100, help="total iterations for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--context_length", type=int, default=1024, help="max sequence length")
    parser.add_argument("--d_model", type=int, default=10, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=40,)
    parser.add_argument("--vocab_size", type=int, default=10000, help="10000 for tinystories, 30000 for owt")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--theta", type=float, default=10000, help="theta for rope")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    print(args)
    train(args=args)


    # load data 
   

