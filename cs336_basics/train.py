import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loading import data_loading, valid_data_loading
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.gradient_clipping import gradient_clipping

import numpy as np
import wandb
import torch
torch.set_float32_matmul_precision('high')

def train(run, args): 
    train_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tinystories_train_tokenized.npy")
    train_data = np.load(train_data_path, mmap_mode="r")
    valid_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tinystories_valid_tokenized.npy")
    valid_data = np.load(valid_data_path, mmap_mode="r")
    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, rope_theta=args.theta).to(args.device)
    model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    iterations = args.total_tokens_processed // args.batch_size // args.context_length
    for iter in range(iterations):
        model.train()
        current_lr = learning_rate_schedule(iter+1, args.lr_max, args.lr_min, args.iter_warmup, t_cos_anneal=args.iter_cos_annealing)
        run.log({"lr": current_lr})
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad()
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device=args.device)
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()
        run.log({"loss/train": loss.item()})
        print("iter: ", iter, "loss: ", loss.item())
        if iter % args.eval_interval == 0:
            valid_loss = validate(valid_data, model, args)
            run.log({"loss/valid": valid_loss}) 
            print("iter: ", iter, "train loss: ", loss.item(), "valid loss: ", valid_loss)
    save_checkpoint(model, optimizer, iterations, "final_model.pt")


def validate(valid_data, model, args):
    num_sequences = (len(valid_data) - 1) // args.context_length
    if num_sequences <= 0:
        return float("nan")

    iters = (num_sequences + args.batch_size - 1) // args.batch_size
    model.eval()
    total_loss = 0.0
    total_sequences = 0
    with torch.no_grad():
        for i in range(iters):
            inputs, targets = valid_data_loading(
                valid_data, args.batch_size, args.context_length, args.device, i
            )
            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)
            current_batch_size = inputs.shape[0]
            total_loss += loss.item() * current_batch_size
            total_sequences += current_batch_size
    return total_loss / total_sequences


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="GPT-2",
        description="Assignment 1 of cs336",
        epilog="N/A",
    )
    # parser.add_argument("--iterations", type=int, default=1000, help="total iterations for training")
    parser.add_argument("--total_tokens_processed", type=int, default=327680000, help="batch size * total step count * context length")
    parser.add_argument("--iter_warmup", type=int, default=200, help="warm up iterations")
    parser.add_argument("--lr_max", type=float, default=3e-3, help="learning rate max")
    parser.add_argument("--lr_min", type=float, default=3e-4, help="learning rate min")
    parser.add_argument("--iter_cos_annealing", type=int, default=20000, help="iteration for cosine annealing")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay (l1/l2 norm coefficient)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--context_length", type=int, default=256, help="max sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=1344,)
    parser.add_argument("--vocab_size", type=int, default=10000, help="10000 for tinystories, 30000 for owt")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--theta", type=float, default=10000, help="theta for rope")
    parser.add_argument("--eval_interval", type=int, default=1000, help="interval to run validation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="max L2 norm for gradient clipping")

    args = parser.parse_args()
    run = wandb.init(project=parser.prog, config=vars(args))
    print(args)
    train(run, args=args)


    # load data 
   

