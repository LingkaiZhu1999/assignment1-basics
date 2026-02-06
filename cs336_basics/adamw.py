from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-5):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                t = state.get("t", 1) # Get iteration number from the state, or initial value
                m = state.get("m", torch.zeros_like(p)) # get the first moment, or initial value
                v = state.get("v", torch.zeros_like(p)) # get the second moment, or initial value
                grad = p.grad.data # Get the gradient of loss w.r.t p
                m = beta1 * m + (1 - beta1) * grad # update the first moment estimate
                v = beta2 * v + (1 - beta2) * grad * grad # update the second moment estimate
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1. - math.pow(beta1, t)) # compute adjusted lr for iteration t
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data # apply weight decay
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss 
    

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1e-3)

    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters
        loss = (weights**2).mean() # Compute a scalar loss value
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients
        opt.step() # Run optimizer step