import torch

def gradient_clipping(params, max_l2_norm):
    eps = 1e-6
    # Compute the total L2 norm across all parameter gradients
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            param_norm = torch.sum(param.grad * param.grad)
            total_norm += param_norm 
    total_norm = torch.sqrt(total_norm)
    
    # Scale all gradients if the total norm exceeds max_l2_norm
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad.data = param.grad.data * clip_coef