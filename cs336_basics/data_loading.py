import torch
import numpy as np

def data_loading(x: np.array, batch_size: int, context_length: int, device: str):
    # Sample batch_size random starting indices
    # Each index can be from 0 to len(x) - context_length - 1 (inclusive)
    # so that x[start_idx : start_idx + context_length] is valid
    max_start_index = len(x) - context_length
    start_indices = np.random.randint(low=0, high=max_start_index, size=batch_size)
    
    # Create input and target arrays
    inputs = np.zeros((batch_size, context_length), dtype=x.dtype)
    targets = np.zeros((batch_size, context_length), dtype=x.dtype)
    
    for i, start_idx in enumerate(start_indices):
        inputs[i] = x[start_idx : start_idx + context_length]
        targets[i] = x[start_idx + 1 : start_idx + context_length + 1]
    
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def valid_data_loading(x: np.array, batch_size: int, context_length: int, device: str, index: int):
    num_sequences = (len(x) - 1) // context_length
    seq_idx0 = index * batch_size

    if seq_idx0 >= num_sequences:
        raise IndexError("Validation batch index out of range")

    effective_batch_size = min(batch_size, num_sequences - seq_idx0)

    # Create input and target arrays
    inputs = np.zeros((effective_batch_size, context_length), dtype=x.dtype)
    targets = np.zeros((effective_batch_size, context_length), dtype=x.dtype)
    for i, seq_idx in enumerate(range(seq_idx0, seq_idx0 + effective_batch_size)):
        start_idx = seq_idx * context_length
        inputs[i] = x[start_idx : start_idx + context_length]
        targets[i] = x[start_idx + 1 : start_idx + context_length + 1]
    
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)
 
if __name__ == "__main__":
    x = np.random.randint(0, 1000,(1000, ))
    # inputs, targets = data_loading(x, 8, 10, "cuda:0")
    # print(inputs, targets)
    inputs, targets = valid_data_loading(x, 8, 10, "cuda:0", 0)
    print(inputs, targets)
    
    
