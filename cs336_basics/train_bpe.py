from typing import Tuple
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import multiprocessing as mp

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    merges = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        frequency_table = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # remove special tokens from chunk
            # with mp.Pool(5) as pool:
            splits = re.split("|".join(re.escape(tok) for tok in special_tokens), chunk)
            for split in splits:
                if split.strip() == "":
                    continue
                pretokenized_data = re.findall(PAT, split)
                for token in pretokenized_data:
                    token_bytes = token.encode('utf-8')
                    token_tuple = tuple(bytes([b]) for b in token_bytes)
                    frequency_table[token_tuple] += 1

        
    pair_frequencies = {}
    for token_tuple, freq in frequency_table.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + freq
    while len(vocab) < vocab_size:
        # take the max lexicographical pair
        best_pair = max(pair_frequencies.items(), key=lambda x: (x[1], x[0]))[0]
        if best_pair not in merges:
            merges.append(best_pair) 
        # if best_pair frequency is less than 2, stop
        if pair_frequencies[best_pair] < 2:
            print("break")
            break
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        # update frequency table with merged pairs
        for token_tuple, freq in list(frequency_table.items()):
            merged = []
            a, b = best_pair
            i = 0
            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and token_tuple[i] == a and token_tuple[i + 1] == b:
                    merged.append(a + b)
                    if i - 1 >= 0:
                        new_pair = (token_tuple[i - 1], a + b)
                        pair_frequencies[new_pair] = pair_frequencies.get(new_pair, 0) + freq
                        pair_frequencies[(token_tuple[i - 1], a)] = pair_frequencies.get((token_tuple[i - 1], a), 0) - freq
                    if i + 2 < len(token_tuple):
                        new_pair = (a + b, token_tuple[i + 2])
                        pair_frequencies[new_pair] = pair_frequencies.get(new_pair, 0) + freq
                        pair_frequencies[(b, token_tuple[i + 2])] = pair_frequencies.get((b, token_tuple[i + 2]), 0) - freq

                    i += 2
                else:
                    merged.append(token_tuple[i])
                    i += 1
            frequency_table[tuple(merged)] = frequency_table.pop(token_tuple)
        pair_frequencies[best_pair] = 0
        # remove pairs with zero or negative frequency
        for pair in list(pair_frequencies.keys()):
            if pair_frequencies[pair] <= 0:
                del pair_frequencies[pair]


    

    return vocab, merges


if __name__ == "__main__":
   vocab, merges = train_bpe("sample_text.txt", 1000, ["<|endoftext|>"])
#    print(vocab)
#    print(merges)
