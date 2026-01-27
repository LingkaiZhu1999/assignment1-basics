import regex as re
from typing import Iterable
import json

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a vocabulary and merge list.

        Args:
            vocab (dict[int, bytes]): Mapping from token id to raw byte token.
            merges (list[tuple[bytes, bytes]]): Ordered BPE merge operations.
            special_tokens (list[str] | None): Optional list of string special tokens.
        """
        # self.vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
        # self.merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.tokens_to_ids = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens
        # Create merge priority lookup: merge tuple -> priority (lower index = higher priority)
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            token_to_ids = json.load(f) # tokens to ids
        ids_to_token = {int(k): v for v, k in token_to_ids.items()}
        bytes_decoder = gpt2_bytes_to_unicode()
        bytes_decoder = {v: k for k, v in bytes_decoder.items()}
        for id, token in ids_to_token.items():
            byte_list = [bytes_decoder[char] for char in token]
            ids_to_token[id] = bytes(byte_list)
        
        vocab = ids_to_token
        
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) == 2:
                    a, b = parts
                    byte_list_a = [bytes_decoder[char] for char in a]
                    byte_list_b = [bytes_decoder[char] for char in b]
                    a = bytes(byte_list_a)
                    b = bytes(byte_list_b)
                    parts = (a, b)
                    merges.append(tuple(parts))
        
        # if special_tokens is not in vocab, add them
        if special_tokens:
            max_id = max(vocab.keys())
            for token in special_tokens:
                byte_tok = token.encode("utf-8")
                if byte_tok not in vocab.values():
                    max_id += 1
                    vocab[max_id] = byte_tok
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens is not None:
            # Sort special tokens by length (descending) so longer tokens take precedence in matching
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
            splits = re.split(pattern, text)
        else:
            splits = [text]
        encoded_special_tokens = []
        pretokenized_data = []
        ids = []
        idx = 0
        for split in splits:
            if self.special_tokens is not None and split in self.special_tokens:
                encoded_special_tokens.append(split.encode('utf-8'))
                ids.append(self.tokens_to_ids[split.encode('utf-8')])
                continue
            # pretokenized_data.extend(re.findall(PAT, split))
            pretokenized_data = re.findall(PAT, split)

            for token in pretokenized_data:
                token_bytes = token.encode('utf-8')
                char_list = [bytes([b]) for b in token_bytes]

                while len(char_list) > 1:
                    # Find all consecutive pairs and their merge priority
                    best_pair = None
                    best_priority = float('inf')
                    best_pos = -1
                    
                    for i in range(len(char_list) - 1):
                        pair = (char_list[i], char_list[i+1])
                        if pair in self.merge_priority:
                            priority = self.merge_priority[pair]
                            if priority < best_priority:
                                best_priority = priority
                                best_pair = pair
                                best_pos = i
                    
                    # If no merge found, we're done
                    if best_pair is None:
                        break
                    
                    new_char_list = []
                    i = 0
                    while i < len(char_list):
                        if i < len(char_list) - 1 and (char_list[i], char_list[i+1]) == best_pair:
                            new_char_list.append(char_list[i] + char_list[i+1])
                            i += 2
                        else:
                            new_char_list.append(char_list[i])
                            i += 1
                    char_list = new_char_list
                # to ids

                # print(token, initial_representation)
                for byte_tok in char_list:
                    ids.append(self.tokens_to_ids[byte_tok])
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # return (self.encode(text) for text in iterable)
        for text in iterable:
            ids = self.encode(text)
            for id_ in ids:
                yield id_
        
    
    def decode(self, ids: list[int]) -> str:
        decoded_str = b""
        for id_ in ids:
            token = self.vocab.get(id_, b'')
            if token:
                decoded_str += token
        return decoded_str.decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    import tiktoken
    tokenizer = Tokenizer.from_files(
        "../tests/fixtures/gpt2_vocab.json",
        "../tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    )

    corpus_path = "../tests/fixtures/tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    all_ids = []
    with open(corpus_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(corpus_path) as f:
        corpus_contents = f.read()
    print(all_ids)
    assert tokenizer.decode(all_ids) == corpus_contents

    # ids = tokenizer.encode("the cat ate")
    # ids = tokenizer.encode("Hello, how are you?")
    # ids = tokenizer.encode("🙃")
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # ids = tokenizer.encode(test_string)
    # # print(ids)
    # decoded = tokenizer.decode(ids)
    # print(decoded)
    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # print(tokenized_string)
    # print(tokenized_string.count("<|endoftext|>"), test_string.count("<|endoftext|>"))
    # assert tokenized_string.count("<|endoftext|>") == 1
    # assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # # Test roundtrip
    # assert tokenizer.decode(ids) == test_string
    # ids = tokenizer.encode_iterable(["the cat", " ate"])
    # print(list(ids))


    # print(f"Loaded tokenizer with vocab size: {len(tokenizer.id_to_token)}")
    # print(f"Loaded tokenizer with merges count: {len(tokenizer.merges)}")