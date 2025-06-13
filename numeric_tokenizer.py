import re
from typing import List, Tuple

import tiktoken


class NumericTokenizer:
    """Tokenizer that maps integers to special binary tokens with a tag bit."""

    def __init__(self, base_encoding: str = "gpt2"):
        self.enc = tiktoken.get_encoding(base_encoding)
        self.base_vocab_size = self.enc.n_vocab
        self.num_to_id = {}
        self.id_to_num = {}
        self.next_id = self.base_vocab_size

    def _allocate_id(self, value: int) -> int:
        if value not in self.num_to_id:
            self.num_to_id[value] = self.next_id
            self.id_to_num[self.next_id] = value
            self.next_id += 1
        return self.num_to_id[value]

    def encode(self, text: str) -> Tuple[List[int], List[List[float]]]:
        tokens: List[int] = []
        binary: List[List[float]] = []
        for token in re.findall(r"\d+|\D+", text):
            if token.isdigit():
                val = int(token)
                tid = self._allocate_id(val)
                bin_vec = [float((val >> i) & 1) for i in range(7, -1, -1)] + [1.0]
                tokens.append(tid)
                binary.append(bin_vec)
            else:
                ids = self.enc.encode(token)
                for t in ids:
                    tokens.append(t)
                    binary.append([0.0] * 9)
        return tokens, binary

    def decode(self, tokens: List[int]) -> str:
        pieces = []
        for t in tokens:
            if t in self.id_to_num:
                pieces.append(str(self.id_to_num[t]))
            else:
                pieces.append(self.enc.decode([t]))
        return "".join(pieces)
