"""Small helpers shared by PyTorch experiments."""

import torch


def bpe(token, ranks, cache):
    """Apply byte-pair merges to a pre-tokenized string and cache the result."""
    if token in cache:
        return cache[token]

    word = tuple(token)
    while len(word) > 1:
        pairs = set(zip(word, word[1:]))
        pair = min(pairs, key=lambda candidate: ranks.get(candidate, float("inf")))
        if pair not in ranks:
            break

        first, second = pair
        merged = []
        index = 0
        while index < len(word):
            try:
                next_index = word.index(first, index)
                merged.extend(word[index:next_index])
                index = next_index
            except ValueError:
                merged.extend(word[index:])
                break

            if index + 1 < len(word) and word[index:index + 2] == pair:
                merged.append(first + second)
                index += 2
            else:
                merged.append(word[index])
                index += 1
        word = tuple(merged)

    cache[token] = word
    return word


def get_device() -> str:
    """Return the best available device name: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
