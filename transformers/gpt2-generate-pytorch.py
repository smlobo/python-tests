"""Run GPT-2 from the Hugging Face cache with PyTorch, without Transformers."""

import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import constants
from safetensors.torch import load_file

from utilities import get_device


PROMPT = "The process of making tea begins with"
# MAX_NEW_TOKENS = 40
MAX_NEW_TOKENS = 21
CACHE_DIR = Path(constants.HF_HUB_CACHE) / "models--openai-community--gpt2"


def cached_file(filename):
    """Resolve a file from this repo's Hugging Face cache without network access."""
    revision = (CACHE_DIR / "refs" / "main").read_text(encoding="utf-8").strip()
    tree_file = CACHE_DIR / "trees" / f"{revision}.json"
    tree = json.loads(tree_file.read_text(encoding="utf-8"))
    try:
        entry = tree["files"][filename]
    except KeyError as exc:
        raise FileNotFoundError(f"{filename} is not listed in the cached GPT-2 revision") from exc

    # LFS files are stored by their content hash; small Git files use blob_id.
    candidates = [entry.get("lfs_sha256"), entry.get("blob_id")]
    for content_id in candidates:
        if content_id:
            path = CACHE_DIR / "blobs" / content_id
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"{filename} is listed in the cached GPT-2 revision, but its blob is missing "
        f"from {CACHE_DIR / 'blobs'}"
    )


def bytes_to_unicode():
    """GPT-2's reversible mapping from bytes to printable Unicode chars."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(visible, chars)}


class GPT2Tokenizer:
    def __init__(self, vocab_file, merges_file):
        self.encoder = json.loads(vocab_file.read_text(encoding="utf-8"))
        self.decoder = {token_id: token for token, token_id in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {char: byte for byte, char in self.byte_encoder.items()}
        merges = merges_file.read_text(encoding="utf-8").splitlines()[1:]
        self.ranks = {tuple(line.split()): rank for rank, line in enumerate(merges)}
        self.cache = {}
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+\s*|\s+(?!\S)|\s+",
            re.IGNORECASE,
        )

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            pair = min(pairs, key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    merged.extend(word[i:j])
                    i = j
                except ValueError:
                    merged.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        self.cache[token] = word
        return word

    def encode(self, text):
        ids = []
        for piece in re.findall(self.pattern, text):
            encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[token] for token in self._bpe(encoded))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[int(token_id)] for token_id in ids)
        raw = bytearray(self.byte_decoder[char] for char in text)
        return raw.decode("utf-8", errors="replace")


class Conv1D(nn.Module):
    """GPT-2 linear layer, whose checkpoint weights are [input, output]."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).view(
            *x.shape[:-1], self.weight.shape[-1]
        )


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.c_attn = Conv1D(width, 3 * width)
        self.c_proj = Conv1D(width, width)
        self.first_time = True

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        head_width = width // self.heads
        shape = (batch, length, self.heads, head_width)
        if self.first_time:
            print(f'[{id(self)}] Attention q, k, v before transpose: {shape}')
        self.first_time = False
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
        # scores = (q @ k.transpose(-2, -1)) / (head_width ** 0.5)
        # scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        # attended = F.softmax(scores, dim=-1) @ v
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.c_proj(attended)


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = Conv1D(width, 4 * width)
        self.c_proj = Conv1D(4 * width, width)

    def forward(self, x):
        hidden = self.c_fc(x)
        hidden = 0.5 * hidden * (1.0 + torch.tanh(
            0.7978845608 * (hidden + 0.044715 * hidden.pow(3))
        ))
        return self.c_proj(hidden)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp = MLP(width)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config["n_embd"]
        # Word token embeddings
        self.wte = nn.Embedding(config["vocab_size"], self.width)
        # Word position embeddings
        self.wpe = nn.Embedding(config["n_positions"], self.width)
        self.h = nn.ModuleList(
            Block(self.width, config["n_head"]) for _ in range(config["n_layer"])
        )
        self.ln_f = nn.LayerNorm(self.width, eps=1e-5)
        self.first_time = True

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)[None, :, :]
        if self.first_time:
            print(f'GPT2 shape sent to transformer blocks: {x.shape}')
        self.first_time = False
        for block in self.h:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T


def load_model(device):
    config = json.loads(cached_file("config.json").read_text(encoding="utf-8"))
    checkpoint = load_file(str(cached_file("model.safetensors")), device="cpu")
    model = GPT2(config)
    expected_keys = model.state_dict().keys()
    weights = {
        key.removeprefix("transformer."): value
        for key, value in checkpoint.items()
        if key.removeprefix("transformer.") in expected_keys
    }
    model.load_state_dict(weights)
    model.to(device).eval()

    print(f'Model:')
    print(f'  layers: {len(model.h)}')
    print(f'  dimension (n_embd): {model.width}')
    print(f'  token embedding (vocab_size x width(token_dim)): {model.wte}')
    print(f'  position embedding (tokens(max) x width(token_dim)): {model.wpe}')
    print(f'  attention heads: {model.h[0].attn.heads}')
    return model


def main():
    device = get_device()
    print(f'Using device: {device}')
    tokenizer = GPT2Tokenizer(cached_file("vocab.json"), cached_file("merges.txt"))
    model = load_model(device)
    input_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long, device=device)
    print(f'Initial input_ids shape (batch x tokens): {input_ids.shape}')

    with torch.inference_mode():
        for index in range(MAX_NEW_TOKENS):
            logits = model(input_ids[:, -1024:])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            # print(f'  [{index}] After cat input_ids shape: {input_ids.shape}')

    print(tokenizer.decode(input_ids[0].cpu().tolist()))


if __name__ == "__main__":
    main()
