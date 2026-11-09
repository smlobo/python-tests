"""Run Qwen3-0.6B-Base using PyTorch, without Transformers."""
# https://huggingface.co/Qwen/Qwen3-0.6B-Base

import json
import unicodedata
from pathlib import Path

import regex
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import constants
from safetensors.torch import load_file

from utilities import bpe, get_device


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
PROMPT = "The process of making tea begins with"
MAX_NEW_TOKENS = 25
MODEL_CACHE = Path(constants.HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}"


def cached_file(filename):
    """Find a file in this model's Hugging Face cache without downloading it."""
    revision = (MODEL_CACHE / "refs" / "main").read_text(encoding="utf-8").strip()
    tree_path = MODEL_CACHE / "trees" / f"{revision}.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    try:
        entry = tree["files"][filename]
    except KeyError as exc:
        raise FileNotFoundError(f"{filename} is not listed in the cached model revision") from exc

    # Large Xet/LFS files use their SHA-256; smaller files use their Git blob ID.
    for content_id in (entry.get("lfs_sha256"), entry.get("blob_id")):
        if content_id:
            path = MODEL_CACHE / "blobs" / content_id
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"{filename} is listed in the model cache, but its blob is missing or incomplete"
    )


def bytes_to_unicode():
    """Return Qwen/GPT-style reversible mapping from bytes to Unicode characters."""
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


class QwenTokenizer:
    """Small byte-level BPE tokenizer for the downloaded Qwen3 tokenizer files."""
    def __init__(self, vocab_file, merges_file, tokenizer_config_file):
        self.encoder = json.loads(vocab_file.read_text(encoding="utf-8"))
        self.decoder = {token_id: token for token, token_id in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {char: byte for byte, char in self.byte_encoder.items()}

        merge_lines = merges_file.read_text(encoding="utf-8").splitlines()[1:]
        self.ranks = {
            tuple(line.split()): rank for rank, line in enumerate(merge_lines) if line.strip()
        }
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))
        added = tokenizer_config.get("added_tokens_decoder", {})
        self.added_tokens = {
            item["content"]: int(token_id) for token_id, item in added.items()
        }
        self.special_ids = {
            int(token_id) for token_id, item in added.items() if item.get("special", False)
        }
        special_strings = sorted(self.added_tokens, key=len, reverse=True)
        self.special_pattern = (
            regex.compile("(" + "|".join(regex.escape(s) for s in special_strings) + ")")
            if special_strings else None
        )

        # This is the Unicode-aware pre-tokenization expression in Qwen's tokenizer.json.
        self.pattern = regex.compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        self.cache = {}

    def _bpe(self, token):
        return bpe(token, self.ranks, self.cache)

    def _encode_text(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        for piece in self.pattern.findall(text):
            byte_encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[token] for token in self._bpe(byte_encoded))
        return ids

    def encode(self, text):
        if self.special_pattern is None:
            return self._encode_text(text)

        ids = []
        pieces = self.special_pattern.split(text)
        for piece in pieces:
            if piece in self.added_tokens:
                ids.append(self.added_tokens[piece])
            else:
                ids.extend(self._encode_text(piece))
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        decoded = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.special_ids:
                continue
            if token_id in self.decoder:
                decoded.append(self.decoder[token_id])
            else:
                token_string = next(
                    (text for text, added_id in self.added_tokens.items() if added_id == token_id),
                    None,
                )
                if token_string is not None:
                    decoded.append(token_string)

        byte_text = "".join(decoded)
        raw = bytearray(self.byte_decoder[char] for char in byte_text)
        return raw.decode("utf-8", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config["rope_theta"]
        eps = config["rms_norm_eps"]

        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

        self.first_time = True

    def forward(self, x, positions):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim)

        if self.first_time:
            print(f'[{id(self)}] Attention before transpose; q: {q.shape}, kv: {k.shape}')
        self.first_time = False

        # Qwen3 normalizes each query/key head before applying rotary positions.
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        frequencies = positions.float().unsqueeze(-1) * inv_freq
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = embedding.sin().to(dtype=q.dtype)[None, None, :, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Grouped-query attention: each K/V head is shared by multiple Q heads.
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(attended)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        intermediate = config["intermediate_size"]
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        eps = config["rms_norm_eps"]
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(width, eps)
        self.post_attention_layernorm = RMSNorm(width, eps)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.first_time = True

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed_tokens(input_ids)
        if self.first_time:
            print(f'Qwen3 shape sent to transformer blocks: {x.shape}')
            self.first_time = False
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        # Qwen3-0.6B-Base ties its output projection to the token embedding matrix.
        return x @ self.embed_tokens.weight.T


def load_model(device):
    config = json.loads(cached_file("config.json").read_text(encoding="utf-8"))
    checkpoint = load_file(str(cached_file("model.safetensors")), device="cpu")
    model = Qwen3(config)
    expected_keys = model.state_dict().keys()
    weights = {
        key.removeprefix("model."): value
        for key, value in checkpoint.items()
        if key.removeprefix("model.") in expected_keys
    }

    # load_state_dict requires model field names and tensor dimensions to match.
    model.load_state_dict(weights)
    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float16)
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()

    print(f'Model:')
    print(f'  layers: {len(model.layers)}')
    print(f'  dimension (hidden_size): {config["hidden_size"]}')
    print(f'  token embedding (vocab_size x width(token_dim)): {model.embed_tokens}')
    print(f'  Attention heads: {model.layers[0].self_attn.num_heads}')
    print(f'  Attention head dim: {model.layers[0].self_attn.head_dim}')
    print(f'  Attention KV heads: {model.layers[0].self_attn.num_key_value_heads}')
    print(f'  Attention query projection: {model.layers[0].self_attn.q_proj}')
    print(f'  Attention kv projection: {model.layers[0].self_attn.k_proj}')
    print(f'  Attention output projection: {model.layers[0].self_attn.o_proj}')
    print(f'  MLP gate/up/down proj: {model.layers[0].mlp.gate_proj}')

    return model, config


def main():
    device = torch.device(get_device())
    print(f"Using device: {device}")

    tokenizer = QwenTokenizer(
        cached_file("vocab.json"),
        cached_file("merges.txt"),
        cached_file("tokenizer_config.json"),
    )
    model, config = load_model(device)
    input_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long, device=device)
    print(f"Initial input_ids shape (batch x tokens): {input_ids.shape}")

    eos_token_id = config["eos_token_id"]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            context = input_ids[:, -config["max_position_embeddings"]:]
            logits = model(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == eos_token_id:
                break

    print(tokenizer.decode(input_ids[0].cpu().tolist()))


if __name__ == "__main__":
    main()
