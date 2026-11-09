"""Run Falcon3-1B-Base from the Hugging Face cache without Transformers."""
# https://huggingface.co/tiiuae/Falcon3-1B-Base

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import constants
from safetensors.torch import load_file
from tokenizers import Tokenizer

from utilities import get_device


MODEL_ID = "tiiuae/Falcon3-1B-Base"
PROMPT = "The process of making tea begins with"
MAX_NEW_TOKENS = 32
MODEL_CACHE = Path(constants.HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}"


def cached_file(filename):
    """Find a model file in the Hugging Face cache without downloading it."""
    try:
        revision = (MODEL_CACHE / "refs" / "main").read_text(encoding="utf-8").strip()
        tree_path = MODEL_CACHE / "trees" / f"{revision}.json"
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        entry = tree["files"][filename]
    except (FileNotFoundError, KeyError) as exc:
        raise FileNotFoundError(
            f"{filename} is not present in the cached {MODEL_ID} revision; "
            f"download it with: hf download {MODEL_ID} {filename}"
        ) from exc

    # Xet/LFS files are stored by SHA-256; smaller files use their Git blob ID.
    for content_id in (entry.get("lfs_sha256"), entry.get("blob_id")):
        if content_id:
            path = MODEL_CACHE / "blobs" / content_id
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"{filename} is listed in the cached model revision, but its blob is missing"
    )


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
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=False)
        self.first_time = True

    def forward(self, x, positions):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim)
        if self.first_time:
            print(f'[{id(self)}] Attention before transpose; q: {q.shape}, kv: {k.shape}')
            self.first_time = False

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
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

        # Falcon3 uses grouped-query attention: K/V heads serve multiple Q heads.
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
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


class Falcon3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        # separate out projection
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )
        self.first_time = True

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed_tokens(input_ids)
        if self.first_time:
            print(f"Falcon3 shape sent to transformer blocks: {x.shape}")
            self.first_time = False
        for layer in self.layers:
            x = layer(x, positions)
        return self.lm_head(self.norm(x))


def load_model(device):
    config = json.loads(cached_file("config.json").read_text(encoding="utf-8"))
    checkpoint = load_file(str(cached_file("model.safetensors")), device="cpu")
    model = Falcon3(config)
    # The Transformers checkpoint nests the backbone under "model."; this
    # standalone module exposes those backbone fields directly.
    weights = {
        key.removeprefix("model."): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(weights)

    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float16)
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()

    attention = model.layers[0].self_attn
    print("Model:")
    print(f"  layers: {len(model.layers)}")
    print(f"  dimension (hidden_size): {config['hidden_size']}")
    print(f"  token embedding (vocab_size x width(token_dim)): {model.embed_tokens}")
    print(f"  out proj (width(token_dim) x vocab_size): {model.lm_head}")
    print(f"  Attention heads: {attention.num_heads}")
    print(f"  Attention head dim: {attention.head_dim}")
    print(f"  Attention KV heads: {attention.num_key_value_heads}")
    print(f"  Attention query projection: {attention.q_proj}")
    print(f"  Attention kv projection: {attention.k_proj}, {attention.v_proj}")
    print(f"  Attention output projection: {attention.o_proj}")
    print(f'  MLP gate/up/down proj: {model.layers[0].mlp.gate_proj}')
    return model, config


def main():
    device = torch.device(get_device())
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(str(cached_file("tokenizer.json")))
    model, config = load_model(device)
    encoded = tokenizer.encode(PROMPT)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    print(f"Initial input_ids shape (batch x tokens): {input_ids.shape}")

    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            context = input_ids[:, -config["max_position_embeddings"]:]
            logits = model(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == config["eos_token_id"]:
                break

    print(tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
