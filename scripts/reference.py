"""Independent numpy reference for a single forward pass of GPT-2.

Used to verify that the Rust implementation produces the same next-token
logits. Usage: python3 scripts/reference.py <token_id> [<token_id> ...]
"""

import json
import struct
import sys

import numpy as np

MODEL = "models/gpt2/model.safetensors"
D, NH, NL, HD, EPS = 768, 12, 12, 64, 1e-5


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(buf[start:end], dtype=np.float32).reshape(
            info["shape"]
        )
    return out


W = load(MODEL)


def ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def forward(ids):
    t = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:t]
    for layer in range(NL):
        p = f"h.{layer}."
        h = ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        # Conv1D weights are stored as [in, out].
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(t, NH, HD).transpose(1, 0, 2)
        k = k.reshape(t, NH, HD).transpose(1, 0, 2)
        v = v.reshape(t, NH, HD).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
        mask = np.tril(np.ones((t, t), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        scores -= scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(-1, keepdims=True)
        o = (probs @ v).transpose(1, 0, 2).reshape(t, D)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return x[-1] @ W["wte.weight"].T


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]]
    logits = forward(ids)
    print("numpy top-5:")
    for i in np.argsort(-logits)[:5]:
        print(f"  {i:6d}  {logits[i]:10.4f}")
