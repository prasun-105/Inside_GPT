"""GPT model shared by training and inference.

The architecture matches ``gpt.py`` exactly, but every size is read from a
``GPTConfig`` instead of module-level globals, so a checkpoint can rebuild the
model on any device without re-running training code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # Same computation as softmax(q k^T / sqrt(d) + causal mask) v from gpt.py,
        # using PyTorch's fused kernel, which applies the causal mask itself.
        q, k, v = self.query(x), self.key(x), self.value(x)
        p = self.dropout.p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(head_size * cfg.n_head, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def trace(model: GPTLanguageModel, idx: torch.Tensor) -> dict:
    """Run one forward pass on a (1, T) window and record what happens to the
    LAST position, i.e. how the next-character prediction is built up.

    Returns plain tensors (on CPU) per stage:
      tok_emb, pos_emb, x0        - the input vector for the last character
      layers[l].attn              - (n_head, T) attention weights of the last position
      layers[l].attn_norm/ffn_norm - how much attention / feed-forward changed the vector
      layers[l].active            - fraction of feed-forward neurons that fired (ReLU > 0)
      layers[l].resid             - the vector after the block
      lens[l]                     - next-char probabilities if the model stopped after
                                    stage l ("logit lens"); l=0 is embeddings only
      logits                      - final logits
    """
    model.eval()
    B, T = idx.shape
    tok = model.token_embedding_table(idx)
    pos = model.position_embedding_table(torch.arange(T, device=idx.device))
    x = tok + pos

    def lens(h):
        return torch.softmax(model.lm_head(model.ln_f(h[:, -1])).float(), dim=-1)[0].cpu()

    out = {
        "tok_emb": tok[0, -1].float().cpu(),
        "pos_emb": pos[-1].float().cpu(),
        "x0": x[0, -1].float().cpu(),
        "lens": [lens(x)],
        "layers": [],
    }
    for block in model.blocks:
        h = block.ln1(x)
        attn = []
        for head in block.sa.heads:
            q = head.query(h)[:, -1]  # (1, hs)
            k = head.key(h)  # (1, T, hs)
            w = (k @ q.unsqueeze(-1)).squeeze(-1) * k.shape[-1] ** -0.5  # (1, T)
            attn.append(torch.softmax(w.float(), dim=-1)[0])
        sa = block.sa(h)
        x = x + sa
        h2 = block.ln2(x)
        pre = block.ffwd.net[0](h2[:, -1])
        ff = block.ffwd(h2)
        x = x + ff
        out["layers"].append({
            "attn": torch.stack(attn).cpu(),
            "attn_norm": float(sa[0, -1].norm()),
            "ffn_norm": float(ff[0, -1].norm()),
            "active": float((pre > 0).float().mean()),
            "resid": x[0, -1].float().cpu(),
        })
        out["lens"].append(lens(x))
    out["logits"] = model.lm_head(model.ln_f(x))[0, -1].float().cpu()
    return out


def sampling_distribution(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Turn a 1-D logit vector into the distribution the sampler draws from."""
    temperature = max(float(temperature), 1e-4)
    logits = logits / temperature
    if top_k and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        # keep the smallest prefix whose mass reaches top_p (always keep the top token)
        drop = (cum - sorted_p) >= top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, sorted_i, sorted_p)
        probs = probs / probs.sum()
    return probs


def entropy_bits(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * torch.log2(p)).sum())


def max_entropy_bits(vocab_size: int) -> float:
    return math.log2(vocab_size)
