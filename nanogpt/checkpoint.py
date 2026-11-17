"""Save and load frozen model checkpoints.

A checkpoint is a single ``.pt`` file that contains everything needed to run
the model: config, character vocabulary, weights and training metadata. Loading
puts the model in eval mode and turns off gradients, so the weights are frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import GPTConfig, GPTLanguageModel

FORMAT_VERSION = 1


class CharTokenizer:
    # Characters people commonly type that the corpus spells differently.
    _FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                           "—": "-", "–": "-", "…": "...", "\t": " ", "\r": None})

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def clean(self, s: str) -> tuple[str, list[str]]:
        """Fold or drop characters outside the vocabulary. Returns (text, dropped)."""
        s = s.translate(self._FOLD)
        kept, dropped = [], []
        for c in s:
            if c in self.stoi:
                kept.append(c)
            elif c not in dropped:
                dropped.append(c)
        return "".join(kept), dropped


def save_checkpoint(path: Path, model: GPTLanguageModel, tokenizer: CharTokenizer, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": model.cfg.to_dict(),
            "chars": tokenizer.chars,
            "state_dict": state,
            "meta": meta,
        },
        path,
    )


def load_frozen(path: Path, device: str | torch.device = "cpu"):
    """Load a checkpoint as a frozen, inference-only model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    cfg.dropout = 0.0  # no dropout at inference
    model = GPTLanguageModel(cfg)
    # Older checkpoints also stored the causal-mask buffer; it is no longer needed.
    state = {k: v for k, v in ckpt["state_dict"].items() if not k.endswith(".tril")}
    model.load_state_dict(state)
    model.eval().requires_grad_(False).to(device)
    return model, CharTokenizer(ckpt["chars"]), ckpt.get("meta", {})
