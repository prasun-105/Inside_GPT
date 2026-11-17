"""Train the character-level GPT and save frozen weights.

    python train.py                     # defaults match gpt.py (5000 steps)
    python train.py --max-iters 2000    # shorter run

The checkpoint with the lowest validation loss is written to
``checkpoints/shakespeare-gpt.pt`` and is what ``server.py`` serves.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from nanogpt import CharTokenizer, GPTConfig, GPTLanguageModel, save_checkpoint

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "input.txt")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "shakespeare-gpt.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(step: int, a) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    progress = (step - a.warmup) / max(1, a.max_iters - a.warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    text = a.data.read_text(encoding="utf-8")
    tok = CharTokenizer(sorted(set(text)))
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - a.block_size, (a.batch_size,))
        x = torch.stack([d[i : i + a.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + a.block_size + 1] for i in ix])
        return x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)

    cfg = GPTConfig(tok.vocab_size, a.block_size, a.n_embd, a.n_head, a.n_layer, a.dropout)
    model = GPTLanguageModel(cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.1)

    use_bf16 = a.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(a.eval_iters)
            for k in range(a.eval_iters):
                with autocast:
                    _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    print(f"device={a.device} bf16={use_bf16} params={model.num_params() / 1e6:.2f}M vocab={tok.vocab_size}")
    history, best_val, best_step = [], float("inf"), -1
    t0 = time.time()
    for step in range(a.max_iters + 1):
        if step % a.eval_interval == 0 or step == a.max_iters:
            losses = estimate_loss()
            history.append({"step": step, **losses})
            mark = ""
            if losses["val"] < best_val:
                best_val, best_step = losses["val"], step
                save_checkpoint(a.out, model, tok, {
                    "step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                    "history": history, "corpus": a.data.name, "params": model.num_params(),
                    "trained_on": a.device, "train_seconds": round(time.time() - t0, 1),
                })
                mark = "  * saved"
            print(f"step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"lr {lr_at(step, a):.2e}  {time.time() - t0:6.0f}s{mark}", flush=True)
        if step == a.max_iters:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        xb, yb = get_batch("train")
        with autocast:
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Keep the full curve in the saved checkpoint even if best was earlier.
    ckpt = torch.load(a.out, map_location="cpu", weights_only=True)
    ckpt["meta"]["history"] = history
    ckpt["meta"]["train_seconds"] = round(time.time() - t0, 1)
    torch.save(ckpt, a.out)
    (a.out.with_suffix(".json")).write_text(json.dumps(ckpt["meta"], indent=2))
    print(f"done. best val {best_val:.4f} at step {best_step} -> {a.out}")


if __name__ == "__main__":
    main()
