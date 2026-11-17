"""Chat server for the frozen GPT.

    python server.py            # then open http://127.0.0.1:8000

Loads ``checkpoints/shakespeare-gpt.pt`` once, with gradients off, and streams
generated characters to the browser as Server-Sent Events. Each event carries
the sampled character together with the probability distribution it was drawn
from, so the UI can show how the output was produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nanogpt import load_frozen
from nanogpt.model import entropy_bits, max_entropy_bits, sampling_distribution, trace

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DEFAULT_CKPT = ROOT / "checkpoints" / "shakespeare-gpt.pt"
TOP_N = 10  # alternatives sent per generated character


def r(x) -> float:
    """Round a probability for the wire: 5 decimals is plenty for display."""
    return round(float(x), 5)


class Turn(BaseModel):
    role: str  # "user" | "model"
    text: str


class GenerateRequest(BaseModel):
    history: list[Turn] = Field(default_factory=list)
    prompt: str = ""
    mode: str = "dialogue"  # "dialogue" | "story" | "continue"
    speaker: str = "ROMEO"
    system: str = ""  # steering text that always stays in the model's context
    narrator: str = "KING RICHARD II"
    temperature: float = Field(0.8, ge=0.05, le=2.5)
    top_k: int = Field(0, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(400, ge=1, le=3000)
    seed: int | None = None


class StepRequest(GenerateRequest):
    """One step of step-by-step generation. The client sends the same request
    every time plus what has been written so far; the server is stateless."""
    generated: str = ""
    step: int = 0


class XrayRequest(BaseModel):
    pinned: str = ""  # system prompt part of the window
    body: str = ""  # everything after it; only the newest characters fit
    target: str = ""  # the character that was actually chosen, if known
    temperature: float = Field(0.8, ge=0.05, le=2.5)
    top_k: int = Field(0, ge=0, le=200)
    top_p: float = Field(1.0, gt=0.0, le=1.0)


# The model has no instruction following: it only continues text. A story is
# started by putting the word into an opening line spoken by one of the
# corpus's natural storytellers, and the model carries on in that voice.
NARRATORS = ["KING RICHARD II", "PROSPERO", "FRIAR LAURENCE"]
STORY_OPENINGS = [
    "Come, sit, and I will tell thee of the {w}:\n",
    "Let us sit upon the ground\nAnd tell sad stories of the {w}.\n",
    "There was a {w}, as I have heard it told,\n",
    "Hear me, and I will tell thee of the {w},\n",
    "Now mark me well; this is the {w}'s story.\n",
]
STORY_MIN_CHARS = 350  # the storyteller keeps the floor at least this long
SYSTEM_MAX_CHARS = 120  # leaves at least half of the 256-char window for the chat


@dataclass
class Prompt:
    pinned: str  # system prompt; always kept at the start of the window
    body: str  # conversation / story opening; oldest text scrolls out first
    dropped: list[str]
    user_text: str  # the user's message as written into the script
    prefill: str = ""  # story opening shown in the reply but not generated
    speaker: str = ""  # narrator for stories


def build_prompt(req: GenerateRequest, tok, rng: random.Random) -> Prompt:
    dropped: list[str] = []

    def clean(s: str) -> str:
        text, d = tok.clean(s)
        dropped.extend(c for c in d if c not in dropped)
        return text

    system = clean(req.system).strip()[:SYSTEM_MAX_CHARS]
    pinned = system + "\n\n" if system else ""
    user = clean(req.prompt)

    if req.mode == "continue":
        return Prompt(pinned, user, dropped, user)

    if req.mode == "story":
        word = " ".join(user.split())[:40] or "crown"
        narrator = req.narrator if req.narrator in NARRATORS else rng.choice(NARRATORS)
        opening = rng.choice(STORY_OPENINGS).format(w=word)
        return Prompt(pinned, f"{narrator}:\n{opening}", dropped, word, opening, narrator)

    # Dialogue mode: the conversation is written as a play script. Each turn is
    # "NAME:\n<line>" separated by blank lines, the format of the whole corpus.
    speaker = clean(req.speaker.strip().upper() or "ROMEO")
    user = user.strip()
    first = user.split("\n", 1)[0]
    if not (first.endswith(":") and first[:-1].replace(" ", "").isalpha()):
        user = f"{speaker}:\n{user}"

    parts = [t.strip() for t in (tok.clean(h.text)[0] for h in req.history) if t.strip()]
    parts.append(user)
    return Prompt(pinned, "\n\n".join(parts) + "\n\n", dropped, user)


def create_app(ckpt_path: Path, device: str) -> FastAPI:
    model, tok, meta = load_frozen(ckpt_path, device)
    block = model.cfg.block_size
    lock = threading.Lock()  # one generation at a time on the GPU
    app = FastAPI(title="NanoGPT Chat")

    @app.get("/api/info")
    def info():
        return {
            "name": "Shakespeare GPT",
            "params": model.num_params(),
            "config": model.cfg.to_dict(),
            "vocab": tok.chars,
            "device": str(device),
            "checkpoint": ckpt_path.name,
            "frozen": not any(p.requires_grad for p in model.parameters()),
            "meta": {k: v for k, v in meta.items()},
            "max_entropy_bits": max_entropy_bits(tok.vocab_size),
            "narrators": NARRATORS,
            "story_min_chars": STORY_MIN_CHARS,
            "system_max_chars": SYSTEM_MAX_CHARS,
        }

    @app.post("/api/step")
    def step(req: StepRequest):
        """Next-character distribution for step-by-step mode, plus one draw from it.

        The draw is seeded by (seed, step) so undoing and redoing a step gives
        the same character, and the story template stays fixed for the reply."""
        seed = req.seed if req.seed is not None else int(time.time_ns() % 2**31)
        prompt = build_prompt(req, tok, random.Random(seed))
        written, _ = tok.clean(req.generated)
        pinned = tok.encode(prompt.pinned)
        room = block - len(pinned)
        text = prompt.body + written
        window = pinned + tok.encode(text[-room:])
        if not window:
            raise HTTPException(400, "Prompt is empty after removing unsupported characters.")
        with lock, torch.inference_mode():
            logits, _ = model(torch.tensor([window], dtype=torch.long, device=device))
            logits = logits[0, -1].float()
            raw = torch.softmax(logits, dim=-1)
            masked = (req.mode == "story" and len(written) < STORY_MIN_CHARS and written[-1:] in ("", "\n"))
            if masked:
                logits[tok.stoi["\n"]] = float("-inf")
            probs = sampling_distribution(logits, req.temperature, req.top_k, req.top_p)
            gen = torch.Generator(device=device)
            gen.manual_seed((seed * 1_000_003 + req.step) % 2**63)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        top_p, top_i = torch.topk(probs, TOP_N)
        return {
            "seed": seed, "step": req.step, "masked_newline": masked,
            "start": {"pinned": prompt.pinned, "context": prompt.body[-room:], "cut": max(0, len(prompt.body) - room),
                      "dropped": prompt.dropped, "user_text": prompt.user_text,
                      "prefill": prompt.prefill, "speaker": prompt.speaker},
            "entropy": round(entropy_bits(probs), 3),
            "candidates": int((probs > 0).sum()),
            "top": [[tok.chars[j], r(p), r(raw[j])] for p, j in zip(top_p.tolist(), top_i.tolist())],
            "draw": {"t": tok.chars[nxt], "p": r(probs[nxt]), "raw_p": r(raw[nxt]),
                     "rank": int((probs > probs[nxt]).sum()) + 1},
        }

    @app.post("/api/xray")
    def xray(req: XrayRequest):
        """Trace one prediction through every stage of the network."""
        pinned_text, _ = tok.clean(req.pinned)
        body, dropped = tok.clean(req.body)
        room = block - len(pinned_text)
        window = pinned_text + body[-room:] if room > 0 else pinned_text[-block:]
        if not window:
            raise HTTPException(400, "Type some text first.")
        idx = torch.tensor([tok.encode(window)], dtype=torch.long, device=device)
        with lock, torch.inference_mode():
            t = trace(model, idx)
        final = t["lens"][-1]
        adjusted = sampling_distribution(t["logits"], req.temperature, req.top_k, req.top_p)
        target = req.target if req.target in tok.stoi else tok.chars[int(final.argmax())]
        tid = tok.stoi[target]

        def top(p, n=5):
            v, i = torch.topk(p, n)
            return [[tok.chars[j], r(x)] for x, j in zip(v.tolist(), i.tolist())]

        def vec(v):
            return [round(float(x), 3) for x in v]

        fv, fi = torch.topk(final, 10)
        return {
            "window": window, "pinned_len": len(pinned_text), "dropped": dropped,
            "last": window[-1], "last_id": tok.stoi[window[-1]], "position": len(window) - 1,
            "target": target, "target_id": tid,
            "vectors": {"tok_emb": vec(t["tok_emb"]), "pos_emb": vec(t["pos_emb"]), "x0": vec(t["x0"])},
            "stages": [{"top": top(p), "target_p": r(p[tid]), "entropy": round(entropy_bits(p), 3)}
                       for p in t["lens"]],
            "layers": [{
                "attn": [[round(float(x), 4) for x in h] for h in L["attn"]],
                "attn_norm": round(L["attn_norm"], 3), "ffn_norm": round(L["ffn_norm"], 3),
                "active": round(L["active"], 4), "resid": vec(L["resid"]),
            } for L in t["layers"]],
            "final": {
                "top": [[tok.chars[j], r(p), r(adjusted[j])] for p, j in zip(fv.tolist(), fi.tolist())],
                "logits_top": [round(float(t["logits"][j]), 3) for j in fi.tolist()],
                "entropy": round(entropy_bits(final), 3),
                "target_raw": r(final[tid]), "target_adjusted": r(adjusted[tid]),
            },
            "settings": {"temperature": req.temperature, "top_k": req.top_k, "top_p": req.top_p},
            "dims": {"n_embd": model.cfg.n_embd, "n_head": model.cfg.n_head, "n_layer": model.cfg.n_layer,
                     "ffn": 4 * model.cfg.n_embd, "vocab": tok.vocab_size, "block": block},
        }

    @app.post("/api/generate")
    async def generate(req: GenerateRequest, request: Request):
        seed = req.seed if req.seed is not None else int(time.time_ns() % 2**31)
        prompt = build_prompt(req, tok, random.Random(seed))
        if not (prompt.pinned + prompt.body):
            raise HTTPException(400, "Prompt is empty after removing unsupported characters.")
        pinned = tok.encode(prompt.pinned)
        room = block - len(pinned)  # window left for the conversation itself
        newline = tok.stoi["\n"]
        stop_event = threading.Event()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(obj):
            loop.call_soon_threadsafe(queue.put_nowait, obj)

        def worker():
            with lock, torch.inference_mode():
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                head = torch.tensor(pinned, dtype=torch.long, device=device)
                tail = torch.tensor(tok.encode(prompt.body), dtype=torch.long, device=device)
                visible = prompt.body[-room:] if len(prompt.body) > room else prompt.body
                emit({"type": "start", "pinned": prompt.pinned, "context": visible,
                      "cut": max(0, len(prompt.body) - room), "dropped": prompt.dropped,
                      "user_text": prompt.user_text, "prefill": prompt.prefill,
                      "speaker": prompt.speaker, "seed": seed})
                out, t0, reason = "", time.perf_counter(), "length"
                for i in range(req.max_new_tokens):
                    if stop_event.is_set():
                        reason = "stopped"
                        break
                    # The system prompt stays pinned at the front; only the
                    # newest `room` characters of everything else fit after it.
                    window = torch.cat([head, tail[-room:]]).unsqueeze(0)
                    logits, _ = model(window)
                    logits = logits[0, -1].float()
                    raw = torch.softmax(logits, dim=-1)
                    if req.mode == "story" and len(out) < STORY_MIN_CHARS and out[-1:] in ("", "\n"):
                        # A blank line would end the storyteller's speech: forbid it for now.
                        logits[newline] = float("-inf")
                    probs = sampling_distribution(logits, req.temperature, req.top_k, req.top_p)
                    nxt = int(torch.multinomial(probs, 1, generator=gen))
                    top_p, top_i = torch.topk(probs, TOP_N)
                    rank = int((probs > probs[nxt]).sum()) + 1
                    ch = tok.chars[nxt]
                    emit({
                        "type": "token", "i": i, "t": ch, "p": r(probs[nxt]),
                        "raw_p": r(raw[nxt]), "rank": rank,
                        "entropy": round(entropy_bits(probs), 3),
                        "candidates": int((probs > 0).sum()),
                        "top": [[tok.chars[j], r(p), r(raw[j])]
                                for p, j in zip(top_p.tolist(), top_i.tolist())],
                    })
                    out += ch
                    tail = torch.cat([tail[-block:], torch.tensor([nxt], device=device)])
                    if out.endswith("\n\n") and out.strip():
                        if req.mode == "dialogue" or (req.mode == "story" and len(out) >= STORY_MIN_CHARS):
                            reason = "end_of_speech"
                            break
                dt = time.perf_counter() - t0
                emit({"type": "done", "reason": reason, "tokens": len(out),
                      "seconds": round(dt, 3), "tokens_per_sec": round(len(out) / max(dt, 1e-9), 1)})
            emit(None)

        threading.Thread(target=worker, daemon=True).start()

        async def stream():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
                    if await request.is_disconnected():
                        stop_event.set()
            finally:
                stop_event.set()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="open the chat in the default browser")
    a = p.parse_args()
    if not a.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {a.checkpoint}. Run `python train.py` first.")
    app = create_app(a.checkpoint, a.device)
    url = f"http://{a.host}:{a.port}"
    print(f"NanoGPT chat on {url}  (device: {a.device}). Press Ctrl+C to stop.")
    if a.open:
        import webbrowser
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
