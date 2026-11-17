# Shakespeare GPT

A 10.8M-parameter, character-level GPT built from scratch in PyTorch and trained on Shakespeare. It comes with a local chat interface that shows the probabilities behind every generated character and how the prediction forms inside the network.

Trained weights are included (`checkpoints/shakespeare-gpt.pt`, validation loss 1.475), so you can chat without training.

## Features

- **Chat modes:** *Dialogue* (you speak as a character and the model answers as another), *Story* (one word becomes a tale told by King Richard II, Prospero or Friar Laurence), and *Continue* (plain text continuation).
- **Probability view:** each character is tinted by its probability. Hover or click one to see the top candidates, the entropy and its rank.
- **Step-by-step mode:** sample the reply one character at a time, pick candidates yourself, and change the temperature between draws.
- **Model X-ray:** follow one prediction through the embeddings and all 6 layers: where each attention head looked, how many neurons fired, and the best guess after each layer (logit lens).
- **System prompt:** steering text that always stays in the 256-character context window.
- **Glossary:** plain-English definitions of every term used in the app.

## Requirements

- Python 3.10+
- An NVIDIA GPU is optional. Everything also runs on CPU.

## Setup

```bash
git clone https://github.com/xenonnn4w/shakespeare-gpt.git
cd shakespeare-gpt
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

Install dependencies. For an NVIDIA GPU, install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # optional, NVIDIA GPU only
pip install -r requirements.txt
```

## Run

```bash
python server.py --open
```

This opens http://127.0.0.1:8000. You can also double-click `chat.bat` (Windows) or run `./chat.sh` (macOS / Linux). Use `--device cpu` to force CPU.

## Train your own weights

```bash
python train.py
```

This trains for 5,000 steps (~30 min on an RTX 4050) and saves the checkpoint with the lowest validation loss to `checkpoints/shakespeare-gpt.pt`. Run `python train.py --help` for the options.

## Project layout

| Path | Contents |
| --- | --- |
| `nanogpt/` | Model, tokenizer, checkpoint loading, prediction tracing |
| `train.py` | Training script |
| `server.py` | FastAPI server: streaming generation, step sampling, X-ray |
| `web/` | Chat interface (HTML, CSS, JavaScript; no build step) |
| `checkpoints/` | Trained weights and training metadata |
| `gpt.py`, `biagram.py` | Original single-file transformer and bigram baseline |
| `notebooks/GPT.ipynb` | Development notebook |
| `input.txt` | Training corpus (~1 MB of Shakespeare) |

## Troubleshooting

- **`WinError 4551 ... An Application Control policy has blocked this file`**: Windows Smart App Control is blocking PyTorch. Turn it off in Windows Security → App & browser control → Smart App Control.
- **Port 8000 in use**: run `python server.py --open --port 8001`.
