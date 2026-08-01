# Local data directories (repo-side)

This tree holds **training corpora you keep on disk for `adapter create`**.  
It is separate from runtime state under `~/.local/share/lpc-llm/` (models, adapters, cache).

```text
data/
  train/          # put your .txt / .jsonl here (gitignored; not published)
  README.md       # this file (tracked)
examples/
  train-sample.txt   # tiny public sample for docs / smoke tests (tracked)
```

## Training input (`--from`)

Recommended:

```bash
# copy or write your corpus (never commit private text)
cp ~/Documents/my-domain.txt data/train/my-domain.txt

lpc-llm adapter create \
  --from data/train/my-domain.txt \
  --out my-lora \
  --base smollm2:360m
```

Formats:

| File | Meaning |
|------|---------|
| `*.txt` | Each non-empty line is one sample |
| `*.jsonl` | Each line: `{"text":"..."}` or a JSON string |

A safe public sample lives at [`examples/train-sample.txt`](../examples/train-sample.txt).

## Training output (backup these)

`adapter create` writes under the **user data dir** (not this repo):

```text
~/.local/share/lpc-llm/adapters/<name>/
  adapter.json
  weights.bin
```

Also registered in `~/.local/share/lpc-llm/manifest.json`.

**Back up `adapters/`** (and optionally `blobs/` if you care about offline re-download).  
`cache/` is regenerable and usually not worth backing up.
