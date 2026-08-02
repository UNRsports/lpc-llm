# Local data directories (repo-side)

**Privacy rule:** private training corpora and runtime state belong under the
user data tree from `config_lpcllm` (default `~/.local/share/lpc-llm/`), not in
this git repository. `.gitignore` under `data/train/` is only a safety net.

```text
~/.local/share/lpc-llm/          # paths.data_dir (see lpc-llm config show)
  train/                         # paths.train_dir — private .txt / .jsonl
  blobs/  adapters/  cache/ …

data/train/                      # optional: empty / gitignored; do NOT store private data here
examples/train-sample.txt        # public sample (tracked)
examples/pref-sample.jsonl       # public DPO sample (tracked)
config_lpcllm.example            # path / install template
```

Write a user config once:

```bash
lpc-llm config init
lpc-llm config show
```

## Training input (`--from`)

Recommended (private corpora under home):

```bash
# place corpus under train_dir (created automatically)
cp ~/Documents/my-domain.txt "$(lpc-llm config get train_dir)/my-domain.txt"

lpc-llm adapter create \
  --from my-domain.txt \
  --out my-lora \
  --base smollm2:360m
```

`--from` accepts:

1. An absolute / cwd-relative path to an existing file
2. A bare filename resolved under `train_dir`

Formats:

| File | Meaning |
|------|---------|
| `*.txt` | Each non-empty line is one sample |
| `*.jsonl` | Each line: `{"text":"..."}` or a JSON string |

Public smoke-test samples (safe to commit):

- [`examples/train-sample.txt`](../examples/train-sample.txt)
- [`examples/pref-sample.jsonl`](../examples/pref-sample.jsonl) — `{"prompt","chosen","rejected"}` for `lpc-llm train dpo`

Optional Documents layout (edit `~/.config/lpc-llm/config_lpcllm`):

```toml
[paths]
train_dir = "~/Documents/lpc-llm/train"
```

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

## Binary install vs user data

| Install | Script | Binary | User data |
|---------|--------|--------|-----------|
| Dev (symlink) | `scripts/install-dev.sh` | `~/.local/bin` (or `install.bin_dir`) | per-user `data_dir` |
| Shared system | `scripts/install-system.sh` | `/usr/local/bin` (binary only) | still per-user `data_dir` |

Never put private corpora in the repo tree before `git push`.
