# lpc-llm

## Contents / 目次

1. [English](#english)
   1. [Current specification](#1-current-specification)
   2. [Manual (setup → start → stop)](#2-manual-setup--start--stop)
   3. [Command reference](#3-command-reference)
   4. [Troubleshooting](#4-troubleshooting)
   5. [Development notes](#5-development-notes)
   6. [Roadmap / planned work](#6-roadmap--planned-work)
   7. [License](#7-license)
2. [日本語](#日本語)
   1. [現状の仕様](#1-現状の仕様)
   2. [マニュアル（導入〜起動〜停止）](#2-マニュアル導入起動停止)
   3. [コマンドリファレンス](#3-コマンドリファレンス)
   4. [トラブルシューティング](#4-トラブルシューティング)
   5. [開発メモ](#5-開発メモ)
   6. [今後の工程予定](#6-今後の工程予定)
   7. [ライセンス](#7-ライセンス)

---

# English

Ollama-free **pure-Rust local LLM player**.  
It runs quantized GGUF with Candle, and on the hybrid path streams weights via per-layer pack layout plus `io_uring` / `O_DIRECT` double buffering.

- **Inference engine**: in-house (Candle + hybrid I/O). No Ollama / llama.cpp binaries
- **CUI**: Ollama-like `list` / `pull` / `run` / `rm` / `show` / `adapter`
- **Storage**: model blobs and engine-derived cache are separated. Engine upgrades do not force re-download
- **LoRA**: train with `adapter create` (private corpora under `train_dir` from `config_lpcllm`); results under `adapters/` (**backup**)
- **Paths**: `config_lpcllm` sets binary install dir + per-user data/train dirs (`lpc-llm config show`)
- **Model creation**: `train scratch|sft|dpo` (tiny from-scratch → GGUF → `run`); scale-up via `job`
- **Roadmap**: progress and planned phases are tracked in [`todo.md`](todo.md)

## English table of contents

1. [Current specification](#1-current-specification)
2. [Manual (setup → start → stop)](#2-manual-setup--start--stop)
3. [Command reference](#3-command-reference)
4. [Troubleshooting](#4-troubleshooting)
5. [Development notes](#5-development-notes)
6. [Roadmap / planned work](#6-roadmap--planned-work)
7. [License](#7-license)

### Paths at a glance

Configured by **`config_lpcllm`** (`lpc-llm config init` / `config show`). Defaults below.

| What | Where (default) | Git? | Backup? |
|------|-------|------|---------|
| Build output | `./target/debug/lpc-llm` or `./target/release/lpc-llm` | ignored (`/target/`) | no |
| Dev PATH install | `~/.local/bin/lpc-llm` → symlink (`install-dev.sh`) | n/a | no |
| System binary (shared) | `/usr/local/bin/lpc-llm` (`install-system.sh`; **binary only**) | n/a | no |
| Config file | `~/.config/lpc-llm/config_lpcllm` (+ optional `/etc/lpc-llm/…`) | n/a | optional |
| Base models (GGUF) | `~/.local/share/lpc-llm/blobs/` | outside repo | optional (re-downloadable) |
| Engine packs | `~/.local/share/lpc-llm/cache/packs/` | outside repo | no (regenerable) |
| **Training corpora** | **`~/.local/share/lpc-llm/train/`** (`paths.train_dir`) | outside repo | your choice |
| Public / dev samples | `examples/*.txt` / `*.jsonl` | tracked | n/a |
| **Trained LoRA adapters** | **`~/.local/share/lpc-llm/adapters/<name>/`** | outside repo | **yes — back these up** |

Private data must not live in the git tree. Repo `data/train/` is gitignored as a safety net only. See [`data/README.md`](data/README.md) and [`config_lpcllm.example`](config_lpcllm.example).

---

## 1. Current specification

### Architecture overview

```text
┌─ CUI (clap) ─────────────────────────────────────┐
│  list / pull / run / rm / show / prefetch / io     │
└───────────────────────────┬───────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
   Eager path                              Hybrid path
   (Candle full load)                      (default: gemma*)
        │                                       │
        │              ┌────────────────────────┤
        │              ▼                        ▼
        │     ensure_packed (engine cache)   hot layers pin
        │              │                        │
        │              ▼                        ▼
        │     io_uring ping-pong DMA      resident RAM
        │              │                        │
        └──────────────┴──────────┬─────────────┘
                                  ▼
                         forward (arch branch)
                         Gemma2: post-norm / √emb /
                         Neox RoPE / softcap / GeLU
```

### Memory / I/O model (hybrid)

```text
[ embeddings + norm + lm_head ]     resident (read directly from blobs)
[ hot layers 0 .. H-1 ]             pinned in RAM (within budget, default max 8)
[ Prefetch A | Prefetch B ]         2× pack-layer DMA (io_uring)
[ KV cache ]                        grows with context
```

Tuning levers (perceived impact):

| Lever | Impact | Implementation |
|------|------|------|
| Pack layout + double buffer | High | `{gguf} → cache/.../layers.pack`, 1 DMA per layer |
| Hot resident ratio | Medium–high | `--ram-mib` / `--hot-layers` |
| Chunked thinking (short first reply) | Medium | `--burst`, REPL `/more` |
| Chunk-size micro-tuning | Low | DMA window align at pack time, I/O wait EMA |

### Data layout (module separation)

**Runtime state** (from `config_lpcllm` / XDG — outside the git repo):

```text
~/.local/share/lpc-llm/          # paths.data_dir
  blobs/                 # model module (durable GGUF + tokenizer)
    <hf-repo--name>/
      *.gguf
      tokenizer.json
  adapters/              # LoRA / diff modules (durable) ← BACK UP
    <name>/
      adapter.json
      weights.bin
  train/                 # paths.train_dir — private corpora ← not in git
  cache/                 # engine module (regenerable)
    packs/
      <model_name>/<engine_ver>/
        layers.pack
        layers.pack.json
  manifest.json          # soft index (models + adapters)
```

**Public samples in the repo** (safe to commit):

```text
examples/train-sample.txt
examples/pref-sample.jsonl
config_lpcllm.example
```

| Area | Safe to delete? | Notes |
|------|-------------|------|
| `blobs/` | Keep by default | Deleting forces re-download |
| `adapters/` | **Keep / back up** | Trained LoRA; losing them means re-train |
| `train/` | Your data | Private corpora under home; never in git |
| `cache/` | Yes | Regenerated on next hybrid / train run |
| `manifest.json` | Yes | Restored by `reconcile` at startup |

`rm` only removes the registry entry; it **does not delete blobs**.

### Catalog models

| Name | Contents | Approx. size | hybrid |
|------|------|------------|--------|
| `smollm2:360m` | SmolLM2 360M Instruct Q4_K_M | ~260 MB | enable with `--hybrid` |
| `gemma2:2b` | Gemma 2 2B Instruct Q4_K_M | ~1.7 GB | **hybrid by default** |
| `qwen2.5:1.5b` | Qwen2.5 1.5B Instruct Q4_K_M | ~1.1 GB | enable with `--hybrid` |
| `phi3:mini` | Phi-3 Mini 4K Instruct Q4_K_M | ~2.2 GB | enable with `--hybrid` |

For Gemma2, post-attention / post-ffw norms, embedding √hidden scale, Neox RoPE, attn/final logit softcap, and GeLU are implemented.  
GGUF RMSNorm weights are already `(1+δ)` after HF→GGUF conversion, so runtime multiplies by `w` as-is (no double application).

### Runtime assumptions

- OS: Linux (`io_uring` / `O_DIRECT`; e.g. Fedora)
- CPU inference (currently)
- Rust toolchain (relatively new stable for `edition = "2024"`)
- Download: system `curl` or `wget` (avoids OpenSSL linking)
- Optional: `HF_TOKEN` (gated HF repos)

---

## 2. Manual (setup → start → stop)

### 0. Prerequisites

```bash
rustc --version    # newer stable recommended
curl --version     # or wget
```

You already have the repository (e.g. `~/dev/lpc-llm`).

### 1. Build

```bash
cd ~/dev/lpc-llm

# Cursor / some environments point CARGO_TARGET_DIR elsewhere;
# unset it if you want artifacts under local ./target
unset CARGO_TARGET_DIR

# Day-to-day development (PATH symlink tracks this binary)
cargo build
./scripts/install-dev.sh --no-build   # once: ~/.local/bin/lpc-llm → target/debug/lpc-llm

# Or a release binary
cargo build --release
# → ./target/release/lpc-llm
```

Linker `warning: linker stderr: ignoring deprecated...` can be ignored.

| Goal | Command | Binary |
|------|---------|--------|
| Dev (updates on every `cargo build`) | `./scripts/install-dev.sh` then `cargo build` | `lpc-llm` on PATH (symlink; `install.bin_dir`) |
| Shared system binary | `./scripts/install-system.sh` | `/usr/local/bin` (**binary only**; data stays per-user) |
| Copied release install | `cargo install --path . --force` | `~/.cargo/bin/lpc-llm` (does **not** refresh on compile) |
| No install | `./target/debug/lpc-llm` or `./target/release/lpc-llm` | path as given |
| Path config | `lpc-llm config init` / `config show` | writes/reads `config_lpcllm` |

### 2. Install a model (pull)

List:

```bash
lpc-llm list
```

Fetch (skips re-download if blobs already exist):

```bash
lpc-llm pull smollm2:360m    # smoke / light
lpc-llm pull gemma2:2b
```

Success example (reuse):

```text
· gemma2:2b already in model module — reusing blobs (no download)
  model     ~/.local/share/lpc-llm/blobs/.../gemma-2-2b-it-Q4_K_M.gguf
  tokenizer ~/.local/share/lpc-llm/blobs/.../tokenizer.json
```

Inspect:

```bash
lpc-llm show gemma2:2b
```

If a gated model fails:

```bash
export HF_TOKEN=hf_xxxxxxxx
lpc-llm pull gemma2:2b
```

### 3. Start (chat / LLM use)

```bash
lpc-llm run gemma2:2b
lpc-llm run gemma2:2b --hybrid --ram-mib 4096 --burst 24
lpc-llm run smollm2:360m
lpc-llm run smollm2:360m --adapter my-lora
lpc-llm run gemma2:2b --agent
```

Omit the name to pick from an interactive menu:

```bash
lpc-llm run
lpc-llm          # menu
```

First hybrid run builds the pack (may take minutes; GGUF is not modified):

```text
packing 26 layers → ~/.local/share/lpc-llm/cache/packs/gemma2_2b/0.1.0/layers.pack
…
✓ ready on CPU+pack+io_uring (gemma2)
>>>
```

`mlock failed ... using unlocked arenas` is a warning; inference continues (raise `ulimit -l` if needed).

### 4. Training data placement

Put **private** corpora under `train_dir` from `config_lpcllm` (default `~/.local/share/lpc-llm/train/` — outside git):

```bash
lpc-llm config init          # once: ~/.config/lpc-llm/config_lpcllm
TRAIN="$(lpc-llm config get train_dir)"
mkdir -p "$TRAIN"
cp /path/to/your-corpus.txt "$TRAIN/my-domain.txt"
# JSONL also OK: each line {"text":"..."}
```

Public smoke sample: [`examples/train-sample.txt`](examples/train-sample.txt).  
Details: [`data/README.md`](data/README.md).

### 5. Train a LoRA adapter

```bash
lpc-llm adapter list

# bare name resolves under train_dir
lpc-llm adapter create \
  --from my-domain.txt \
  --out my-lora \
  --base smollm2:360m \
  --steps 64 --rank 8 --last-layers 4

# short smoke with the public sample
lpc-llm adapter create \
  --from examples/train-sample.txt \
  --out smoke-lora \
  --base smollm2:360m \
  --steps 8 --rank 4 --last-layers 2
```

CPU training prints progress every step (each step can take tens of seconds).

### 6. Training results location (backup)

```text
~/.local/share/lpc-llm/adapters/<out>/
  adapter.json
  weights.bin
```

```bash
lpc-llm adapter list
ls ~/.local/share/lpc-llm/adapters/my-lora/
lpc-llm run smollm2:360m --adapter my-lora
```

**Back up `~/.local/share/lpc-llm/adapters/`** for trained deltas.  
`blobs/` is optional (re-downloadable); `cache/` is regenerable.

### 6b. Tiny from-scratch / SFT / DPO (Phase 5)

```bash
# Train a tiny Llama-family model, export F16 GGUF, register for `run`
lpc-llm train scratch \
  --from examples/train-sample.txt \
  --out tiny:demo \
  --steps 64 --n-embd 128 --n-layers 2 --ram-mib 1024

lpc-llm run tiny:demo

# Full fine-tune a tiny checkpoint
lpc-llm train sft --ckpt tiny:demo --from examples/train-sample.txt --out tiny:sft --steps 32

# Preference optimization (JSONL: prompt / chosen / rejected)
lpc-llm train dpo --ckpt tiny:sft --from examples/pref-sample.jsonl --out tiny:dpo --steps 32

# Export only
lpc-llm train export --ckpt tiny:dpo --register --name tiny:dpo
```

Checkpoints live under `~/.local/share/lpc-llm/cache/train/<name>/`  
(`config.json`, `weights.bin`, `tokenizer.json`, optional `model.gguf`).

### 6c. Scale-up jobs (Phase 6)

```bash
lpc-llm job init --template rlhf --out job.json
lpc-llm job run --config job.json --local
lpc-llm job status rlhf-demo

# Import an external GGUF
lpc-llm job import --gguf /path/model.gguf --tokenizer /path/tokenizer.json --name my:import

# Convert: builtin (Phase 5 ckpt) or external ($LPC_LLM_CONVERT_CMD)
lpc-llm job convert --from-dir ~/.local/share/lpc-llm/cache/train/tiny_demo \
  --name tiny:from-ckpt --backend builtin
```

`job.json` can set `remote.launch` / `status_cmd` for cluster hand-off.  
Multi-billion HF→GGUF: set `LPC_LLM_CONVERT_CMD` to write a `.gguf` into `$LPC_LLM_CONVERT_OUT`.

### 7. In-chat controls

| Input | Action |
|------|------|
| Normal text | Send to model; stream tokens |
| `/more` | Continue generating the last reply |
| `/clear` | Clear history and KV |
| `/bye` `/exit` `/quit` | Leave chat |

The first reply is capped by `--burst` (default 24 tokens); use `/more` for more.

### 8. Stop

- **In session**: type `/bye` (preferred)
- **Force quit**: `Ctrl+C` in the terminal
- If left in the background:

```bash
pkill -f 'lpc-llm run'    # only when needed
```

There is no daemon. Stopping the process stops inference. Model files and adapters remain on disk.

### 9. Typical daily flow (shortest)

```bash
cd ~/dev/lpc-llm
unset CARGO_TARGET_DIR
cargo build                         # refreshes PATH binary after install-dev.sh
lpc-llm pull smollm2:360m           # first time
lpc-llm run smollm2:360m
# … chat …
# >>> /bye
```

### 10. (Optional) I/O bench

```bash
lpc-llm prefetch gemma2:2b
lpc-llm io --help
```

---

## 3. Command reference

| Command | Description |
|----------|------|
| `lpc-llm` | Interactive menu |
| `lpc-llm list` | Catalog and local / available |
| `lpc-llm pull <name>` | Fetch into blobs (reuse if present) |
| `lpc-llm run [name] [options]` | Start chat |
| `lpc-llm show <name>` | Catalog + local paths |
| `lpc-llm rm <name>` | Remove from registry (blobs kept) |
| `lpc-llm adapter list` | List LoRA adapters |
| `lpc-llm config show\|init\|get\|example` | Resolve / write `config_lpcllm` (data_dir, train_dir, bin_dir) |
| `lpc-llm adapter create …` | Train LoRA from `--from` → `adapters/<out>/` |
| `lpc-llm adapter install-demo` | Zero LoRA fixture for tests |
| `lpc-llm train scratch\|sft\|dpo\|export` | Tiny from-scratch / full SFT / DPO / GGUF (Phase 5) |
| `lpc-llm job init\|run\|status\|import\|convert` | Declarative jobs / remote bridge (Phase 6) |
| `lpc-llm prefetch <name>` | Pack + io_uring ping-pong timing |
| `lpc-llm io` | I/O demo with synthetic weights |

### `run` options

| Option | Default | Meaning |
|------------|------|------|
| `--pull` | off | Pull without prompt if missing |
| `--hybrid` | on for gemma* | Layer-streaming inference |
| `--hot-layers N` | auto | Force number of RAM-resident layers |
| `--ram-mib N` | 4096 | Soft budget for hot layers + 2 slots (MiB) |
| `--burst N` | 24 | Max tokens for the first reply |
| `--adapter <name>` | none | Bind LoRA side-path (forces hybrid) |
| `--agent` | off | SmolLM2 router before main (exclusive RAM) |
| `--agent-model` | `smollm2:360m` | Router model for `--agent` |

### `adapter create` options

| Option | Default | Meaning |
|--------|---------|---------|
| `--from <path>` | required | Training `.txt` / `.jsonl` (cwd path or name under `train_dir`) |
| `--out <name>` | required | Adapter directory name under `adapters/` |
| `--base <model>` | required | Catalog base (e.g. `smollm2:360m`) |
| `--rank` | 8 | LoRA rank |
| `--alpha` | 16 | LoRA α |
| `--steps` | 64 | AdamW steps |
| `--lr` | 1e-3 | Learning rate |
| `--max-seq` | 128 | Tokens per chunk |
| `--last-layers N` | 0 (all) | Train only last N layers |
| `--ram-mib` | 4096 | Soft RAM budget for trainer load |
| `--pull` | off | Pull base without confirmation |
---

## 4. Troubleshooting

| Symptom | Fix |
|------|------|
| Garbage like `Jove Jove…` | Possibly an old binary. `unset CARGO_TARGET_DIR && cargo build --release`, then use `./target/release/lpc-llm` |
| `mlock failed` | Warning only. Optionally `ulimit -l unlimited` (depends on privileges) |
| Downloads every time | Check `~/.local/share/lpc-llm/blobs`. Migrating from old `~/.local/share/l3m`: rename / symlink |
| Pack is slow | First run only. Delete `cache/packs` to regenerate |
| `--from file not found` | Place corpora under `train_dir` (`lpc-llm config get train_dir`) or pass a real path. See [`data/README.md`](data/README.md) |
| `adapter … not found` | Run `adapter create` successfully first, or `adapter list` / check `~/.local/share/lpc-llm/adapters/` |
| HF 401 | `HF_TOKEN` and license acceptance |
| Long builds | release + LTO. Warnings alone are not failure |

Old data migration example:

```bash
# when the new path does not exist yet
mv ~/.local/share/l3m ~/.local/share/lpc-llm
```

---

## 5. Development notes

- Language: Rust 2024
- Main crates: `candle-core` / `candle-nn` / `candle-transformers` / `tokenizers` / `io-uring` / `memmap2`
- Binary name: `lpc-llm` (`Cargo.toml` package name)
- Relation to Ollama: **independent**. Only the CUI feel is similar
- License: Apache-2.0 (`LICENSE` / `Cargo.toml`)

### Dev binary on PATH

Once per machine / clone:

```bash
unset CARGO_TARGET_DIR   # keep artifacts in ./target (symlink target)
./scripts/install-dev.sh
```

After that, day-to-day:

```bash
cargo build             # refreshes target/debug/lpc-llm → PATH `lpc-llm`
cargo check             # typecheck only (does not rewrite the binary)
lpc-llm adapter list    # always the latest debug build
```

Track release instead: `./scripts/install-dev.sh --release`.

```bash
cargo check
cargo build --release
```

---

## 6. Roadmap / planned work

Implementation progress (MoE, delta adapters, lightweight agent, and later phases) and the recommended next steps are maintained in:

- **[`todo.md`](todo.md)** — extension roadmap (English / Japanese)

See especially the summary, engineering checklist (Phases 4–8), and recommended next steps in that document.

---

## 7. License

[Apache License 2.0](LICENSE)

---

# 日本語

Ollama に依存しない、**純 Rust のローカル LLM プレイヤー**です。  
量子化 GGUF を Candle で推論し、ハイブリッド経路では層ごとの pack 再配置 + `io_uring` / `O_DIRECT` ダブルバッファで重みをストリーミングします。

- **推論エンジン**: 自前（Candle + hybrid I/O）。Ollama / llama.cpp バイナリは使いません
- **CUI**: Ollama 風の `list` / `pull` / `run` / `rm` / `show` / `adapter`
- **ストレージ**: モデル本体（blobs）とエンジン派生物（cache）を分離。エンジン更新でも再ダウンロードしません
- **LoRA**: `adapter create` で学習（非公開コーパスは `config_lpcllm` の `train_dir`）。成果は `adapters/`（**バックアップ対象**）
- **パス**: `config_lpcllm` でバイナリ位置とユーザデータ位置を設定（`lpc-llm config show`）
- **モデル作成**: `train scratch|sft|dpo`（超小型 from-scratch → GGUF → `run`）；大規模化は `job`
- **今後の工程**: 進捗と予定は [`todo.md`](todo.md) を参照

## 日本語目次

1. [現状の仕様](#1-現状の仕様)
2. [マニュアル（導入〜起動〜停止）](#2-マニュアル導入起動停止)
3. [コマンドリファレンス](#3-コマンドリファレンス)
4. [トラブルシューティング](#4-トラブルシューティング)
5. [開発メモ](#5-開発メモ)
6. [今後の工程予定](#6-今後の工程予定)
7. [ライセンス](#7-ライセンス)

### パス早見表

**`config_lpcllm`** で設定（`lpc-llm config init` / `config show`）。既定値は下表。

| 対象 | 場所（既定） | Git | バックアップ |
|------|------|-----|--------------|
| ビルド成果 | `./target/debug/lpc-llm` または `./target/release/lpc-llm` | 無視（`/target/`） | 不要 |
| 開発用 PATH | `~/.local/bin/lpc-llm` → symlink（`install-dev.sh`） | n/a | 不要 |
| 共有バイナリ | `/usr/local/bin/lpc-llm`（`install-system.sh`；**バイナリのみ**） | n/a | 不要 |
| 設定ファイル | `~/.config/lpc-llm/config_lpcllm`（任意で `/etc/lpc-llm/…`） | n/a | 任意 |
| ベースモデル | `~/.local/share/lpc-llm/blobs/` | リポ外 | 任意（再 DL 可） |
| エンジン pack | `~/.local/share/lpc-llm/cache/packs/` | リポ外 | 不要（再生成可） |
| **学習用コーパス** | **`~/.local/share/lpc-llm/train/`**（`paths.train_dir`） | リポ外 | 任意 |
| 公開 / 開発サンプル | `examples/*` | 追跡 | n/a |
| **学習済み LoRA** | **`~/.local/share/lpc-llm/adapters/<name>/`** | リポ外 | **要バックアップ** |

非公開データは git ツリーに置かない。リポ内 `data/train/` は保険の gitignore のみ。詳細は [`data/README.md`](data/README.md) / [`config_lpcllm.example`](config_lpcllm.example)。

---

## 1. 現状の仕様

### アーキテクチャ概要

```text
┌─ CUI (clap) ─────────────────────────────────────┐
│  list / pull / run / rm / show / prefetch / io     │
└───────────────────────────┬───────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
   Eager 経路                              Hybrid 経路
   (Candle 一括ロード)                     (既定: gemma*)
        │                                       │
        │              ┌────────────────────────┤
        │              ▼                        ▼
        │     ensure_packed (engine cache)   hot layers pin
        │              │                        │
        │              ▼                        ▼
        │     io_uring ping-pong DMA      resident RAM
        │              │                        │
        └──────────────┴──────────┬─────────────┘
                                  ▼
                         forward (arch 分岐)
                         Gemma2: post-norm / √emb /
                         Neox RoPE / softcap / GeLU
```

### メモリ・I/O モデル（hybrid）

```text
[ embeddings + norm + lm_head ]     常駐 (blobs から直接読込)
[ hot layers 0 .. H-1 ]             RAM 固定（予算内、既定最大 8）
[ Prefetch A | Prefetch B ]         2× pack 層 DMA（io_uring）
[ KV cache ]                        コンテキストに応じて成長
```

チューニング方針（体感への効き）:

| 施策 | 効き | 実装 |
|------|------|------|
| パック再配置 + ダブルバッファ | 大 | `{gguf} → cache/.../layers.pack`、層ごと 1 DMA |
| ホット常駐比率 | 中〜大 | `--ram-mib` / `--hot-layers` |
| 思考の小分け（短い初回応答） | 中 | `--burst`、REPL の `/more` |
| チャンクサイズ微調整 | 小 | pack 時の DMA 窓アライン、I/O wait EMA |

### データレイアウト（モジュール分離）

**ランタイム状態**（`config_lpcllm` / XDG — git リポジトリの外）:

```text
~/.local/share/lpc-llm/          # paths.data_dir
  blobs/                 # モデルモジュール（永続 GGUF + tokenizer）
    <hf-repo--name>/
      *.gguf
      tokenizer.json
  adapters/              # LoRA / 差分モジュール（永続）← バックアップ対象
    <name>/
      adapter.json
      weights.bin
  train/                 # paths.train_dir — 非公開コーパス ← git に置かない
  cache/                 # エンジンモジュール（再生成可）
    packs/
      <model_name>/<engine_ver>/
        layers.pack
        layers.pack.json
  manifest.json          # ソフト索引（models + adapters）
```

**リポジトリ内の公開サンプル**（コミット可）:

```text
examples/train-sample.txt
examples/pref-sample.jsonl
config_lpcllm.example
```

| 領域 | 消してよいか | 備考 |
|------|-------------|------|
| `blobs/` | 原則残す | 再 DL が必要になる |
| `adapters/` | **残す / バックアップ** | 学習済み LoRA。消すと再学習が必要 |
| `train/` | 自分のデータ | ホーム配下の非公開コーパス。git に置くな |
| `cache/` | 消してよい | 次回 hybrid / train で再生成 |
| `manifest.json` | 消してよい | 起動時 `reconcile` で復旧 |

`rm` はレジストリから外すだけで **blobs は削除しません**。

### カタログモデル

| 名前 | 内容 | 目安サイズ | hybrid |
|------|------|------------|--------|
| `smollm2:360m` | SmolLM2 360M Instruct Q4_K_M | ~260 MB | `--hybrid` で有効 |
| `gemma2:2b` | Gemma 2 2B Instruct Q4_K_M | ~1.7 GB | **既定で hybrid** |
| `qwen2.5:1.5b` | Qwen2.5 1.5B Instruct Q4_K_M | ~1.1 GB | `--hybrid` で有効 |
| `phi3:mini` | Phi-3 Mini 4K Instruct Q4_K_M | ~2.2 GB | `--hybrid` で有効 |

Gemma2 向けに、post-attention / post-ffw norm、埋め込み √hidden スケール、Neox RoPE、attn/final logit softcap、GeLU を実装しています。  
GGUF の RMSNorm 重みは HF→GGUF 変換で既に `(1+δ)` 済みのため、ランタイムでは `w` をそのまま掛けます（二重適用しない）。

### 実行環境（想定）

- OS: Linux（io_uring / `O_DIRECT` 前提。Fedora 等）
- CPU 推論（現状）
- Rust toolchain（`edition = "2024"` のため比較的新しい stable）
- ダウンロード: システム `curl` または `wget`（OpenSSL リンクを避けるため）
- 任意: `HF_TOKEN`（ゲート付き HF リポ用）

---

## 2. マニュアル（導入〜起動〜停止）

### 0. 前提

```bash
rustc --version    # 新しい stable を推奨
curl --version     # または wget
```

リポジトリを取得済みであること（例: `~/dev/lpc-llm`）。

### 1. ビルド

```bash
cd ~/dev/lpc-llm

# Cursor / 一部環境で CARGO_TARGET_DIR が別パスを指す場合があるため、
# 手元の ./target に出したいときは unset する
unset CARGO_TARGET_DIR

# 日常開発（PATH の symlink がこのバイナリを追従）
cargo build
./scripts/install-dev.sh --no-build   # 一度だけ: ~/.local/bin/lpc-llm → target/debug/lpc-llm

# または release
cargo build --release
# → ./target/release/lpc-llm
```

リンカの `warning: linker stderr: ignoring deprecated...` は無視して問題ありません。

| 目的 | コマンド | バイナリ |
|------|---------|----------|
| 開発（`cargo build` ごとに更新） | `./scripts/install-dev.sh` のあと `cargo build` | PATH 上の symlink（`install.bin_dir`） |
| 共有バイナリ | `./scripts/install-system.sh` | `/usr/local/bin`（**バイナリのみ**；データはユーザごと） |
| コピーして入れる | `cargo install --path . --force` | `~/.cargo/bin/lpc-llm`（コンパイルでは自動更新されない） |
| インストールなし | `./target/debug/lpc-llm` など | 指定パス |
| パス設定 | `lpc-llm config init` / `config show` | `config_lpcllm` の読み書き |

### 2. モデルの導入（pull）

```bash
lpc-llm list
lpc-llm pull smollm2:360m    # 軽量スモーク
lpc-llm pull gemma2:2b
lpc-llm show gemma2:2b
```

成功例（再利用）:

```text
· gemma2:2b already in model module — reusing blobs (no download)
  model     ~/.local/share/lpc-llm/blobs/.../gemma-2-2b-it-Q4_K_M.gguf
  tokenizer ~/.local/share/lpc-llm/blobs/.../tokenizer.json
```

ゲート付きモデルで失敗する場合:

```bash
export HF_TOKEN=hf_xxxxxxxx
lpc-llm pull gemma2:2b
```

### 3. 起動（チャット / LLM 利用）

```bash
lpc-llm run gemma2:2b
lpc-llm run gemma2:2b --hybrid --ram-mib 4096 --burst 24
lpc-llm run smollm2:360m
lpc-llm run smollm2:360m --adapter my-lora
lpc-llm run gemma2:2b --agent
```

名前省略でメニュー:

```bash
lpc-llm run
lpc-llm
```

初回 hybrid では pack 生成が走ります（数分かかることがあります。GGUF は変更しません）。

```text
packing 26 layers → ~/.local/share/lpc-llm/cache/packs/gemma2_2b/0.1.0/layers.pack
…
✓ ready on CPU+pack+io_uring (gemma2)
>>>
```

`mlock failed ... using unlocked arenas` は警告です。推論は継続します（必要なら `ulimit -l` を上げる）。

### 4. 学習用データの設置場所

**非公開**コーパスは `config_lpcllm` の `train_dir`（既定 `~/.local/share/lpc-llm/train/` — git 外）に置く:

```bash
lpc-llm config init          # 一度: ~/.config/lpc-llm/config_lpcllm
TRAIN="$(lpc-llm config get train_dir)"
mkdir -p "$TRAIN"
cp /path/to/your-corpus.txt "$TRAIN/my-domain.txt"
# JSONL も可: 各行 {"text":"..."}
```

公開用スモークサンプル: [`examples/train-sample.txt`](examples/train-sample.txt)。  
詳細: [`data/README.md`](data/README.md)。

### 5. LoRA アダプタの学習

```bash
lpc-llm adapter list

# ファイル名のみなら train_dir 配下を解決
lpc-llm adapter create \
  --from my-domain.txt \
  --out my-lora \
  --base smollm2:360m \
  --steps 64 --rank 8 --last-layers 4

# 公開サンプルでの短いスモーク
lpc-llm adapter create \
  --from examples/train-sample.txt \
  --out smoke-lora \
  --base smollm2:360m \
  --steps 8 --rank 4 --last-layers 2
```

CPU 学習は step ごとに進捗を出します（1 step に数十秒かかることがあります）。

### 6. 学習結果の格納場所（バックアップ対象）

```text
~/.local/share/lpc-llm/adapters/<out>/
  adapter.json
  weights.bin
```

```bash
lpc-llm adapter list
ls ~/.local/share/lpc-llm/adapters/my-lora/
lpc-llm run smollm2:360m --adapter my-lora
```

**学習済み差分は `~/.local/share/lpc-llm/adapters/` をバックアップ**してください。  
`blobs/` は任意（再 DL 可）、`cache/` は再生成可です。

### 6b. 超小型 from-scratch / SFT / DPO（Phase 5）

```bash
lpc-llm train scratch \
  --from examples/train-sample.txt \
  --out tiny:demo \
  --steps 64 --n-embd 128 --n-layers 2 --ram-mib 1024

lpc-llm run tiny:demo

lpc-llm train sft --ckpt tiny:demo --from examples/train-sample.txt --out tiny:sft --steps 32
lpc-llm train dpo --ckpt tiny:sft --from examples/pref-sample.jsonl --out tiny:dpo --steps 32
lpc-llm train export --ckpt tiny:dpo --register --name tiny:dpo
```

チェックポイント: `~/.local/share/lpc-llm/cache/train/<name>/`

### 6c. 大規模化ジョブ（Phase 6）

```bash
lpc-llm job init --template rlhf --out job.json
lpc-llm job run --config job.json --local
lpc-llm job import --gguf /path/model.gguf --tokenizer /path/tokenizer.json --name my:import
lpc-llm job convert --from-dir ~/.local/share/lpc-llm/cache/train/tiny_demo \
  --name tiny:from-ckpt --backend builtin
```

数十億級 HF→GGUF は `LPC_LLM_CONVERT_CMD`（出力先 `$LPC_LLM_CONVERT_OUT`）を設定。

### 7. チャット中の操作

| 入力 | 動作 |
|------|------|
| 通常の文章 | モデルへ送信し、トークンをストリーム表示 |
| `/more` | 直前の応答をさらに生成（続き） |
| `/clear` | 会話履歴と KV をクリア |
| `/bye` `/exit` `/quit` | チャット終了 |

初回応答は `--burst`（既定 24 トークン）で短く出し、続きは `/more` で足せます。

### 8. 停止

- **対話中**: `/bye` を入力（推奨）
- **強制終了**: ターミナルで `Ctrl+C`
- バックグラウンドに残した場合:

```bash
pkill -f 'lpc-llm run'    # 必要時のみ
```

デーモン化はしていません。プロセスを止めれば推論も止まります。モデルとアダプタはディスクに残ります。

### 9. 典型的な一日の流れ（最短）

```bash
cd ~/dev/lpc-llm
unset CARGO_TARGET_DIR
cargo build                         # install-dev.sh 後は PATH の lpc-llm が更新される
lpc-llm pull smollm2:360m           # 初回
lpc-llm run smollm2:360m
# … 会話 …
# >>> /bye
```

### 10. （任意）I/O ベンチ

```bash
lpc-llm prefetch gemma2:2b
lpc-llm io --help
```

---

## 3. コマンドリファレンス

| コマンド | 説明 |
|----------|------|
| `lpc-llm` | 対話メニュー |
| `lpc-llm list` | カタログと local / available |
| `lpc-llm pull <name>` | blobs へ取得（既存は再利用） |
| `lpc-llm run [name] [options]` | チャット起動 |
| `lpc-llm show <name>` | カタログ + ローカルパス |
| `lpc-llm rm <name>` | レジストリから削除（blobs は残す） |
| `lpc-llm adapter list` | LoRA アダプタ一覧 |
| `lpc-llm config show\|init\|get\|example` | `config_lpcllm` の解決 / 作成（data_dir, train_dir, bin_dir） |
| `lpc-llm adapter create …` | `--from` から LoRA 学習 → `adapters/<out>/` |
| `lpc-llm adapter install-demo` | 検証用ゼロ LoRA |
| `lpc-llm train scratch\|sft\|dpo\|export` | 超小型 from-scratch / フル SFT / DPO / GGUF（Phase 5） |
| `lpc-llm job init\|run\|status\|import\|convert` | 宣言的ジョブ / リモートブリッジ（Phase 6） |
| `lpc-llm prefetch <name>` | pack + io_uring ping-pong 計測 |
| `lpc-llm io` | 合成重みでの I/O デモ |

### `run` オプション

| オプション | 既定 | 意味 |
|------------|------|------|
| `--pull` | off | 未導入なら確認なしで pull |
| `--hybrid` | gemma* は on | 層ストリーミング推論 |
| `--hot-layers N` | 自動 | RAM 常駐層数を強制 |
| `--ram-mib N` | 4096 | ホット層 + 2 スロットのソフト予算 (MiB) |
| `--burst N` | 24 | 初回応答の最大トークン数 |
| `--adapter <name>` | なし | LoRA サイドパス（hybrid 強制） |
| `--agent` | off | ルーター後にメイン（RAM 排他） |
| `--agent-model` | `smollm2:360m` | `--agent` 用ルーター |

### `adapter create` オプション

| オプション | 既定 | 意味 |
|------------|------|------|
| `--from <path>` | 必須 | 学習用 `.txt` / `.jsonl`（cwd パスまたは `train_dir` 下の名前） |
| `--out <name>` | 必須 | `adapters/<name>/` の名前 |
| `--base <model>` | 必須 | カタログのベース（例: `smollm2:360m`） |
| `--rank` | 8 | LoRA rank |
| `--alpha` | 16 | LoRA α |
| `--steps` | 64 | AdamW 更新回数 |
| `--lr` | 1e-3 | 学習率 |
| `--max-seq` | 128 | チャンクあたりトークン数 |
| `--last-layers N` | 0（全層） | 末尾 N 層だけ学習 |
| `--ram-mib` | 4096 | 学習時ロードのソフト予算 |
| `--pull` | off | ベース未導入時に確認なし pull |
---

## 4. トラブルシューティング

| 症状 | 対処 |
|------|------|
| `Jove Jove…` などゴミ出力 | 古いバイナリの可能性。`unset CARGO_TARGET_DIR && cargo build --release` 後に `./target/release/lpc-llm` を使う |
| `mlock failed` | 警告のみ。必要なら `ulimit -l unlimited`（権限による） |
| 毎回ダウンロードされる | `~/.local/share/lpc-llm/blobs` を確認。旧 `~/.local/share/l3m` から移行する場合は rename / symlink |
| pack が遅い | 初回のみ。`cache/packs` を消せば再生成される |
| `--from file not found` | コーパスを `train_dir`（`lpc-llm config get train_dir`）に置くか実在パスを渡す。[`data/README.md`](data/README.md) |
| `adapter … not found` | 先に `adapter create` を成功させる。`adapter list` / `~/.local/share/lpc-llm/adapters/` を確認 |
| HF 401 | `HF_TOKEN` とライセンス同意 |
| ビルドが長い | release + LTO。警告だけなら失敗ではない |

旧データ移行例:

```bash
# 新パスが未作成のとき
mv ~/.local/share/l3m ~/.local/share/lpc-llm
```

---

## 5. 開発メモ

- 言語: Rust 2024
- 主要クレート: `candle-core` / `candle-nn` / `candle-transformers` / `tokenizers` / `io-uring` / `memmap2`
- バイナリ名: `lpc-llm`（`Cargo.toml` の package name）
- Ollama との関係: **非依存**。CUI の操作感のみ類似
- ライセンス: Apache-2.0（`LICENSE` / `Cargo.toml`）

### 開発用バイナリを PATH に載せる

マシン / クローンごとに一度:

```bash
unset CARGO_TARGET_DIR   # 成果物を ./target に置く（symlink 先）
./scripts/install-dev.sh
```

以降の日常作業:

```bash
cargo build             # target/debug/lpc-llm が更新 → PATH の lpc-llm も追従
cargo check             # 型チェックのみ（バイナリは書き換わらない）
lpc-llm adapter list    # 常に最新の debug ビルド
```

release を追従させたいとき: `./scripts/install-dev.sh --release`。

```bash
cargo check
cargo build --release
```

---

## 6. 今後の工程予定

MoE・差分アダプタ・軽量エージェント以降の実装進捗と推奨次工程は、次のドキュメントにまとめています。

- **[`todo.md`](todo.md)** — 拡張ロードマップ（英語 / 日本語）

特に総括・工程チェックリスト（Phase 4〜8）・推奨する次工程を参照してください。

---

## 7. ライセンス

[Apache License 2.0](LICENSE)
