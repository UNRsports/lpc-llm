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
- **CUI**: Ollama-like `list` / `pull` / `run` / `rm` / `show`
- **Storage**: model blobs and engine-derived cache are separated. Engine upgrades do not force re-download
- **Roadmap**: progress and planned phases are tracked in [`todo.md`](todo.md)

## English table of contents

1. [Current specification](#1-current-specification)
2. [Manual (setup → start → stop)](#2-manual-setup--start--stop)
3. [Command reference](#3-command-reference)
4. [Troubleshooting](#4-troubleshooting)
5. [Development notes](#5-development-notes)
6. [Roadmap / planned work](#6-roadmap--planned-work)
7. [License](#7-license)

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

```text
~/.local/share/lpc-llm/
  blobs/                 # model module (durable)
    <hf-repo--name>/
      *.gguf
      tokenizer.json
  cache/                 # engine module (regenerable)
    packs/
      <model_name>/<engine_ver>/
        layers.pack
        layers.pack.json
  manifest.json          # soft index (auto-recovered from blobs)
```

| Area | Safe to delete? | Notes |
|------|-------------|------|
| `blobs/` | Keep by default | Deleting forces re-download |
| `cache/` | Yes | Regenerated on next hybrid run |
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

cargo build --release
```

On success you get:

```text
./target/release/lpc-llm
```

Linker `warning: linker stderr: ignoring deprecated...` can be ignored.

To put it on PATH:

```bash
cargo install --path . --force
# then: lpc-llm ...
```

### 2. Install a model (pull)

List:

```bash
./target/release/lpc-llm list
```

Fetch (skips re-download if blobs already exist):

```bash
# smoke test (lightweight)
./target/release/lpc-llm pull smollm2:360m

# Gemma 2 2B
./target/release/lpc-llm pull gemma2:2b
```

Success example (reuse):

```text
· gemma2:2b already in model module — reusing blobs (no download)
  model     ~/.local/share/lpc-llm/blobs/.../gemma-2-2b-it-Q4_K_M.gguf
  tokenizer ~/.local/share/lpc-llm/blobs/.../tokenizer.json
```

Inspect:

```bash
./target/release/lpc-llm show gemma2:2b
```

If a gated model fails:

```bash
export HF_TOKEN=hf_xxxxxxxx
./target/release/lpc-llm pull gemma2:2b
```

### 3. Start (chat)

```bash
# Gemma (hybrid default)
./target/release/lpc-llm run gemma2:2b

# explicit options
./target/release/lpc-llm run gemma2:2b --hybrid --ram-mib 4096 --burst 24

# light model (eager; pass --hybrid to use hybrid)
./target/release/lpc-llm run smollm2:360m
```

Omit the name to pick from an interactive menu:

```bash
./target/release/lpc-llm run
# or no subcommand → menu
./target/release/lpc-llm
```

First hybrid run builds the pack (may take minutes; GGUF is not modified):

```text
packing 26 layers → ~/.local/share/lpc-llm/cache/packs/gemma2_2b/0.1.0/layers.pack
…
✓ ready on CPU+pack+io_uring (gemma2)
>>>
```

`mlock failed ... using unlocked arenas` is a warning; inference continues (raise `ulimit -l` if needed).

### 4. In-chat controls

| Input | Action |
|------|------|
| Normal text | Send to model; stream tokens |
| `/more` | Continue generating the last reply |
| `/clear` | Clear history and KV |
| `/bye` `/exit` `/quit` | Leave chat |

The first reply is capped by `--burst` (default 24 tokens); use `/more` for more.

### 5. Stop

- **In session**: type `/bye` (preferred)
- **Force quit**: `Ctrl+C` in the terminal
- If left in the background:

```bash
pkill -f 'lpc-llm run'    # only when needed
```

There is no daemon. Stopping the process stops inference. Model files remain on disk.

### 6. Typical daily flow (shortest)

```bash
cd ~/dev/lpc-llm
unset CARGO_TARGET_DIR
cargo build --release          # only after changes
./target/release/lpc-llm pull gemma2:2b   # first time or check
./target/release/lpc-llm run gemma2:2b
# … chat …
# >>> /bye
```

### 7. (Optional) I/O bench

```bash
./target/release/lpc-llm prefetch gemma2:2b
./target/release/lpc-llm io --help
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

---

## 4. Troubleshooting

| Symptom | Fix |
|------|------|
| Garbage like `Jove Jove…` | Possibly an old binary. `unset CARGO_TARGET_DIR && cargo build --release`, then use `./target/release/lpc-llm` |
| `mlock failed` | Warning only. Optionally `ulimit -l unlimited` (depends on privileges) |
| Downloads every time | Check `~/.local/share/lpc-llm/blobs`. Migrating from old `~/.local/share/l3m`: rename / symlink |
| Pack is slow | First run only. Delete `cache/packs` to regenerate |
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
- **CUI**: Ollama 風の `list` / `pull` / `run` / `rm` / `show`
- **ストレージ**: モデル本体（blobs）とエンジン派生物（cache）を分離。エンジン更新でも再ダウンロードしません
- **今後の工程**: 進捗と予定は [`todo.md`](todo.md) を参照

## 日本語目次

1. [現状の仕様](#1-現状の仕様)
2. [マニュアル（導入〜起動〜停止）](#2-マニュアル導入起動停止)
3. [コマンドリファレンス](#3-コマンドリファレンス)
4. [トラブルシューティング](#4-トラブルシューティング)
5. [開発メモ](#5-開発メモ)
6. [今後の工程予定](#6-今後の工程予定)
7. [ライセンス](#7-ライセンス)

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

```text
~/.local/share/lpc-llm/
  blobs/                 # モデルモジュール（永続）
    <hf-repo--name>/
      *.gguf
      tokenizer.json
  cache/                 # エンジンモジュール（再生成可）
    packs/
      <model_name>/<engine_ver>/
        layers.pack
        layers.pack.json
  manifest.json          # ソフト索引（blobs から自動復旧）
```

| 領域 | 消してよいか | 備考 |
|------|-------------|------|
| `blobs/` | 原則残す | 再 DL が必要になる |
| `cache/` | 消してよい | 次回 hybrid で pack 再生成 |
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

cargo build --release
```

成功すると次ができます。

```text
./target/release/lpc-llm
```

リンカの `warning: linker stderr: ignoring deprecated...` は無視して問題ありません。

PATH に載せたい場合:

```bash
cargo install --path . --force
# 以降: lpc-llm ...
```

### 2. モデルの導入（pull）

一覧:

```bash
./target/release/lpc-llm list
```

取得（既存 blobs があれば再 DL しません）:

```bash
# 動作確認用（軽量）
./target/release/lpc-llm pull smollm2:360m

# Gemma 2 2B
./target/release/lpc-llm pull gemma2:2b
```

成功例（再利用）:

```text
· gemma2:2b already in model module — reusing blobs (no download)
  model     ~/.local/share/lpc-llm/blobs/.../gemma-2-2b-it-Q4_K_M.gguf
  tokenizer ~/.local/share/lpc-llm/blobs/.../tokenizer.json
```

詳細確認:

```bash
./target/release/lpc-llm show gemma2:2b
```

ゲート付きモデルで失敗する場合:

```bash
export HF_TOKEN=hf_xxxxxxxx
./target/release/lpc-llm pull gemma2:2b
```

### 3. 起動（チャット）

```bash
# Gemma（hybrid 既定）
./target/release/lpc-llm run gemma2:2b

# 明示オプション例
./target/release/lpc-llm run gemma2:2b --hybrid --ram-mib 4096 --burst 24

# 軽量モデル（eager。hybrid にするなら --hybrid）
./target/release/lpc-llm run smollm2:360m
```

名前を省略すると対話メニューから選択できます。

```bash
./target/release/lpc-llm run
# またはサブコマンドなし → メニュー
./target/release/lpc-llm
```

初回 hybrid では pack 生成が走ります（数分かかることがあります。GGUF は変更しません）。

```text
packing 26 layers → ~/.local/share/lpc-llm/cache/packs/gemma2_2b/0.1.0/layers.pack
…
✓ ready on CPU+pack+io_uring (gemma2)
>>>
```

`mlock failed ... using unlocked arenas` は警告です。推論は継続します（必要なら `ulimit -l` を上げる）。

### 4. チャット中の操作

| 入力 | 動作 |
|------|------|
| 通常の文章 | モデルへ送信し、トークンをストリーム表示 |
| `/more` | 直前の応答をさらに生成（続き） |
| `/clear` | 会話履歴と KV をクリア |
| `/bye` `/exit` `/quit` | チャット終了 |

初回応答は `--burst`（既定 24 トークン）で短く出し、続きは `/more` で足せます。

### 5. 停止

- **対話中**: `/bye` を入力（推奨）
- **強制終了**: ターミナルで `Ctrl+C`
- バックグラウンドに残した場合:

```bash
pkill -f 'lpc-llm run'    # 必要時のみ
```

デーモン化はしていません。プロセスを止めれば推論も止まります。モデルファイルはディスクに残ります。

### 6. 典型的な一日の流れ（最短）

```bash
cd ~/dev/lpc-llm
unset CARGO_TARGET_DIR
cargo build --release          # 変更後のみ
./target/release/lpc-llm pull gemma2:2b   # 初回 or 確認
./target/release/lpc-llm run gemma2:2b
# … 会話 …
# >>> /bye
```

### 7. （任意）I/O ベンチ

```bash
./target/release/lpc-llm prefetch gemma2:2b
./target/release/lpc-llm io --help
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

---

## 4. トラブルシューティング

| 症状 | 対処 |
|------|------|
| `Jove Jove…` などゴミ出力 | 古いバイナリの可能性。`unset CARGO_TARGET_DIR && cargo build --release` 後に `./target/release/lpc-llm` を使う |
| `mlock failed` | 警告のみ。必要なら `ulimit -l unlimited`（権限による） |
| 毎回ダウンロードされる | `~/.local/share/lpc-llm/blobs` を確認。旧 `~/.local/share/l3m` から移行する場合は rename / symlink |
| pack が遅い | 初回のみ。`cache/packs` を消せば再生成される |
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
