# lpc-llm extension roadmap (progress)

## Contents / 目次

1. [English](#english)
   1. [Summary](#1-summary)
   2. [Feasibility of theme add-ons](#2-feasibility-of-theme-add-ons)
   3. [Engineering checklist](#3-engineering-checklist)
   4. [Spec section status](#4-spec-section-status)
   5. [Recommended next steps](#5-recommended-next-steps)
   6. [Notes (done outside the spec)](#6-notes-done-outside-the-spec)
2. [日本語](#日本語)
   1. [総括](#1-総括)
   2. [テーマ追加要件の実現可能性](#2-テーマ追加要件の実現可能性)
   3. [工程チェックリスト](#3-工程チェックリスト)
   4. [仕様書セクション別の対応状況](#4-仕様書セクション別の対応状況)
   5. [推奨する次工程](#5-推奨する次工程)
   6. [補足（仕様外だが実施済み）](#6-補足仕様外だが実施済み)

---

# English

Implementation status against the spec “MoE support · delta-adapter driven · lightweight agent integration”.  
Project theme: **efficient LLM execution and model creation under constrained resources**  
Last updated: 2026-08-02

## English table of contents

1. [Summary](#1-summary)
2. [Feasibility of theme add-ons](#2-feasibility-of-theme-add-ons)
3. [Engineering checklist](#3-engineering-checklist)
4. [Spec section status](#4-spec-section-status)
5. [Recommended next steps](#5-recommended-next-steps)
6. [Notes (done outside the spec)](#6-notes-done-outside-the-spec)

---

## 1. Summary

| Axis | Content | Progress |
|----|------|------|
| Foundation (existing) | GGUF layer pack + io_uring double-buffer hybrid | **Done** (prerequisite for this extension) |
| Axis 2 / Phase 1 | Delta adapter mgmt · side-path LoRA · `--adapter` | **Done** |
| Axis 1 / Phase 2 | MoE expert split pack + dynamic DMA | **Done** |
| Axis 3 / Phase 3 | Ultra-light router agent + memory exclusivity | **Done** |
| Axis 2 / Phase 4 | `adapter create` trainer prototype | **Done** |
| Axis 2 / Phase 5 | Tiny from-scratch · GGUF export · local SFT/DPO | **Done** |
| Long-term / Phase 6 | Scale-up bridge (remote jobs · convert · RLHF stages) | **Done** (bridge; full cluster PPO external) |
| Extension / Phase 7 | Auto knowledge acquisition & user adaptation (Web + auto-train) | **Not started** (conditionally feasible) |
| Extension / Phase 8 | NVMe-resident project-map & overview memory | **Not started** (conditionally feasible) |

**Available now:**  
`lpc-llm run <model> --adapter <name>` (Hybrid LoRA),  
`lpc-llm run <model> --agent` (SmolLM2 router → auto adapter/expert hints, exclusive RAM),  
`lpc-llm adapter create --from … --out … --base …` (LoRA SFT → `adapters/<name>/`),  
`lpc-llm train scratch|sft|dpo|export` (tiny from-scratch → GGUF → `run`),  
`lpc-llm job init|run|import|convert` (declarative stages / remote bridge / RLHF stubs),  
`lpc-llm config show|init|get` (`config_lpcllm`: bin_dir + per-user data/train),  
On MoE GGUF: `experts.pack` + Top-K expert DMA (hybrid).  
**Not available yet:** multi-GPU PPO in-process, Web knowledge acquisition, `user_profile` auto-train, `--project-map`.

---

## 2. Feasibility of theme add-ons

How to treat the following three requirements under the theme “efficient execution and model creation under constrained resources”.

| Requirement | As-is under constrained resources? | Verdict | Realistic landing in this repo |
|------|---------------------------|------|--------------------------------|
| Full base-model training from scratch | Full training of multi-billion models on home CPU / low RAM is unrealistic (compute, data, power differ by orders of magnitude) | **Conditionally feasible** | First ship a **tiny (millions–hundreds of M) from-scratch training loop** in pure Rust/Candle. Large scale via external jobs or checkpoint import |
| Build a new multi-billion-param GGUF from scratch | “Train from scratch then emit multi-billion GGUF” is the same as above. A **GGUF export pipeline as a format** is feasible | **Conditionally feasible** | (1) small training → GGUF write-out (2) quantize/convert existing weights → register in `blobs/`. The multi-billion training body is a separate cluster-oriented stage |
| Full SFT / RLHF pipeline | Full RLHF (large reward model + PPO etc.) assumes many GPUs; tension with the theme | **Conditionally feasible** | Pipeline locally up to **SFT (LoRA/QLoRA) → light preference opt (DPO/ORPO etc.)**. Keep “full RLHF” as staged / external-accelerator work |

**Conclusion:** All three are “engineerable,” but **finishing full scale on the current machine alone** contradicts the theme. This todo carries both (A) intermediate artifacts that complete under constrained resources and (B) long-term stages aimed at full scale.

### Feasibility of Phase 7 / 8 add-ons

| Requirement | Verdict | Prerequisites / landing |
|------|------|-------------------|
| Web search → accumulate in `cache/knowledge/` | **Feasible** | DuckDuckGo Instant Answer / HTML scrape / Custom API. Sync in-chat; async jobs in background. Store chunks + metadata (source URL, fetch time) locally; inject into prompts RAG-style at inference |
| User habits → auto LoRA in `adapters/user_profile/` | **Conditionally feasible** | **Phase 4 (`adapter create`) required**. Accumulate edits/prompt logs in `cache/user_logs/` → idle detect (Linux: idle time / D-Bus) for delta training. Avoid always-on full train; small batch, low rank, time-capped |
| Auto-attach `--adapter user_profile` | **Feasible** | Reuse Phase 1 hybrid side-path. Auto-load at `run` if present (no restart = in-process attach; daemon optional) |
| Project AST/dep graph → `map.bin` | **Feasible** | Extract AST/symbols/call edges with tree-sitter etc. Light embeddings (hash or small model) as node attrs. Keep structured index on NVMe without loading all code into RAM |
| `io_uring` on-demand symbol fetch | **Feasible** | Same shape as layer-pack DMA. Fixed-length node records + offset table, `O_DIRECT` prefetch. Millisecond delta updates = reparse changed files → rewrite affected subgraph only |
| `--project-map` overview context | **Conditionally feasible** | Cannot dump hundreds of thousands of lines into the prompt. Synthesize **summaries / signature lists of relevant subgraphs**. Cursor-class IDE integration is out of scope; CLI graph RAG is the realistic endpoint |

**Conclusion:** Phases 7 and 8 are engineerable. Phase 7 auto-train needs Phase 4 done; Phase 8 DMA fetch extends the existing io_uring stack. Put constrained-resource intermediate artifacts first; approach the ideal (fully automatic, full overview) gradually.

---

## 3. Engineering checklist

### 0. Existing foundation (spec prerequisite · already reached)

- [x] Ollama-independent pure Rust (Candle) inference
- [x] `blobs/` / `cache/packs/` / `manifest.json` separation
- [x] Layer re-layout via `layers.pack` + `layers.pack.json`
- [x] io_uring + O_DIRECT double-buffer streaming
- [x] Hot-layer budget via `--ram-mib` / `--hot-layers`
- [x] Catalog (`gemma2:2b`, `smollm2:360m`, …) and CUI (list/pull/run/…)

### Phase 1: Delta adapter (LoRA) load foundation — **Done**

- [x] `adapters/` storage management (`LocalStore`)
- [x] `manifest.json` `adapters` index wiring (discover / reconcile / record)
- [x] On-disk form `adapters/<name>/{adapter.json,weights.bin}` (FP16 A/B)
- [x] Side-path LoRA module (`y = Wq(x) + (α/r)·(x@Aᵀ)@Bᵀ`)
- [x] Dynamic inject into Hybrid `QMatMul` / Attention · MLP (`src/adapter/`, `hybrid.rs`)
- [x] Deduct adapter resident bytes from hot-layer budget
- [x] `lpc-llm run <model> --adapter <name>` (forces hybrid when set)
- [x] `lpc-llm adapter list`
- [x] Zero fixture for integration: `lpc-llm adapter install-demo`
- [x] Do not rewrite base `blobs/` / `layers.pack`
- [ ] (Optional) Mid-conversation adapter hot-swap
- [ ] (Optional) LoRA on Eager path
- [ ] (Optional) Safetensors / PEFT load compatibility

### Phase 2: MoE pack + expert streaming — **Done**

- [x] GGUF MoE tensor parse (`ffn_gate_exps`, `ffn_down_exps` / `ffn_gate.N`, etc.)
- [x] Separate resident (embeddings / norm / lm_head / router) from on-demand experts
- [x] Re-layout into `cache/packs/.../experts.pack`
- [x] Expert index / offset table in `experts.pack.json` (also referenced from `layers.pack.json`)
- [x] Gating network (router) inference + Top-K expert select
- [x] io_uring DMA for selected experts
- [x] Extend 2× buffers to expert-unit dynamic ring (`PrefetchRing`)
- [x] Arch branches for DeepSeek / Mixtral / Qwen-MoE (`MoeFamily` + both layouts)

### Phase 3: Ultra-light router agent — **Done**

- [x] `lpc-llm run … --agent` CLI (`--agent-model` to swap router)
- [x] Intent-classify prompt with SmolLM2 360M (default)
- [x] Decision → auto `--adapter` / expert prefetch (explicit `--adapter` wins)
- [x] Hand context to main after router (time-share)
- [x] Exclusive router KV vs main KV within `--ram-mib` (drop router Engine before loading main)

### Phase 4: Adapter creator — **Done**

- [x] Implement `lpc-llm adapter create --from … --out … --base …`  
      (`src/adapter/train.rs` — Hybrid LoRA SFT + AdamW)
- [x] Train/save a few-MB delta from small text in minutes
- [x] Match output to Phase 1 form (`adapter.json` + `weights.bin`)
- [x] Document build / run / train-data / adapter-backup paths in README  
      (private corpora under `train_dir` from `config_lpcllm`; results under `adapters/`)
- [ ] (Optional) Separate crate / Safetensors output

### Privacy / install layout — **Done** (user isolation + GitHub leak avoidance)

Goal: shared **binary only**; per-user data under home; private corpora never in the git tree.

- [x] `config_lpcllm` schema: `[paths]` (`data_dir`, `train_dir`) + `[install]` (`mode`, `bin_dir`)
- [x] Load order: defaults → `/etc/lpc-llm/config_lpcllm` → `~/.config/lpc-llm/config_lpcllm` → `$LPC_LLM_CONFIG` → env overrides
- [x] `lpc-llm config show|init|get|example` CLI
- [x] `LocalStore` / `--from` resolve via configured `data_dir` / `train_dir` (default `~/.local/share/lpc-llm/train`)
- [x] User install: `scripts/install-dev.sh` → `install.bin_dir` (default `~/.local/bin`)
- [x] System install: `scripts/install-system.sh` → `/usr/local/bin` (**binary only**; no user data)
- [x] Demote repo `data/train/` to safety-net gitignore only; public samples stay in `examples/`
- [x] Document privacy rule in README / `data/README.md` / `config_lpcllm.example`
- [ ] (Optional) Package unit / distro packaging that ships `/etc/lpc-llm/config_lpcllm` with `mode = "system"`

### Phase 5: Constrained-resource “model creation” foundation — **Done** (theme-critical · runnable)

**Front stage** of the three full-scale requirements. Completes on home–workstation scale.

- [x] Tiny Transformer from-scratch training loop (Candle; CPU) — `lpc-llm train scratch`
- [x] Training checkpoint → GGUF (F16 llama) write-out — `train export` / auto on register
- [x] Register artifacts in `blobs/` + `manifest` and run with `lpc-llm run`
- [x] Local SFT pipeline (full FT on tiny ckpt; LoRA via Phase 4 `adapter create`)
- [x] Minimal light preference opt (DPO) — `lpc-llm train dpo` + `examples/pref-sample.jsonl`
- [x] Memory-aware training design (`--ram-mib` / `--grad-checkpoint`, seq clamp)

### Phase 6: Scale-up bridge — **Done** (conditional · external resources)

Bridge so “multi-billion” and “full RLHF” can be handled as **extensions of this toolchain**. Single-machine local completion is not required.

- [x] **Full base-model training from scratch**  
      - Interfaces to launch / resume remote/distributed jobs and import artifacts (`job` + `remote.launch`)  
      - Declarative dataset / tokenizer / train config (`job.json` stages)  
      - Wire progress / checkpoints to `cache/jobs/` + `cache/train/`
- [x] **Build a new multi-billion-param GGUF from scratch**  
      - Large checkpoint → GGUF conversion bridge (`job convert --backend external` + `$LPC_LLM_CONVERT_CMD`)  
      - Builtin tiny ckpt → GGUF + register; hybrid pack on first `--hybrid` run  
      - ※ Training compute itself stays on remote/cluster side
- [x] **Full SFT / RLHF pipeline**  
      - Stage defs: SFT → preference (DPO) → PPO stub → export (`job init --template rlhf`)  
      - Eval / emit to `adapters/` or `blobs/` via export/import stages  
      - Accelerator left to external `remote.launch` / convert cmd (io_uring inference path unchanged)

### Phase 7: Auto knowledge acquisition & user adaptation — **Not started** (conditionally feasible)

Async Web knowledge acquisition and automatic delta-LoRA updates from user tendencies.  
**Deps:** Training for 7.2 / 7.3 requires **Phase 4 (`adapter create`)**. 7.1 and auto-attach can start with Phase 1 alone.

#### 7.1 Web search · knowledge injection (`search` integration)

- [ ] Search backend abstraction (DuckDuckGo / Custom HTTP API swappable)
- [ ] In-chat “knowledge gap” heuristics (unknown entities, explicit search, low-confidence answers)
- [ ] Background search jobs (async fetch → parse → persist)
- [ ] `cache/knowledge/` store (chunk body · source URL · fetch time · tags)
- [ ] Knowledge inject at inference (RAG-style related chunks into prompt; respect KV budget)
- [ ] CLI: `lpc-llm search <query>` / `lpc-llm knowledge list|purge` (optional)

#### 7.2 Auto-adapterize user habits / context (`adapter auto-train`)

- [ ] Local logs of chats / edits / prompt tendencies (`cache/user_logs/`; privacy + rotation policy)
- [ ] Extract coding-style features (indent · naming · comment density, etc.)
- [ ] Linux idle detect (X11/Wayland idle or simple idle timer)
- [ ] On idle, call Phase 4 trainer and update delta LoRA under `adapters/user_profile/`
- [ ] Job guards (time cap · RAM cap · min samples · rollback on failure)
- [ ] CLI: `lpc-llm adapter auto-train [--once|--daemon]` (optional)

#### 7.3 Seamless auto-attach

- [ ] At `run` start, if `adapters/user_profile/` is valid, auto-wire into Hybrid side-path
- [ ] Priority rules vs `--no-user-profile` / explicit `--adapter`
- [ ] (Optional) In-process hot reload (new weights from next turn after train)
- [ ] (Optional) Mid-chat hot-swap merges with Phase 1 optional work

### Phase 8: NVMe-resident project-map & overview memory — **Not started** (conditionally feasible)

Without loading all code into 16GB RAM, pull only needed nodes from a structured graph on NVMe via `io_uring`.  
**Deps:** Existing layer-pack `io_uring` / `O_DIRECT` pipeline. Can start independent of Phase 2 (buffering strategy may be shared).

#### 8.1 Map project graph onto NVMe

- [ ] Language frontends (tree-sitter etc.) for file AST / symbol extract
- [ ] Graph call / type-dependency edges for functions/classes
- [ ] Light node embeddings (hash n-gram or small embedder; full LLM embed optional)
- [ ] On-disk `cache/projects/<hash>/map.bin` + offset/index meta (`map.json`, etc.)
- [ ] Watch file updates and **delta index updates** (changed files + affected edges only)
- [ ] CLI: `lpc-llm project-map build|status|rebuild <path>`

#### 8.2 On-demand symbol fetch via `io_uring`

- [ ] Fixed-length or chunk-boundary records for nodes/edges (`O_DIRECT` aligned)
- [ ] Query → related node set (BM25 / embedding neighborhood / graph neighborhood combo)
- [ ] `io_uring` prefetch of selected nodes → RAM ring buffer
- [ ] Token-budget cap when assembling context (aligned with hot-layer budget)

#### 8.3 `--project-map` wide-context overview

- [ ] `lpc-llm run … --project-map [<path|hash>]` CLI
- [ ] Synthesize call/type deps as **subgraph summaries** into the prompt (no full dump)
- [ ] Structural hints for refactor/codegen (callee signature lists · impact scope)
- [ ] Regression bench that tens/hundreds of kLOC can be handled “as structure” on ~16GB RAM (optional)

---

## 4. Spec section status

### Data layout

| Path | Spec | Status |
|------|------|------|
| `blobs/` | Base GGUF | As existing |
| `adapters/` | Delta modules | **Implemented** (dir + json/bin); **back up** |
| `adapters/user_profile/` | Auto-train user LoRA | **Not implemented** (Phase 7) |
| `paths.train_dir` (home) | Private corpora for `adapter create` / `train` | **Implemented** (default `<data_dir>/train`; via `config_lpcllm`) |
| `data/train/` (repo) | Dev leftover only | **gitignore safety net** — do not store private data here |
| `cache/packs/.../layers.pack` | Base layer pack | Existing (name is `layers.pack`; rename to spec’s `base_layers.pack` not done) |
| `cache/packs/.../experts.pack` | MoE expert pack | **Implemented** |
| `cache/knowledge/` | Web-fetched knowledge | **Not implemented** (Phase 7) |
| `cache/user_logs/` | Habit-train logs | **Not implemented** (Phase 7) |
| `cache/projects/<hash>/map.bin` | Project structure graph | **Not implemented** (Phase 8) |
| `manifest.json` | models + adapters | **`adapters` key added** |

### CLI

| Command | Status |
|----------|------|
| `run … --adapter <name>` | **Implemented** |
| `run … --agent` | **Implemented** (with `--agent-model`) |
| `run … --project-map` | **Not implemented** (Phase 8) |
| `adapter list` | **Implemented** |
| `adapter install-demo` | **Implemented** (for verification) |
| `config show\|init\|get\|example` | **Implemented** (`config_lpcllm` path / install layout) |
| `adapter create …` | **Implemented** (LoRA SFT → Phase 1 on-disk form; `--from` via `train_dir`) |
| `train scratch|sft|dpo|export` | **Implemented** (Phase 5 tiny train → GGUF → `run`) |
| `job init|run|status|import|convert` | **Implemented** (Phase 6 bridge) |
| `adapter auto-train` | **Not implemented** (Phase 7) |
| `search` / `knowledge …` | **Not implemented** (Phase 7) |
| `project-map build|status` | **Not implemented** (Phase 8) |

### Memory / I/O pipeline

| Item | Status |
|------|------|
| Layer pack + ping-pong DMA | Existing |
| LoRA side-path (attach at compute) | **Implemented** (DMA buffers non-destructive) |
| Expert-unit index / dynamic DMA | **Implemented** (`experts.pack` + `PrefetchRing`) |
| project-map node `io_uring` prefetch | **Not implemented** (Phase 8) |
| ΔW merge at CQE (weight rewrite) | Not adopted (side-path policy) |

---

## 5. Recommended next steps

1. **Phase 7** — Web knowledge → `user_profile` auto-train · auto-attach (can reuse Phase 4 trainer); keep logs under user `cache/` only
2. **Phase 8** — `project-map` index + `io_uring` on-demand fetch + `--project-map`
3. **Phase 6 follow-ups** — Wire real cluster launchers / CUDA backends into `job.remote` and `$LPC_LLM_CONVERT_CMD`
4. **(Optional)** Distro / package install that ships system `config_lpcllm` with `install.mode = "system"`

---

## 6. Notes (done outside the spec)

- [x] Fix Backspace on Japanese input truncating UTF-8 bytes (REPL switched to `rustyline`)
- [x] Dev PATH install via `scripts/install-dev.sh` (symlink → `target/debug`; respects `config_lpcllm` `bin_dir`)
- [x] System binary install via `scripts/install-system.sh` (shared binary only)
- [x] Privacy: private corpora under `train_dir` (home/XDG); repo `data/train/` is gitignore-only; public samples in `examples/`

---

# 日本語

仕様書「MoE 対応・差分アダプタ駆動・軽量エージェント統合」に対する実装状況。  
プロジェクトテーマ: **限定的リソース下での LLM 効率化実行とモデル作成**  
最終更新: 2026-08-02

## 日本語目次

1. [総括](#1-総括)
2. [テーマ追加要件の実現可能性](#2-テーマ追加要件の実現可能性)
3. [工程チェックリスト](#3-工程チェックリスト)
4. [仕様書セクション別の対応状況](#4-仕様書セクション別の対応状況)
5. [推奨する次工程](#5-推奨する次工程)
6. [補足（仕様外だが実施済み）](#6-補足仕様外だが実施済み)

---

## 1. 総括

| 軸 | 内容 | 進捗 |
|----|------|------|
| 基盤（既存） | GGUF 層パック + io_uring ダブルバッファ hybrid | **完了**（本拡張の前提） |
| 軸2 / Phase 1 | 差分アダプタ管理・サイドパス LoRA・`--adapter` | **完了** |
| 軸1 / Phase 2 | MoE Expert 分割パック + 動的 DMA | **完了** |
| 軸3 / Phase 3 | 超軽量ルーターエージェント + メモリ排他 | **完了** |
| 軸2 / Phase 4 | `adapter create` 学習器プロトタイプ | **完了** |
| 軸2 / Phase 5 | 超小型 from-scratch · GGUF 出力 · ローカル SFT/DPO | **完了** |
| 長期 / Phase 6 | 大規模化ブリッジ（リモートジョブ · 変換 · RLHF ステージ） | **完了**（ブリッジ；クラスタ PPO は外部） |
| 拡張 / Phase 7 | 自動知識獲得 & ユーザー適応（Web + auto-train） | **未着手**（条件付き可能） |
| 拡張 / Phase 8 | NVMe 常駐 project-map & 俯瞰記憶 | **未着手**（条件付き可能） |

**いま使えるもの:**  
`lpc-llm run <model> --adapter <name>`（Hybrid LoRA）、  
`lpc-llm run <model> --agent`（SmolLM2 ルーター → アダプタ/Expert ヒント自動選択、RAM 排他）、  
`lpc-llm adapter create --from … --out … --base …`（LoRA SFT → `adapters/<name>/`）、  
`lpc-llm train scratch|sft|dpo|export`（超小型 from-scratch → GGUF → `run`）、  
`lpc-llm job init|run|import|convert`（宣言的ステージ / リモートブリッジ / RLHF スタブ）、  
`lpc-llm config show|init|get`（`config_lpcllm`: bin_dir + ユーザごとの data/train）、  
MoE GGUF では `experts.pack` + Top-K Expert DMA（hybrid）。  
**まだ使えないもの:** プロセス内マルチ GPU PPO、Web 知識獲得、`user_profile` 自動学習、`--project-map`。

---

## 2. テーマ追加要件の実現可能性

テーマ「効率化による限定リソース下での実行とモデル作成」に対し、次の 3 要件をどう扱うか。

| 要件 | 限定リソース下でそのまま？ | 判定 | 本リポでの現実的な落としどころ |
|------|---------------------------|------|--------------------------------|
| ゼロから基盤モデルをフル学習 | 数十億級を家庭用 CPU/少 RAM でフル学習は非現実（計算・データ・電力が桁違い） | **条件付き可能** | まず **超小型（数 M〜数百 M）の from-scratch 学習ループ** を純 Rust/Candle で持つ。大規模は外部計算資源へのジョブ投入 or チェックポイント取込 |
| 数十億パラメータ級の新規 GGUF を一から作る | 「一から学習して数十億 GGUF」は同上。**形式としての GGUF 出力パイプライン**は可能 | **条件付き可能** | (1) 小規模学習結果 → GGUF 書き出し (2) 既存重みの量子化・変換 → `blobs/` 登録。数十億の学習本体はクラスタ前提の別ステージ |
| 本格的な SFT / RLHF パイプライン全体 | フル RLHF（大規模報酬モデル + PPO 等）は GPU 多枚が前提。テーマとは緊張関係 | **条件付き可能** | ローカル向けに **SFT（LoRA/QLoRA）→ 嗜好最適化の軽量版（DPO/ORPO 等）** までをパイプライン化。「本格 RLHF」は段階的・外部アクセラレータ対応として残す |

**結論:** 3 要件とも「エンジニアリングとして追える」が、**現行マシンだけでフルスケール完遂**はテーマと矛盾する。todo には (A) 限定リソースで完結する中間成果物と (B) フルスケールを見据えた長期ステージの両方を載せる。

### Phase 7 / 8 追加要件の実現可能性

| 要件 | 判定 | 前提・落としどころ |
|------|------|-------------------|
| Web 検索 → `cache/knowledge/` 蓄積 | **可能** | DuckDuckGo Instant Answer / HTML スクレイプ / Custom API。対話中は同期・バックグラウンドは非同期ジョブ。知識はチャンク + メタデータ（出典 URL・取得時刻）でローカル保存し、推論時は RAG 的にプロンプトへ注入 |
| ユーザー癖 → `adapters/user_profile/` 自動 LoRA | **条件付き可能** | **Phase 4（`adapter create`）必須**。修正履歴・プロンプトログを `cache/user_logs/` に蓄積 → アイドル検知（Linux: idle 時間 / D-Bus）で差分学習。常時フル学習は避け、小バッチ・低 rank・時間上限付き |
| `--adapter user_profile` 自動アタッチ | **可能** | Phase 1 の Hybrid サイドパスを流用。`run` 開始時に存在すれば自動ロード（再起動不要はプロセス内アタッチの意味；デーモン化は任意） |
| プロジェクト AST/依存グラフ → `map.bin` | **可能** | tree-sitter 等で AST・シンボル・呼び出し辺を抽出。軽量 Embedding（ハッシュ or 小型モデル）をノード属性として付与。全コードを RAM 展開せず NVMe 上の構造化インデックスに保持 |
| `io_uring` オンデマンド・シンボル引出 | **可能** | 既存の層パック DMA と同型。ノード単位の固定長レコード + オフセット表を `O_DIRECT` でプレフェッチ。ミリ秒差分更新は「変更ファイルの再パース → 影響サブグラフのみ書き換え」 |
| `--project-map` 俯瞰コンテキスト | **条件付き可能** | 「数十万行を丸ごとプロンプト」は不可。**関連部分グラフの要約・シグネチャ列**を合成する。Cursor 級の IDE 統合は範囲外；CLI でのグラフ RAG が本リポの現実的な到達点 |

**結論:** Phase 7・8 ともエンジニアリングとして追える。Phase 7 の自動学習は Phase 4 完了後、Phase 8 の DMA 引出は既存 io_uring 基盤の延長。いずれも「限定リソース下で完結する中間成果物」を先に置き、理想仕様（完全自動・全量俯瞰）は段階的に近づける。

---

## 3. 工程チェックリスト

### 0. 既存基盤（仕様の前提・既到達）

- [x] Ollama 非依存の純 Rust（Candle）推論
- [x] `blobs/` / `cache/packs/` / `manifest.json` 分離
- [x] `layers.pack` + `layers.pack.json` による層再配置
- [x] io_uring + O_DIRECT ダブルバッファ・ストリーミング
- [x] `--ram-mib` / `--hot-layers` によるホット層予算
- [x] カタログ（`gemma2:2b`, `smollm2:360m` 等）と CUI（list/pull/run/…）

### Phase 1: 差分アダプタ（LoRA）ロード基盤 — **完了**

- [x] `adapters/` ディレクトリのストレージ管理（`LocalStore`）
- [x] `manifest.json` の `adapters` 索引連動（discover / reconcile / record）
- [x] オンディスク形式 `adapters/<name>/{adapter.json,weights.bin}`（FP16 A/B）
- [x] サイドパス LoRA モジュール（`y = Wq(x) + (α/r)·(x@Aᵀ)@Bᵀ`）
- [x] Hybrid `QMatMul` / Attention・MLP への動的差し込み（`src/adapter/`, `hybrid.rs`）
- [x] アダプタ常駐バイトを hot-layer 予算から控除
- [x] `lpc-llm run <model> --adapter <name>`（指定時 hybrid 強制）
- [x] `lpc-llm adapter list`
- [x] 結合用ゼロフィクスチャ `lpc-llm adapter install-demo`
- [x] ベース `blobs/` / `layers.pack` を書き換えないこと
- [ ] （任意改善）会話途中でのアダプタホットスワップ
- [ ] （任意改善）Eager 経路への LoRA 対応
- [ ] （任意改善）Safetensors / PEFT 形式の読込互換

### Phase 2: MoE パック + Expert ストリーミング — **完了**

- [x] GGUF MoE テンソル解析（`ffn_gate_exps`, `ffn_down_exps` / `ffn_gate.N` 等）
- [x] 常駐（embeddings / norm / lm_head / router）とオンデマンド Expert の分離
- [x] `cache/packs/.../experts.pack` への再レイアウト
- [x] `experts.pack.json` に Expert index / offset テーブル（`layers.pack.json` からも参照）
- [x] Gating Network（ルーター）推論 + Top-K Expert 選抜
- [x] 選抜 Expert の io_uring DMA 発行
- [x] 2× バッファを Expert 単位の動的リング（`PrefetchRing`）へ拡張
- [x] DeepSeek / Mixtral / Qwen-MoE 等のアーキ分岐（`MoeFamily` + 両レイアウト）

### Phase 3: 超軽量ルーターエージェント — **完了**

- [x] `lpc-llm run … --agent` CLI（`--agent-model` でルーター差し替え可）
- [x] SmolLM2 360M（既定）による意図分類プロンプト
- [x] 判定結果 → `--adapter` / Expert prefetch の自動選択（明示 `--adapter` が優先）
- [x] ルーター完了後にメインへコンテキスト引き継ぎ（タイムシェア）
- [x] `--ram-mib` 内でルーター用 KV とメイン用 KV の排他管理（ルーター Engine を drop してからメインロード）

### Phase 4: アダプタ作成器 — **完了**

- [x] `lpc-llm adapter create --from … --out … --base …` の実装  
      （`src/adapter/train.rs` — Hybrid LoRA SFT + AdamW）
- [x] 小規模テキストから数 MB 差分を数分で学習・保存する処理線
- [x] 出力を Phase 1 形式（`adapter.json` + `weights.bin`）に合わせる
- [x] README にビルド / 実行 / 学習データ配置 / アダプタバックアップパスを明記  
      （非公開コーパスは `config_lpcllm` の `train_dir`；成果は `adapters/`）
- [ ] （任意）独立クレート化 / Safetensors 出力

### プライバシー / インストール配置 — **完了**（ユーザ隔離 + GitHub 流出回避）

目標: 共有は **バイナリのみ**；データはホーム配下；非公開コーパスは git ツリーに置かない。

- [x] `config_lpcllm` スキーマ: `[paths]`（`data_dir`, `train_dir`）+ `[install]`（`mode`, `bin_dir`）
- [x] 読込順: 既定 → `/etc/lpc-llm/config_lpcllm` → `~/.config/lpc-llm/config_lpcllm` → `$LPC_LLM_CONFIG` → 環境変数
- [x] CLI: `lpc-llm config show|init|get|example`
- [x] `LocalStore` / `--from` を設定の `data_dir` / `train_dir` で解決（既定 `~/.local/share/lpc-llm/train`）
- [x] ユーザ導入: `scripts/install-dev.sh` → `install.bin_dir`（既定 `~/.local/bin`）
- [x] システム導入: `scripts/install-system.sh` → `/usr/local/bin`（**バイナリのみ**；ユーザデータはコピーしない）
- [x] リポ内 `data/train/` は gitignore 保険のみに降格；公開サンプルは `examples/`
- [x] README / `data/README.md` / `config_lpcllm.example` にプライバシー規約を記載
- [ ] （任意）`mode = "system"` の `/etc/lpc-llm/config_lpcllm` を同梱するパッケージ化

### Phase 5: 限定リソース向け「モデル作成」基盤 — **完了**（テーマ直結・実行可能）

フルスケール 3 要件の **前段**。家庭用〜ワークステーション規模で完結させる。

- [x] 超小型 Transformer の from-scratch 学習ループ（Candle、CPU）— `lpc-llm train scratch`
- [x] 学習チェックポイント → GGUF（F16 llama）書き出し — `train export` / 登録時自動
- [x] 書き出した成果を `blobs/` + `manifest` に登録し `lpc-llm run` で推論
- [x] ローカル SFT パイプライン（tiny のフル微調整；LoRA は Phase 4 `adapter create`）
- [x] 軽量嗜好最適化（DPO）の最小実装 — `lpc-llm train dpo` + `examples/pref-sample.jsonl`
- [x] `--ram-mib` / `--grad-checkpoint` 等、学習時もメモリ上限を意識した設計

### Phase 6: 大規模化ブリッジ — **完了**（条件付き・外部資源前提）

「数十億級」「本格 RLHF」を **このツールチェーンの延長**で扱うための橋。ローカル単機完結は求めない。

- [x] **ゼロから基盤モデルをフル学習する**  
      - 分散/リモート学習ジョブの起動・再開・成果物取込（`job` + `remote.launch`）  
      - データセット仕様・トークナイザ・学習設定の宣言的定義（`job.json`）  
      - 進捗・チェックポイントを `cache/jobs/` + `cache/train/` へ接続
- [x] **数十億パラメータ級の新規 GGUF を一から作る**  
      - 大規模チェックポイント → GGUF 変換ブリッジ（`job convert --backend external` + `$LPC_LLM_CONVERT_CMD`）  
      - builtin tiny ckpt → GGUF + 登録；hybrid pack は初回 `--hybrid` で構築  
      - ※学習計算そのものはリモート/クラスタ側
- [x] **本格的な SFT / RLHF パイプライン全体**  
      - SFT → 嗜好（DPO）→ PPO スタブ → export（`job init --template rlhf`）  
      - 評価・成果の `adapters/` / `blobs/` 出力ステージ  
      - アクセラレータは `remote.launch` / convert cmd 側（io_uring 推論パスは不変）

### Phase 7: 自動知識獲得 & ユーザー適応 — **未着手**（条件付き可能）

Web 知識の非同期獲得と、ユーザー傾向の差分 LoRA 自動更新。  
**依存:** 7.2 / 7.3 の学習本体は **Phase 4（`adapter create`）完了が前提**。7.1 と自動アタッチは Phase 1 だけで着手可。

#### 7.1 Web 検索・ナレッジインジェクション（`search` 連携）

- [ ] 検索バックエンド抽象（DuckDuckGo / Custom HTTP API を差し替え可能に）
- [ ] 対話中の「知識不足」ヒューリスティック（未知エンティティ・明示的検索指示・低信頼応答）
- [ ] バックグラウンド検索ジョブ（非同期取得 → パース → 永続化）
- [ ] `cache/knowledge/` ストア（チャンク本文・出典 URL・取得時刻・タグ）
- [ ] 推論時のナレッジ注入（関連チャンクをプロンプトへ RAG 的に合成；KV 予算を意識）
- [ ] CLI: `lpc-llm search <query>` / `lpc-llm knowledge list|purge`（任意）

#### 7.2 ユーザー癖・文脈の自動アダプタ化（`adapter auto-train`）

- [ ] 会話・修正・プロンプト傾向のローカルログ（`cache/user_logs/`；秘匿・ローテーション方針付き）
- [ ] コーディングスタイル特徴の抽出（インデント・命名・コメント密度など軽量特徴）
- [ ] Linux アイドル検知（X11/Wayland idle または簡易無操作タイマー）
- [ ] アイドル時に Phase 4 学習器を呼び、差分 LoRA を `adapters/user_profile/` へ更新
- [ ] 学習ジョブのガード（時間上限・RAM 上限・最小サンプル数・失敗時ロールバック）
- [ ] CLI: `lpc-llm adapter auto-train [--once|--daemon]`（任意）

#### 7.3 シームレスな自動アタッチ

- [ ] `run` 開始時に `adapters/user_profile/` が有効なら Hybrid サイドパスへ自動組込
- [ ] `--no-user-profile` / `--adapter` 明示指定との優先順位ルール
- [ ] （任意）プロセス内ホットリロード（学習完了後の次回ターンから新重み）
- [ ] （任意改善）会話途中ホットスワップは Phase 1 任意改善と統合

### Phase 8: NVMe 常駐 project-map & 俯瞰記憶 — **未着手**（条件付き可能）

全コードを 16GB RAM に載せない前提で、NVMe 上の構造化グラフから必要ノードだけを `io_uring` で引く。  
**依存:** 既存層パックの `io_uring` / `O_DIRECT` パイプライン。Phase 2 とは独立に着手可（バッファリング戦略は共有しうる）。

#### 8.1 NVMe へのプロジェクトグラフマッピング

- [ ] 言語フロントエンド（tree-sitter 等）でファイル AST・シンボル抽出
- [ ] 関数/クラスの呼び出し・型依存の辺をグラフ化
- [ ] ノード軽量 Embedding（ハッシュ n-gram または小型埋め込み；フル LLM 埋め込みは任意）
- [ ] オンディスク形式 `cache/projects/<hash>/map.bin` + オフセット/索引メタ（`map.json` 等）
- [ ] ファイル更新の監視と **差分インデックス更新**（変更ファイル + 影響辺のみ）
- [ ] CLI: `lpc-llm project-map build|status|rebuild <path>`

#### 8.2 `io_uring` 経由のオンデマンド・シンボル引き出し

- [ ] ノード/エッジの固定長またはチャンク境界レコード設計（`O_DIRECT` 整列）
- [ ] クエリ → 関連ノード集合の選定（BM25 / Embedding 近傍 / グラフ近傍の組合せ）
- [ ] 選定ノードの `io_uring` プレフェッチ → RAM リングバッファ
- [ ] コンテキスト組立時のトークン予算キャップ（ホット層予算と整合）

#### 8.3 `--project-map` 広域コンテキスト俯瞰

- [ ] `lpc-llm run … --project-map [<path|hash>]` CLI
- [ ] 呼び出し関係・型依存を **部分グラフ要約**としてプロンプトへ合成（全量貼付はしない）
- [ ] リファクタ/生成向けの構造的ヒント（依存先シグネチャ列・影響範囲）
- [ ] 16GB 級 RAM でも数十万行規模を「構造として」扱えることの回帰ベンチ（任意）

---

## 4. 仕様書セクション別の対応状況

### データレイアウト

| パス | 仕様 | 現状 |
|------|------|------|
| `blobs/` | ベース GGUF | 既存どおり |
| `adapters/` | 差分モジュール | **実装済**（ディレクトリ + json/bin）；**バックアップ対象** |
| `adapters/user_profile/` | 自動学習ユーザー LoRA | **未実装**（Phase 7） |
| `paths.train_dir`（ホーム） | `adapter create` / `train` 用非公開コーパス | **実装済**（既定 `<data_dir>/train`；`config_lpcllm`） |
| `data/train/`（リポ内） | 開発用の残り場所のみ | **gitignore 保険** — 非公開データを置かない |
| `cache/packs/.../layers.pack` | ベース層パック | 既存（名称は `layers.pack`、仕様の `base_layers.pack` 改名は未実施） |
| `cache/packs/.../experts.pack` | MoE Expert パック | **実装済** |
| `cache/knowledge/` | Web 取得ナレッジ | **未実装**（Phase 7） |
| `cache/user_logs/` | 癖学習用ログ | **未実装**（Phase 7） |
| `cache/projects/<hash>/map.bin` | プロジェクト構造グラフ | **未実装**（Phase 8） |
| `manifest.json` | models + adapters | **adapters キー追加済** |

### CLI

| コマンド | 現状 |
|----------|------|
| `run … --adapter <name>` | **実装済** |
| `run … --agent` | **実装済**（`--agent-model` 付き） |
| `run … --project-map` | **未実装**（Phase 8） |
| `adapter list` | **実装済** |
| `adapter install-demo` | **実装済**（検証用） |
| `config show\|init\|get\|example` | **実装済**（`config_lpcllm` のパス / 導入レイアウト） |
| `adapter create …` | **実装済**（LoRA SFT → Phase 1 形式；`--from` は `train_dir` 解決） |
| `train scratch|sft|dpo|export` | **実装済**（Phase 5 超小型学習 → GGUF → `run`） |
| `job init|run|status|import|convert` | **実装済**（Phase 6 ブリッジ） |
| `adapter auto-train` | **未実装**（Phase 7） |
| `search` / `knowledge …` | **未実装**（Phase 7） |
| `project-map build|status` | **未実装**（Phase 8） |

### メモリ・I/O パイプライン

| 項目 | 現状 |
|------|------|
| 層単位 pack + ping-pong DMA | 既存 |
| LoRA サイドパス（計算時アタッチ） | **実装済**（DMA バッファは非破壊） |
| Expert 単位インデックス / 動的 DMA | **実装済**（`experts.pack` + `PrefetchRing`） |
| project-map ノード単位 `io_uring` プレフェッチ | **未実装**（Phase 8） |
| CQE 時の ΔW マージ（重み書き換え） | 採用せず（サイドパス方針） |

---

## 5. 推奨する次工程

1. **Phase 7** — Web 知識獲得 → `user_profile` 自動学習・自動アタッチ（ログはユーザ `cache/` のみ）
2. **Phase 8** — `project-map` 索引 + `io_uring` オンデマンド引出 + `--project-map`
3. **Phase 6 フォロー** — `job.remote` / `$LPC_LLM_CONVERT_CMD` に実クラスタ・CUDA 変換を接続
4. **（任意）** `install.mode = "system"` の `/etc/lpc-llm/config_lpcllm` を同梱するパッケージ化

---

## 6. 補足（仕様外だが実施済み）

- [x] 日本語入力時の Backspace が UTF-8 バイト欠けする問題への対処（REPL を `rustyline` 化）
- [x] 開発用 PATH 導入 `scripts/install-dev.sh`（symlink → `target/debug`；`config_lpcllm` の `bin_dir` 対応）
- [x] 共有バイナリ導入 `scripts/install-system.sh`（バイナリのみ）
- [x] プライバシー: 非公開コーパスは `train_dir`（ホーム/XDG）；リポ `data/train/` は gitignore 保険；公開サンプルは `examples/`
