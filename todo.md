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
Last updated: 2026-08-05

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
| Extension / Phase 7 | Auto knowledge acquisition & user adaptation (Web + auto-train) | **Done** (conditionally feasible) |
| Extension / Phase 8 | NVMe-resident project-map & overview memory | **Done** (conditionally feasible) |
| Extension / Phase 9 | Compute device selection + Candle-stack Vulkan offload | **Done** (first landing) |
| Extension / Phase 10 | Vulkan real speedup (quantized shaders + VRAM-resident hot weights) | **In progress** (Q4_K GEMV + warm; e2e decode → Phase 13) |
| Extension / Phase 11 | Gemma 3 largest (27B) hybrid **text** run | **Done** (text hybrid verified) |
| Extension / Phase 12 | Gemma 4 **26B-A4B (MoE)** hybrid text run | **Done** (text hybrid verified; latency polish → Phase 13) |
| Extension / Phase 13 | Decode throughput (MoE + Vulkan end-to-end) | **In progress** (~2× landed; ≥3× → Phase 14) |
| Extension / Phase 14 | Accuracy-preserving decode speed (≥3× / stretch tok/s) | **Planned** (Softmax GPU · layer pipeline · conditional Q8_0) |

**Available now:**  
`lpc-llm run <model> --adapter <name>` (Hybrid LoRA),  
`lpc-llm run <model> --agent` (SmolLM2 router → auto adapter/expert hints, exclusive RAM),  
`lpc-llm adapter create --from … --out … --base …` (LoRA SFT → `adapters/<name>/`),  
`lpc-llm train scratch|sft|dpo|export` (tiny from-scratch → GGUF → `run`),  
`lpc-llm job init|run|import|convert` (declarative stages / remote bridge / RLHF stubs),  
`lpc-llm config show|init|get` (`config_lpcllm`: bin_dir + per-user data/train),  
`lpc-llm search` / `knowledge list|purge` / `adapter auto-train` (Phase 7),  
`lpc-llm project-map build|status|rebuild` / `run --project-map` / `--knowledge` / `--no-user-profile` (Phase 7–8),  
On MoE GGUF: `experts.pack` + Top-K expert DMA + **RAM LRU expert cache** (hybrid).  
`lpc-llm setup` / `run --device` (Phase 9; Vulkan Q4_K when VRAM-cached).  
`lpc-llm run gemma3:27b --hybrid` (Phase 11 dense text verified).  
**Phase 12 verified:** `gemma4:26b-a4b --hybrid --ram-mib 16384 --device vulkan` text chat works. Prefer **`cargo build --release`**. First-turn TTFT still heavy; **decode ~1 char/s** observed — throughput work is **Phase 13**.  
**Phase 9 (landing):** setup → `[ui]`/`[runtime]` in home `config_lpcllm`.  
**Phase 10 (parallel; throughput owned by 13):** Q4_K GEMV + VRAM warm exist; per-call fence still dominates end-to-end decode.  
**Phase 11 (done):** Gemma 3 **27B** dense instruct via `--hybrid` (text-only).  
**Phase 12 (done):** Gemma 4 **26B-A4B MoE** (~25.2B / ~3.8B active); resident target ≤16 GiB.  
**Phase 13 (in progress):** decode ~**1.8–2.4 tok/s** (~2–2.4× vs ~1 baseline) with DeviceAct, fused Top-K gate/up, shared∥expert, Gemma4 `>>>` UI. Stretch **≥3×** and conversation-class tok/s → **Phase 14** (same weights / no coarser quant).
**Phase 14 (planned):** accuracy-preserving speed — Softmax/RoPE-adjacent fence cut, cross-layer GPU∥CPU overlap, conditional fused Q8_0 downs; success ≥3× on warm turn 2+.

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

### Phase 7: Auto knowledge acquisition & user adaptation — **Done** (conditionally feasible)

Async Web knowledge acquisition and automatic delta-LoRA updates from user tendencies.  
**Deps:** Training for 7.2 / 7.3 requires **Phase 4 (`adapter create`)**. 7.1 and auto-attach can start with Phase 1 alone.

#### 7.1 Web search · knowledge injection (`search` integration)

- [x] Search backend abstraction (DuckDuckGo / Custom HTTP via `LPC_LLM_SEARCH_*`; `curl` transport)
- [x] In-chat “knowledge gap” heuristics (unknown entities, explicit search, low-confidence cues)
- [x] Background search jobs (thread fetch → parse → persist)
- [x] `cache/knowledge/` store (chunk body · source URL · fetch time · tags)
- [x] Knowledge inject at inference (`--knowledge`; RAG-style chunks; char budget)
- [x] CLI: `lpc-llm search <query>` / `lpc-llm knowledge list|purge`

#### 7.2 Auto-adapterize user habits / context (`adapter auto-train`)

- [x] Local logs of chats / edits / prompt tendencies (`cache/user_logs/`; privacy + rotation)
- [x] Extract coding-style features (indent · naming · comment density, etc.)
- [x] Linux idle detect (xprintidle / GNOME IdleMonitor / wall-clock fallback)
- [x] On idle, call Phase 4 trainer and update delta LoRA under `adapters/user_profile/`
- [x] Job guards (time cap · RAM cap · min samples · rollback on failure)
- [x] CLI: `lpc-llm adapter auto-train [--once|--daemon]`

#### 7.3 Seamless auto-attach

- [x] At `run` start, if `adapters/user_profile/` is valid, auto-wire into Hybrid side-path
- [x] Priority rules vs `--no-user-profile` / explicit `--adapter` (explicit > agent > user_profile)
- [ ] (Optional) In-process hot reload (new weights from next turn after train)
- [ ] (Optional) Mid-chat hot-swap merges with Phase 1 optional work

### Phase 8: NVMe-resident project-map & overview memory — **Done** (conditionally feasible)

Without loading all code into 16GB RAM, pull only needed nodes from a structured graph on NVMe via `io_uring`.  
**Deps:** Existing layer-pack `io_uring` / `O_DIRECT` pipeline. Can start independent of Phase 2 (buffering strategy may be shared).

#### 8.1 Map project graph onto NVMe

- [x] Language frontends (pure-Rust heuristics for Rust/Python/JS/TS/Go/C-family; no tree-sitter/C)
- [x] Graph call / type-dependency edges for functions/classes
- [x] Light node embeddings (hashed n-gram; full LLM embed optional)
- [x] On-disk `cache/projects/<hash>/map.bin` + offset/index meta (`map.json`)
- [x] File mtime fingerprints in `map.json`; `rebuild` for clean refresh (full re-walk for cross-file edges)
- [x] CLI: `lpc-llm project-map build|status|rebuild <path>`

#### 8.2 On-demand symbol fetch via `io_uring`

- [x] Fixed-length / chunk-boundary records for nodes (`O_DIRECT` aligned; buffered fallback)
- [x] Query → related node set (BM25-ish / embedding neighborhood / graph neighborhood)
- [x] `io_uring` prefetch of selected nodes → RAM ring (`PrefetchRing`; buffered fallback)
- [x] Token/char-budget cap when assembling context

#### 8.3 `--project-map` wide-context overview

- [x] `lpc-llm run … --project-map [<path|hash>]` CLI
- [x] Synthesize call/type deps as **subgraph summaries** into the prompt (no full dump)
- [x] Structural hints for refactor/codegen (callee signature lists · impact scope)
- [ ] Regression bench that tens/hundreds of kLOC can be handled “as structure” on ~16GB RAM (optional)

### Phase 9: Compute device selection + Candle-stack Vulkan — **Done** (first landing)

Generic accelerator selection (CPU / CUDA / Vulkan / auto) via first-run i18n Q&A; persist to home `config_lpcllm`.  
Vulkan compute offload for quantized MatMul on the Candle inference stack (ash + SPIR-V; no Candle `Device` fork).  
**Deps:** Phase 1 hybrid `QMatMul` path. CUDA path is feature-gated (`--features cuda`).

#### 9.1 First-run setup (i18n) + config

- [x] `lpc-llm setup` / `config init --interactive` — one-question-at-a-time (language → compute device)
- [x] Persist `[ui] language` (`ja`/`en`) and `[runtime] device` (`auto`/`cpu`/`cuda`/`vulkan`) under `~/.config/lpc-llm/config_lpcllm`
- [x] Env overrides: `LPC_LLM_LANGUAGE`, `LPC_LLM_DEVICE`; CLI `--device` on `run`
- [x] Gate: prompt setup when user config missing or `runtime.device` unset (skippable → CPU for session)

#### 9.2 Device resolve + Vulkan backend

- [x] `ComputeBackendKind` resolve: `auto` → Vulkan if present, else CUDA (feature), else CPU; fail-soft to CPU
- [x] ash Vulkan: instance/device/queue + buffer pool + f32 GEMM SPIR-V (WGSL→SPIR-V via naga build.rs)
- [x] QMatMul hot path: dequant (Candle `QTensor`) + Vulkan GEMM; unsupported → CPU `QMatMul::forward`
- [x] Wire Hybrid / Eager load + ready banner (`Vulkan+pack+io_uring` / `CPU+…`)

### Phase 10: Vulkan real speedup (quantized shaders + VRAM hot weights) — **In progress**

Phase 9 proves the Vulkan path and may show GPU utilization, but decode often feels **no faster** (or slower) than Candle CPU: every QMatMul still does full CPU dequant → upload f32 weights → naive f32 GEMM → download.  
**Goal:** make `--device vulkan` competitive with (then faster than) `--device cpu` for hybrid decode.  
**Deps:** Phase 9 ash / SPIR-V stack + Hybrid hot-layer pin.

#### 10.1 GPU-side quantized MatMul (stop per-call full dequant)

- [x] SPIR-V / WGSL shaders for **dequant + GEMV/GEMM** on common GGUF types (at least **Q4_K**; then Q5_K / Q8_0 as needed)
- [x] Keep weights **quantized on device**; do not CPU-dequantize the full matrix on every forward (Q4_K path)
- [x] Avoid uploading the entire f32 weight matrix each call (activation + tiny staging only, or resident buffers)
- [x] Soft-fallback to Candle CPU `QMatMul::forward` for unsupported dtypes (log once / `vulkan-skip:`)

#### 10.2 VRAM-resident hot quantized weights

- [x] Cache quantized Q4_K blobs in VRAM keyed by `Arc<QTensor>` (hot layers + MoE experts after materialize)
- [x] Explicit `warm_q4k` after hot-layer pin and after expert materialize (small-batch decode can hit GPU)
- [ ] Streamed (non-hot) layers: optional path later; first target is hot-resident path that dominates TTFT after warmup
- [x] Explicit teardown on `VulkanContext` Drop (cache drain)
- [x] Raise VRAM weight-cache cap for MoE (768 entries)

#### 10.3 CPU baseline & measurement (A/B)

- [x] Document that **`--device cpu` may still be faster** until 10.1–10.2 land; use it as the comparison baseline
- [x] Unit A/B: CPU reference GEMV vs Vulkan Q4_K shader (`q4k_gpu_matches_cpu_when_vulkan_available`)
- [x] Startup GEMV microbench: if GPU slower, prefer Candle CPU for Q4_K; scratch buffer reuse for submits
- [x] Remove hard `hot≤8` cap so `--ram-mib` can pin all layers of small models (biggest TTFT win for gemma2:2b)
- [ ] Optional: warn at startup when residual non-Q4_K tensors dominate the model

### Phase 11: Gemma 3 largest (27B) hybrid text run — **Done** (text hybrid verified)

Prove the project theme on a **dense** multi-billion model: run Gemma 3 Instruct **27B** (Q4-class GGUF) with RAM + NVMe layer-pack hybrid, without loading the full weight set into RAM.  
**Success:** `lpc-llm run gemma3:27b --hybrid` completes **text** chat on a home/workstation RAM budget.  
**Out of success criteria:** image input / vision encoder (optional leftover or later phase).  
**Deps:** existing hybrid pack + io_uring; Gemma2 path as baseline; Phase 10 optional for speed.

#### 11.1 Catalog + prompt

- [x] Catalog entries: `gemma3:4b`, `gemma3:12b`, `gemma3:27b` (HF GGUF + tokenizer)
- [x] Confirm `PromptStyle::Gemma` (`<start_of_turn>` / `<end_of_turn>`) works for Gemma 3 IT
- [x] Document approx size / `--ram-mib` hints for 27B Q4 (~15–20 GB weights; stream layers)

#### 11.2 GGUF / architecture (vs Gemma 2)

- [x] Parse `gemma3` metadata: sliding window, **sliding_window_pattern** (5 local : 1 global), dual RoPE bases (local vs global)
- [x] Materialize **attn_q_norm** / **attn_k_norm**; keep post-attn / post-ffw norms
- [x] Per-layer attention: local SWA mask + KV trim vs global full causal; no Gemma2-style attn softcap when absent
- [x] Soft-fail clearly if multimodal projector tensors are present but vision path is not implemented

#### 11.3 Hybrid memory for 27B

- [x] Layer pack + hot selection fit 27B Q4 under `--ram-mib` (stream non-hot; embeddings / norms / lm_head resident)
- [x] KV / context budget coherent with rope table length and RAM hint (first landing may cap ctx below 128K)
- [x] Startup hints when `stream > 0` (already present) and recommended `--ram-mib` for 27B

#### 11.4 Verification (27B-first)

- [x] **`gemma3:27b --hybrid` text dialogue** (phase exit criterion; start here)
- [ ] (Optional) `gemma3:4b` / `12b` smoke only if 27B debug needs a smaller repro
- [ ] (Optional) Vision / multimodal Gemma 3 — deferred

#### 11.5 Quant / compute notes

- [x] Q4_K_M path preferred; Q4_0 QAT falls back to Candle CPU MatMul (Phase 10 is Q4_K-first)
- [x] Bump `tokenizers` to 0.21 so Gemma 3 `tokenizer.json` (array merges) loads
- [ ] (Optional) A/B `--device cpu` vs `vulkan` on 4B after Phase 10

### Phase 12: Gemma 4 26B-A4B (MoE) hybrid text run — **Done** (text hybrid verified; latency → Phase 13)

Gemma 3 has **no MoE 27B**. The Gemma-family MoE near that class is **Gemma 4 26B-A4B** (~25.2B total / ~3.8B active per token; 128 routed experts, Top-8, + shared expert).  
**Goal:** run it under the theme (RAM + NVMe hybrid): attention/router/shared in `layers.pack`, routed experts in `experts.pack` with Top-K DMA — so **resident RAM can stay ≤16 GiB** while decode compute tracks ~4B-active, not dense 27B.  
**Success:** `lpc-llm run gemma4:26b-a4b --hybrid --ram-mib 16384` completes **text** chat with process RSS meaningfully under ~16 GiB (not “hot≈all layers” like dense 27B today).  
**Out of success criteria (first landing):** image / vision encoder; full 256K context; smartphone-class TTFT.  
**Deps:** Phase 2 MoE pack + ring; Phase 11 Gemma SWA / QK-norm experience; tighten `--ram-mib` accounting (emb / KV / slots).  
**Note:** Alternate MoE of similar class (`qwen3:30b-a3b`, ~30.5B/3.3B active) may be used as a pathfinder if Gemma 4 GGUF/arch lands slower — document in checklist, do not replace Phase 12 success criterion.

#### 12.1 Catalog + artifacts

- [x] Catalog entry `gemma4:26b-a4b` (IT, Q4_K_M GGUF + tokenizer; bartowski + unsloth)
- [x] Confirm prompt / special tokens for Gemma 4 IT (`PromptStyle::Gemma4` / `<|turn>` … `<turn|>`)
- [x] Document disk size vs **active** params vs `--ram-mib` (catalog hints + README; experts on NVMe)

#### 12.2 GGUF / architecture (Gemma 4 MoE)

- [x] Parse `gemma4` / MoE metadata (expert_count, expert_used_count, shared expert, sliding/global pattern, dual head dims)
- [x] Extend `MoeFamily` / layout for Gemma 4 (`FusedGateUpTrailing` + shared dense FFN)
- [x] Reuse / extend SWA + dual RoPE + QK-norm patterns from Phase 11 (per-layer head_dim / n_kv; partial RoPE; contiguous cos/sin)
- [x] Soft-warn or skip vision tensors (text GGUF has no `v.*`; mmproj separate)
- [x] Prefer GGUF `expert_count` metadata (do not confuse with `expert_feed_forward_length` / 704)
- [x] Candle-reversed trailing-expert slice + gate_up split (experts.pack **v3**)

#### 12.3 Hybrid memory ≤16 GiB

- [x] Treat `--ram-mib` as a **hard-ish resident ceiling**: reserve embeddings (f16), lm_head, KV headroom, 2× layer slots, MoE expert ring
- [x] Core layers (attn / norms / router / shared expert) via `layers.pack`; **routed experts only via `experts.pack` Top-K**
- [x] Avoid f32 full dequant of huge embedding tables on Gemma 4 (f16 embedding path; F32 activations before rms_norm)
- [x] Startup log: estimated resident MiB, hot layers, expert ring slots, stream counts; numbered `[1/5]…` phases + counters

#### 12.4 Latency (MoE-active path)

- [x] Prefetch next Top-K experts overlapping compute (existing `PrefetchRing`; Top-8 slots)
- [x] RAM LRU of materialized experts (capacity from `--ram-mib` spare; survives KV reset across turns)
- [x] VRAM `warm_q4k` for hot layers + cached experts (Vulkan small-batch path)
- [ ] Measure TTFT / tok/s vs `gemma3:27b` dense under same `--ram-mib 16384` (**release** binary) → **moved to Phase 13.1**
- [ ] (Optional) Pathfinder A/B with `qwen3:30b-a3b` on existing Qwen-MoE path if Gemma 4 GGUF delayed → **Phase 13.5**

#### 12.5 Verification

- [x] Pack build: `layers.pack` + `experts.pack` for Gemma 4 26B-A4B (experts=128, top-k=8, pack v3)
- [x] **`gemma4:26b-a4b --hybrid --ram-mib 16384` text dialogue** (phase exit; release recommended)
- [ ] Confirm RSS / hot selection respects ~16 GiB intent (document measurement method) → **Phase 13.1**
- [ ] (Optional) Vision / multimodal Gemma 4 — later phase

**Pull isolation:** blobs under `~/.local/share/lpc-llm/blobs/<hf-repo>/` (XDG user data). Each catalog model is an independent module; `lpc-llm pull` resumes `.part` and does not require stopping another model’s `run`.

**Ops tip:** debug builds make MoE prefill minutes-long; use `./target/release/lpc-llm`. stderr shows prefill layer progress and `expert cache hits/misses`.

### Phase 13: Decode throughput (MoE + Vulkan end-to-end) — **In progress**

Phase 12 proved **correctness** (`gemma4:26b-a4b` text chat). Observed **decode ≈ 1 char/s** on release + Vulkan is not conversation-usable.  
Microbench already shows Q4_K GEMV GPU≈0.18 ms vs CPU≈0.59 ms, yet end-to-end is still crawling — so the bottleneck is **pipeline / sync / I/O**, not “missing a GEMV shader”.

**Field evidence (2026-08-04, release, `--hybrid --ram-mib 16384 --device vulkan --max-tokens 128`):**

| Signal | Observation | Implication |
|--------|-------------|-------------|
| Prefill | 16 prompt tok × 30 layers in **18.7 s** | ~1.2 s/layer; heavy but landed |
| Expert cache after prefill | **1177/2048**, hits=0 / misses=1177 | first turn fully cold; capacity not full |
| Decode feel | **~1 char/s** | ~1 tok/s class if JP chars ≈ tokens; far below ~3.8B-active potential |
| `mlock` | failed (os error 12 / RLIMIT_MEMLOCK) | unlocked arenas → page-fault risk under load |
| GEMV probe | GPU wins when VRAM-cached | GPU *kernels* are fine; **per-call submit+fence** likely dominates |

**Root-cause hypothesis (ordered by expected leverage):**

1. **Vulkan per-QMatMul sync** — each GEMV does activation host↔device + `queue_submit` + `wait_for_fences`. Decode (m=1) issues **hundreds of QMatMuls/token** (attn Q/K/V/O + shared FFN + Top-8×{gate,up,down} × 30 layers). Fence overhead swamps the 0.18 ms kernel.
2. **CPU↔GPU ping-pong** — activations leave GPU every call (`to_vec1` / rebuild `Tensor`); attention / norms / Softmax stay on Candle CPU.
3. **MoE Expert I/O on miss** — Top-8 × 30 = up to 240 expert touches/token; cold miss = NVMe DMA + materialize + `warm_q4k` (sync). Prefill warmed 1177 experts; decode hit-rate unknown (need per-phase counters).
4. **Unlocked arenas** — `mlock` failure amplifies jitter when RSS is large.
5. **Residual non-Q4_K / uncached weights** — silent `vulkan-skip:` → Candle CPU (Phase 10 leftover).

**Goal:** make 2nd+ turn decode on `gemma4:26b-a4b` **conversation-speed** under the same ≤16 GiB resident theme.  
**Success (release binary, after 1 warm turn):** measure and publish TTFT / tok/s; achieve **≥2×** vs documented Phase-13 cold baseline on the same host (**landed ~1.8–2.4×**). Stretch **≥3× / ≥8–15 tok/s** → **Phase 14**.  
**Out of success:** smartphone-class TTFT; full 256K ctx; vision; matching llama.cpp absolute tok/s.  
**Deps:** Phase 10 Q4_K path; Phase 12 MoE hybrid; folds remaining Phase 12.4 measurements + Phase 10 polish.

#### 13.0 Ops baseline (do first — no code)

- [x] Raise memlock soft→hard when allowed; **default path needs no `ulimit`** (unlocked arenas)
- [x] Document: judge speed on **turn 2** (cache warm); stderr `[bench]` line
- [ ] Host A/B `--device cpu` vs `--device vulkan` on release (user machine; decode now CPU on both unless probe wins)
- [ ] Note RSS / expert cache fill after turn 1 (`/proc/<pid>/status` VmRSS + stderr cache line)

#### 13.1 Measurement harness (absorb Phase 12.4 polish)

- [x] stderr `[bench]`: **TTFT**, **decode tok/s**, prefill vs decode split
- [x] Expert cache **hits/misses during decode only** (counters reset after prefill)
- [x] Vulkan submit / skip counters on the bench line; first `vulkan-skip` reason logged once
- [ ] Document numbers vs `gemma3:27b` dense under `--ram-mib 16384` (host run)
- [ ] Confirm RSS / hot selection respects ~16 GiB intent (from Phase 12.5)

#### 13.2 Vulkan decode path — cut sync (highest leverage)

- [x] **Fuse command buffers**: multiple Q4_K/Q6_K GEMVs per submit (Q/K/V, gate/up; up to 16 ops)
- [x] **DeviceAct** + pack activations once per fused group; HOST barrier only after last op; multi-X downs API
- [x] Reuse staging / multi descriptor sets
- [x] GPU for VRAM-warmed weights (hot pin + MoE experts); cold → Candle CPU
- [x] **Q6_K Vulkan shader** (bit-exact vs candle `BlockQ6K::to_float`) — kills Q6_K→CPU skip on attn/down
- [x] **Expert GPU path**: warm Top-K gate/up into VRAM; fuse shared-activation gate+up (≤16 ops/submit); **Q8_0 downs → parallel CPU** (GPU sync loses on k≈704)

#### 13.3 MoE decode I/O

- [x] Skip DMA on cache hit; async prefetch without immediate wait; ring slot occupancy
- [x] Decode-phase hit-rate: after warm turn, **~98%** expert RAM hits on short chats (2026-08-05: hits=2359/misses=41)
- [x] Document opportunistic `mlock` (no `ulimit` required); soft RLIMIT_MEMLOCK raise when allowed
- [ ] Tune ring depth / LRU vs `--ram-mib` spare (2048 slots ≈8.4 GiB — validate thrash vs hit)
- [x] Speculative prefetch from previous-token Top-K ids

#### 13.4 Residual compute (Phase 10 leftovers)

- [x] Log first `vulkan-skip` reason (dtype / policy / cold weight) once + skip counter
- [x] Q8_0 expert-down: shader present (`q8_0_gemv.wgsl`) but **forward prefers Candle CPU** (sync GPU slower for Gemma4 downs); enable when microbench wins
- [ ] (Optional) streamed non-hot layer GPU path — low priority while hot=all 30 for 26B-A4B
- [x] attention Softmax / RoPE stay CPU (fused graph not justified yet)

#### 13.5 Verification

- [x] Bench method documented in README
- [x] Regression smoke: short prompt + decode; turn-2 expert hits (2026-08-05: ~1.8 tok/s ≈2× baseline; submits≈½)
- [ ] (Optional) pathfinder `qwen3:30b-a3b` A/B on same harness

**Host A/B (2026-08-05, release, `--hybrid --ram-mib 16384 --device vulkan`):**

| Turn / note | decode tok/s | expert hits/misses | notes |
|-------------|-------------|-------------------|--------|
| Phase-13 baseline (2026-08-04) | **~1** | cold | documented ~1 char/s |
| After fuse+DeviceAct (turn 2, short) | **1.81** | 2359/41 | ≈2×; submits≈½ |
| + shared∥expert, attn DeviceAct (turn 2, short) | **~2.4** | high | 3-tok replies; optimistic |
| + same (turn 2, longer, `--max-tokens 48`) | **1.82** | 1594/86 | sustained ≈2× |
| Stretch **≥3×** | — | — | **moved to Phase 14** (Softmax GPU / layer pipeline / conditional Q8_0) |

Gemma4 UI: prompt opens empty `<|channel>thought`<channel|>`; stream strips channel leakage; answer prefix **`>>> `** (first visible token).

**Pull isolation / ops:** unchanged from Phase 12. Always measure with `./target/release/lpc-llm`.

### Phase 14: Accuracy-preserving decode speed (≥3×) — **Planned**

Phase 13 delivered **~2×** warm decode without changing GGUF quant or Top-K. Remaining latency is still **pipeline / sync**, not “wrong weights.” Phase 14 keeps **bit-equivalent matmuls** (same Q4_K / Q6_K / Q8_0 tensors; same Top-8) and only removes host waits and CPU/GPU idle gaps.

**Constraint (hard):** no coarser quant, no Top-K reduction, no speculative token accept/reject that can change answers. Numeric path must match Phase-13 CPU/GPU references (existing unit tests + spot A/B logits).

**Baseline (host, release, warm turn 2+):** Phase-13 sustained **~1.8 tok/s**; short-burst up to ~2.4. Phase-13 documented cold baseline **~1 tok/s**.

**Goal:** conversation-usable decode under the same ≤16 GiB hybrid theme.  
**Success:** warm turn 2+ decode **≥3×** vs Phase-13 cold baseline (~1 → ≥3 tok/s) on the same host/command; publish `[bench]` + submits/token.  
**Stretch:** **≥8 tok/s** if fused attention+FFN graph + dual-scratch overlap land (host-dependent).  
**Out of success:** matching llama.cpp absolute tok/s; coarser quant “wins”; vision; full 256K.

**Deps:** Phase 13 DeviceAct / multi-X / fused gate/up; Q8_0 shader already in-tree (forward gated to CPU).

#### 14.0 Measurement gate (no code)

- [ ] Record warm turn-2 `[bench]` before any Phase-14 merge (tok/s, submits, expert hits, skips)
- [ ] Fix compare command:  
  `./target/release/lpc-llm run gemma4:26b-a4b --hybrid --ram-mib 16384 --device vulkan --max-tokens 48`
- [ ] Optional: same prompt A/B logits (max abs diff / KL) CPU-ref vs new path on a tiny layer fixture

#### 14.1 Cut attention-side fences (highest leverage)

- [ ] Keep QKV fused (done in 13); reduce **O / Softmax boundary** round-trips
- [ ] Softmax (and optionally scaled QKᵀ / AV) on GPU for decode `m=1` when VRAM-resident Q/K/V activations allow — **same f32 math as Candle**
- [ ] DeviceAct (or dual Y scratch) so Softmax input never returns to a Candle `Tensor` until residual / next norm needs host
- [ ] RoPE stays CPU unless a verified GPU RoPE lands; do not change rope tables

#### 14.2 Cross-layer GPU ∥ CPU overlap

- [ ] Dual fence + dual scratch (ping-pong): next submit must not wait on prior D2H when buffers differ
- [ ] Decode pipeline: overlap **layer i expert/shared Q8_0 CPU downs** with **layer i+1 QKV GPU** once residual for i+1 is known — or overlap any GPU-idle window with useful CPU work
- [ ] No change to layer math order that would alter numerics (associativity of parallel independent ops only)

#### 14.3 Conditional fused Q8_0 downs

- [ ] Keep `q8_0_gemv.wgsl`; enable forward **only when** fused multi-X microbench beats parallel Candle CPU (Gemma4 down `n≈2816`, `k≈704`, Top-8)
- [ ] Policy: single tiny Q8_0 GEMV stays CPU; **≥4 fused downs / one submit** may use GPU
- [ ] Bit-exact vs candle `BlockQ8_0::to_float` (unit test already-shaped like Q6_K)

#### 14.4 MoE / I/O polish (accuracy-neutral)

- [ ] Hold warm-turn expert RAM hit-rate **≥95%** on short chats; avoid VRAM thrash of gate/up
- [ ] Skip redundant `warm_*` when cached (done for experts; audit shared / attn)
- [ ] Document RSS / `--ram-mib 16384` method (absorb remaining 13.1)

#### 14.5 Out of scope (would risk quality or theme)

- Coarser GGUF (Q3_K etc.), lowering `expert_used_count`, draft/speculative decoding with accept rules that can diverge
- Full Candle Vulkan `Device` rewrite (nice-to-have later, not required for ≥3×)
- llama.cpp parity chase

#### 14.6 Verification

- [ ] Warm turn-2 tok/s ≥3× vs ~1 baseline; note submits/token drop
- [ ] Spot logit A/B on fixed prompt (no user-visible answer drift)
- [ ] Regression: Gemma4 `>>>` UI + no bare `thought` leak
- [ ] Update this file + README bench blurb with host numbers

---

## 4. Spec section status

### Data layout

| Path | Spec | Status |
|------|------|------|
| `blobs/` | Base GGUF | As existing |
| `adapters/` | Delta modules | **Implemented** (dir + json/bin); **back up** |
| `adapters/user_profile/` | Auto-train user LoRA | **Implemented** (Phase 7 `adapter auto-train` + auto-attach) |
| `paths.train_dir` (home) | Private corpora for `adapter create` / `train` | **Implemented** (default `<data_dir>/train`; via `config_lpcllm`) |
| `data/train/` (repo) | Dev leftover only | **gitignore safety net** — do not store private data here |
| `cache/packs/.../layers.pack` | Base layer pack | Existing (name is `layers.pack`; rename to spec’s `base_layers.pack` not done) |
| `cache/packs/.../experts.pack` | MoE expert pack | **Implemented** |
| `cache/knowledge/` | Web-fetched knowledge | **Implemented** (Phase 7) |
| `cache/user_logs/` | Habit-train logs | **Implemented** (Phase 7) |
| `cache/projects/<hash>/map.bin` | Project structure graph | **Implemented** (Phase 8) |
| `manifest.json` | models + adapters | **`adapters` key added** |

### CLI

| Command | Status |
|----------|------|
| `run … --adapter <name>` | **Implemented** |
| `run … --agent` | **Implemented** (with `--agent-model`) |
| `run … --project-map` | **Implemented** (Phase 8; path or hash) |
| `run … --knowledge` / `--no-user-profile` | **Implemented** (Phase 7) |
| `adapter list` | **Implemented** |
| `adapter install-demo` | **Implemented** (for verification) |
| `config show\|init\|get\|example` | **Implemented** (`config_lpcllm` path / install layout) |
| `adapter create …` | **Implemented** (LoRA SFT → Phase 1 on-disk form; `--from` via `train_dir`) |
| `train scratch|sft|dpo|export` | **Implemented** (Phase 5 tiny train → GGUF → `run`) |
| `job init|run|status|import|convert` | **Implemented** (Phase 6 bridge) |
| `adapter auto-train` | **Implemented** (Phase 7) |
| `search` / `knowledge …` | **Implemented** (Phase 7) |
| `project-map build|status|rebuild` | **Implemented** (Phase 8) |
| `setup` / `config init --interactive` | **Implemented** (Phase 9 i18n device wizard) |
| `run … --device <auto\|cpu\|cuda\|vulkan>` | **Implemented** (Phase 9) |

### Memory / I/O pipeline

| Item | Status |
|------|------|
| Layer pack + ping-pong DMA | Existing |
| LoRA side-path (attach at compute) | **Implemented** (DMA buffers non-destructive) |
| Expert-unit index / dynamic DMA | **Implemented** (`experts.pack` + `PrefetchRing`) |
| project-map node `io_uring` prefetch | **Implemented** (`map.bin` + `PrefetchRing`; buffered fallback) |
| Vulkan QMatMul offload (Candle stack) | **Implemented** (Phase 9; ash + SPIR-V; CPU fallback; often slower than CPU decode) |
| Vulkan Q4_K-class dequant+GEMV + VRAM hot weights | **In progress** (Phase 10; Q4_K + warm_q4k; end-to-end decode → Phase 13) |
| Gemma 3 27B hybrid text run | **Done** (Phase 11; text hybrid verified) |
| Gemma 4 26B-A4B MoE hybrid text run | **Done** (Phase 12; text verified) |
| Decode throughput (MoE + Vulkan fuse / MoE I/O) | **In progress** (Phase 13; ~2× landed; ≥3× → Phase 14) |
| Accuracy-preserving decode speed (≥3×) | **Planned** (Phase 14; Softmax GPU · layer pipeline · conditional Q8_0) |
| ΔW merge at CQE (weight rewrite) | Not adopted (side-path policy) |

---

## 5. Recommended next steps

1. **Phase 14 (planned)** — accuracy-preserving ≥3× decode: Softmax/activation residency, dual-scratch layer overlap, conditional fused Q8_0; measure warm turn 2+
2. **Phase 13 wrap** — RSS / vs `gemma3:27b` write-up (13.1); keep ~2× path stable
3. **Phase 6 follow-ups** — Wire real cluster launchers / CUDA backends into `job.remote` and `$LPC_LLM_CONVERT_CMD`
4. **(Optional)** Pathfinder MoE catalog `qwen3:30b-a3b`
5. **(Optional)** In-process adapter hot-reload / mid-chat hot-swap (Phase 1 + 7.3 leftovers)
6. **(Optional)** Packaging that ships `/etc/lpc-llm/config_lpcllm` with `install.mode = "system"`
7. **(Optional)** project-map 16GB-class regression bench / inotify delta watch; Gemma vision

---

## 6. Notes (done outside the spec)

- [x] Fix Backspace on Japanese input truncating UTF-8 bytes (REPL switched to `rustyline`)
- [x] Dev PATH install via `scripts/install-dev.sh` (symlink → `target/debug`; respects `config_lpcllm` `bin_dir`)
- [x] System binary install via `scripts/install-system.sh` (shared binary only)
- [x] Privacy: private corpora under `train_dir` (home/XDG); repo `data/train/` is gitignore-only; public samples in `examples/`
- [x] Hybrid load/prefill progress on stderr (`progress` module; numbered phases + counters)
- [x] Gemma 4 MoE fixes: F16→F32 before rms_norm; expert_count metadata; Candle-reversed expert slice; rope contiguous

---

# 日本語

仕様書「MoE 対応・差分アダプタ駆動・軽量エージェント統合」に対する実装状況。  
プロジェクトテーマ: **限定的リソース下での LLM 効率化実行とモデル作成**  
最終更新: 2026-08-05

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
| 拡張 / Phase 7 | 自動知識獲得 & ユーザー適応（Web + auto-train） | **完了**（条件付き可能） |
| 拡張 / Phase 8 | NVMe 常駐 project-map & 俯瞰記憶 | **完了**（条件付き可能） |
| 拡張 / Phase 9 | 計算デバイス選択 + Candle スタック Vulkan オフロード | **完了**（第一到達） |
| 拡張 / Phase 10 | Vulkan 本格高速化（量子化シェーダ + VRAM ホット重み常駐） | **進行中**（Q4_K GEMV + warm；端到端デコード → Phase 13） |
| 拡張 / Phase 11 | Gemma 3 最大版（27B）hybrid **テキスト**実行 | **完了**（テキスト hybrid 検証済） |
| 拡張 / Phase 12 | Gemma 4 **26B-A4B（MoE）** hybrid テキスト実行 | **完了**（テキスト hybrid 検証済；レイテンシ磨き → Phase 13） |
| 拡張 / Phase 13 | デコードスループット（MoE + Vulkan 端到端） | **進行中**（~2× 到達；≥3× → Phase 14） |
| 拡張 / Phase 14 | 精度維持のままデコード高速化（≥3× / ストレッチ tok/s） | **計画**（Softmax GPU · 層パイプライン · 条件付き Q8_0） |

**いま使えるもの:**  
`lpc-llm run <model> --adapter <name>`（Hybrid LoRA）、  
`lpc-llm run <model> --agent`（SmolLM2 ルーター → アダプタ/Expert ヒント自動選択、RAM 排他）、  
`lpc-llm adapter create --from … --out … --base …`（LoRA SFT → `adapters/<name>/`）、  
`lpc-llm train scratch|sft|dpo|export`（超小型 from-scratch → GGUF → `run`）、  
`lpc-llm job init|run|import|convert`（宣言的ステージ / リモートブリッジ / RLHF スタブ）、  
`lpc-llm config show|init|get`（`config_lpcllm`: bin_dir + ユーザごとの data/train）、  
`lpc-llm search` / `knowledge` / `adapter auto-train`（Phase 7）、  
`lpc-llm project-map` / `run --project-map` / `--knowledge` / `--no-user-profile`（Phase 7–8）、  
MoE GGUF では `experts.pack` + Top-K Expert DMA + **Expert RAM LRU キャッシュ**（hybrid）。  
`lpc-llm setup` / `run --device`（Phase 9；VRAM キャッシュ済み Q4_K を Vulkan）。  
`lpc-llm run gemma3:27b --hybrid`（Phase 11 dense テキスト検証済）。  
**Phase 12 検証済:** `gemma4:26b-a4b --hybrid --ram-mib 16384 --device vulkan` でテキスト対話可。**`cargo build --release` 推奨**。1 通目 TTFT は重い。**デコード ~1 文字/秒**が観測され、スループット改善は **Phase 13**。  
**Phase 9（到達）:** setup → ホーム `config_lpcllm` の `[ui]`/`[runtime]`。  
**Phase 10（並行；スループットは 13 が担当）:** Q4_K GEMV + VRAM warm は存在。呼び出し単位の fence が端到端デコードを支配しやすい。  
**Phase 11（完了）:** Gemma 3 **27B** dense Instruct を `--hybrid` でテキスト実行。  
**Phase 12（完了）:** Gemma 4 **26B-A4B MoE**（総量 ~25.2B / 活性 ~3.8B）；常駐目標 ≤16 GiB。  
**Phase 13（進行中）:** DeviceAct + Top-K gate/up 融合 + shared∥expert 並列 + attn QKV DeviceAct + Gemma4 thought 抑制（`>>>` 回答）。ホスト **~1.8–2.4 tok/s**（≈2–2.4×）。ストレッチ **≥3×** と会話速度帯 → **Phase 14**（同一量子化・Top-K 維持）。
**Phase 14（計画）:** 精度維持の高速化 — Softmax 近傍の fence 削減、層をまたぐ GPU∥CPU 重ね、条件付き融合 Q8_0 down；成功条件は暖機後 2 通目で ≥3×。

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

### Phase 7: 自動知識獲得 & ユーザー適応 — **完了**（条件付き可能）

Web 知識の非同期獲得と、ユーザー傾向の差分 LoRA 自動更新。  
**依存:** 7.2 / 7.3 の学習本体は **Phase 4（`adapter create`）完了が前提**。7.1 と自動アタッチは Phase 1 だけで着手可。

#### 7.1 Web 検索・ナレッジインジェクション（`search` 連携）

- [x] 検索バックエンド抽象（DuckDuckGo / Custom HTTP；`LPC_LLM_SEARCH_*`、`curl` 転送）
- [x] 対話中の「知識不足」ヒューリスティック（未知エンティティ・明示的検索指示・低信頼キュー）
- [x] バックグラウンド検索ジョブ（スレッド取得 → パース → 永続化）
- [x] `cache/knowledge/` ストア（チャンク本文・出典 URL・取得時刻・タグ）
- [x] 推論時のナレッジ注入（`--knowledge`；RAG 的合成；文字数予算）
- [x] CLI: `lpc-llm search <query>` / `lpc-llm knowledge list|purge`

#### 7.2 ユーザー癖・文脈の自動アダプタ化（`adapter auto-train`）

- [x] 会話・修正・プロンプト傾向のローカルログ（`cache/user_logs/`；秘匿・ローテーション）
- [x] コーディングスタイル特徴の抽出（インデント・命名・コメント密度など軽量特徴）
- [x] Linux アイドル検知（xprintidle / GNOME IdleMonitor / 壁時計フォールバック）
- [x] アイドル時に Phase 4 学習器を呼び、差分 LoRA を `adapters/user_profile/` へ更新
- [x] 学習ジョブのガード（時間上限・RAM 上限・最小サンプル数・失敗時ロールバック）
- [x] CLI: `lpc-llm adapter auto-train [--once|--daemon]`

#### 7.3 シームレスな自動アタッチ

- [x] `run` 開始時に `adapters/user_profile/` が有効なら Hybrid サイドパスへ自動組込
- [x] `--no-user-profile` / `--adapter` 明示指定との優先順位（明示 > agent > user_profile）
- [ ] （任意）プロセス内ホットリロード（学習完了後の次回ターンから新重み）
- [ ] （任意改善）会話途中ホットスワップは Phase 1 任意改善と統合

### Phase 8: NVMe 常駐 project-map & 俯瞰記憶 — **完了**（条件付き可能）

全コードを 16GB RAM に載せない前提で、NVMe 上の構造化グラフから必要ノードだけを `io_uring` で引く。  
**依存:** 既存層パックの `io_uring` / `O_DIRECT` パイプライン。Phase 2 とは独立に着手可（バッファリング戦略は共有しうる）。

#### 8.1 NVMe へのプロジェクトグラフマッピング

- [x] 言語フロントエンド（純 Rust ヒューリスティック；Rust/Python/JS/TS/Go/C 系。tree-sitter/C なし）
- [x] 関数/クラスの呼び出し・型依存の辺をグラフ化
- [x] ノード軽量 Embedding（ハッシュ n-gram；フル LLM 埋め込みは任意）
- [x] オンディスク形式 `cache/projects/<hash>/map.bin` + オフセット/索引メタ（`map.json`）
- [x] ファイル mtime フィンガープリント；`rebuild` でクリーン再構築（クロファイル辺のため再走査）
- [x] CLI: `lpc-llm project-map build|status|rebuild <path>`

#### 8.2 `io_uring` 経由のオンデマンド・シンボル引き出し

- [x] ノードの固定長/チャンク境界レコード（`O_DIRECT` 整列；バッファドフォールバック）
- [x] クエリ → 関連ノード集合（BM25 風 / Embedding 近傍 / グラフ近傍）
- [x] 選定ノードの `io_uring` プレフェッチ → RAM リング（`PrefetchRing`；バッファド可）
- [x] コンテキスト組立時のトークン/文字数予算キャップ

#### 8.3 `--project-map` 広域コンテキスト俯瞰

- [x] `lpc-llm run … --project-map [<path|hash>]` CLI
- [x] 呼び出し関係・型依存を **部分グラフ要約**としてプロンプトへ合成（全量貼付はしない）
- [x] リファクタ/生成向けの構造的ヒント（依存先シグネチャ列・影響範囲）
- [ ] 16GB 級 RAM でも数十万行規模を「構造として」扱えることの回帰ベンチ（任意）

### Phase 9: 計算デバイス選択 + Candle スタック Vulkan — **完了**（第一到達）

CPU / CUDA / Vulkan / auto を初期設定の i18n 一問一答で選び、ホームの `config_lpcllm` に保存。  
Candle 推論スタック上で量子化 MatMul を Vulkan（ash + SPIR-V）へオフロード（Candle `Device` の fork はしない）。  
**依存:** Phase 1 hybrid `QMatMul`。CUDA は feature ゲート（`--features cuda`）。

#### 9.1 初期設定（i18n）+ 設定ファイル

- [x] `lpc-llm setup` / `config init --interactive` — 一問一答（言語 → 計算デバイス）
- [x] `[ui] language`（`ja`/`en`）と `[runtime] device`（`auto`/`cpu`/`cuda`/`vulkan`）を `~/.config/lpc-llm/config_lpcllm` に保存
- [x] 環境変数 `LPC_LLM_LANGUAGE` / `LPC_LLM_DEVICE`；`run --device` で一時上書き
- [x] ゲート: ユーザ設定が無い、または `runtime.device` 未設定なら setup を促す（スキップ可→その場は CPU）

#### 9.2 デバイス解決 + Vulkan バックエンド

- [x] `ComputeBackendKind` 解決: `auto` → Vulkan 可なら Vulkan、次に CUDA（feature）、否则 CPU；失敗時は CPU へフォールバック
- [x] ash Vulkan: instance/device/queue + バッファプール + f32 GEMM SPIR-V（WGSL→SPIR-V は naga build.rs）
- [x] QMatMul ホットパス: Candle `QTensor` で dequant + Vulkan GEMM；未対応は CPU `QMatMul::forward`
- [x] Hybrid / Eager の load と ready 表示（`Vulkan+pack+io_uring` / `CPU+…`）

### Phase 10: Vulkan 本格高速化（量子化シェーダ + VRAM ホット重み常駐） — **進行中**

Phase 9 で Vulkan 経路と GPU 使用率は確認できるが、デコード体感は Candle CPU より速くない（遅い）ことが多い。現行は毎回 CPU でフル dequant → f32 重み転送 → 素朴 f32 GEMM → 結果を戻すため。  
**目標:** `--device vulkan` を hybrid デコードで `--device cpu` と同等→上回る。  
**依存:** Phase 9 の ash / SPIR-V スタック + Hybrid ホット層ピン。

#### 10.1 GPU 側量子化 MatMul（毎回フル dequant / 全重み転送をやめる）

- [x] 主要 GGUF 型向け **dequant + GEMV/GEMM** の SPIR-V / WGSL シェーダ（まず **Q4_K**；必要なら Q5_K / Q8_0）
- [x] 重みは **デバイス上で量子化のまま**保持；フォワード毎に行列全体を CPU dequant しない（Q4_K 経路）
- [x] 毎回の f32 全重みアップロードを避ける（活性化＋小さな staging のみ、または常駐バッファ）
- [x] 未対応 dtype は Candle CPU `QMatMul::forward` へソフトフォールバック（`vulkan-skip:`）

#### 10.2 ホット層量子化重みの VRAM 常駐

- [x] Q4_K ブロブを `Arc<QTensor>` キーで VRAM キャッシュ（ホット層 + MoE expert materialize 後）
- [x] ホット層ピン後および Expert materialize 後に明示的 `warm_q4k`（小バッチデコードでも GPU ヒット可）
- [ ] ストリーム（非ホット）層は後続；まずウォームアップ後 TTFT を支配するホット常駐経路を対象
- [x] `VulkanContext` Drop 時にキャッシュ解放
- [x] MoE 向け VRAM 重みキャッシュ上限を拡大（768）

#### 10.3 CPU ベースラインと比較測定（A/B）

- [x] 10.1–10.2 完了まで **`--device cpu` の方が速い可能性**があることを文書化；比較基準として使う
- [x] 単体 A/B: CPU 参照 GEMV vs Vulkan Q4_K（`q4k_gpu_matches_cpu_when_vulkan_available`）
- [x] 起動時 GEMV マイクロベンチ: GPU が遅ければ Candle CPU を優先；スクラッチバッファ再利用
- [x] hot≤8 ハード上限を撤廃（小モデルで `--ram-mib` が効くように）
- [ ] （任意）非 Q4_K テンソルが支配的なときの起動警告

### Phase 11: Gemma 3 最大版（27B）hybrid テキスト実行 — **完了**（テキスト hybrid 検証済）

テーマ実証: **dense** な数十億パラメータを、全重みを RAM に載せず RAM + NVMe 層パック hybrid で動かす。対象は Gemma 3 Instruct **27B**（Q4 系 GGUF）。  
**成功条件:** `lpc-llm run gemma3:27b --hybrid` で家庭用〜ワークステーション RAM 予算の **テキスト**対話が完走すること。  
**成功条件外:** 画像入力 / vision encoder（任意残り、または後続フェーズ）。  
**依存:** 既存 hybrid pack + io_uring；Gemma2 経路をベース；速度は Phase 10 と並行可。

#### 11.1 カタログ + プロンプト

- [x] カタログ: `gemma3:4b` / `gemma3:12b` / `gemma3:27b`（HF GGUF + tokenizer）
- [x] `PromptStyle::Gemma`（`<start_of_turn>` / `<end_of_turn>`）が Gemma 3 IT で使えることを確認
- [x] 27B Q4 の概算サイズ / `--ram-mib` ヒントを文書化（重み ~15–20 GB；層ストリーム）

#### 11.2 GGUF / アーキ（Gemma 2 との差分）

- [x] `gemma3` メタデータ: sliding window、**sliding_window_pattern**（局所5 : 大域1）、二重 RoPE（local / global）
- [x] **attn_q_norm** / **attn_k_norm** の materialize；post-attn / post-ffw norm は既存を維持
- [x] 層ごとの attention: 局所 SWA マスク + KV trim vs 大域フル causal；attn softcap が無い場合は掛けない
- [x] multimodal projector があるが vision 未実装のとき、明確にソフト失敗 / 警告

#### 11.3 27B 向け Hybrid メモリ

- [x] 層パック + hot 選定が 27B Q4 と `--ram-mib` で成立（非ホットはストリーム；emb / norm / lm_head 常駐）
- [x] KV / コンテキスト予算を RoPE 表長・RAM ヒントと整合（第一到達では 128K 未満キャップ可）
- [x] `stream > 0` 時ヒント（既存）と 27B 向け推奨 `--ram-mib`

#### 11.4 検証（27B 優先）

- [x] **`gemma3:27b --hybrid` テキスト対話**（フェーズ完了条件；ここから開始）
- [ ] （任意）`gemma3:4b` / `12b` スモークは 27B の不具合切り分け時のみ
- [ ] （任意）Vision / マルチモーダル Gemma 3 — 後回し

#### 11.5 量子化 / 計算メモ

- [x] Q4_K_M を推奨；Q4_0 QAT は Candle CPU MatMul へフォールバック（Phase 10 は Q4_K 優先）
- [x] Gemma 3 `tokenizer.json`（配列 merges）読込のため `tokenizers` を 0.21 へ更新
- [ ] （任意）4B で `--device cpu` vs `vulkan` の A/B（Phase 10 後）

### Phase 12: Gemma 4 26B-A4B（MoE）hybrid テキスト実行 — **完了**（テキスト hybrid 検証済；レイテンシ → Phase 13）

Gemma 3 に **MoE 27B は無い**。同クラスの Gemma 系 MoE は **Gemma 4 26B-A4B**（総量 ~25.2B / トークンあたり活性 ~3.8B；ルーテッド Expert 128・Top-8 + 共有 Expert）。  
**目標:** テーマどおり RAM + NVMe hybrid で動かす — attention / router / shared は `layers.pack`、ルーテッド Expert は `experts.pack` + Top-K DMA。これにより **常駐 RAM ≤16 GiB** を狙いつつ、デコード計算量は dense 27B ではなく ~4B 活性に寄せる。  
**成功条件:** `lpc-llm run gemma4:26b-a4b --hybrid --ram-mib 16384` で **テキスト**対話が完走し、プロセス RSS が実質 ~16 GiB 未満（現状の dense 27B のような「ほぼ全層 hot」ではないこと）。  
**成功条件外（第一到達）:** 画像 / vision encoder；フル 256K コンテキスト；スマホ級 TTFT。  
**依存:** Phase 2 MoE pack + ring；Phase 11 の Gemma SWA / QK-norm 経験；`--ram-mib` 会計の厳格化（emb / KV / slots）。  
**注:** 同クラス代替 MoE（`qwen3:30b-a3b`、~30.5B/3.3B 活性）は Gemma 4 GGUF/アーキが遅延した場合の **経路探査**として可。Phase 12 の成功条件は置き換えない。

#### 12.1 カタログ + 成果物

- [x] カタログ `gemma4:26b-a4b`（IT、Q4_K_M GGUF + tokenizer；bartowski + unsloth）
- [x] Gemma 4 IT のプロンプト / 特殊トークン確認（`PromptStyle::Gemma4` / `<|turn>` … `<turn|>`）
- [x] ディスクサイズ vs **活性**パラメータ vs `--ram-mib` を文書化（カタログ hint + README；Expert は NVMe）

#### 12.2 GGUF / アーキ（Gemma 4 MoE）

- [x] `gemma4` / MoE メタデータ解析（expert_count、expert_used_count、共有 Expert、sliding/global、二重 head_dim）
- [x] `MoeFamily` / レイアウトを Gemma 4 向けに拡張（`FusedGateUpTrailing` + 共有 dense FFN）
- [x] Phase 11 の SWA / 二重 RoPE / QK-norm を再利用・拡張（層ごと head_dim / n_kv；大域 partial RoPE；cos/sin contiguous）
- [x] vision テンソルはソフト警告またはスキップ（テキスト GGUF に `v.*` なし；mmproj は別）
- [x] GGUF `expert_count` メタを優先（`expert_feed_forward_length` / 704 と混同しない）
- [x] Candle 次元反転に対応した trailing Expert スライス + gate_up 分割（experts.pack **v3**）

#### 12.3 Hybrid メモリ ≤16 GiB

- [x] `--ram-mib` を **実質常駐上限**として扱う: embeddings（f16）/ lm_head / KV / 層スロット×2 / MoE Expert リングを予約
- [x] コア層（attn / norms / router / shared expert）は `layers.pack`；**ルーテッド Expert は `experts.pack` Top-K のみ**
- [x] Gemma 4 で巨大 embedding の f32 全 dequant を避ける（f16 経路；rms_norm 前は F32 活性化）
- [x] 起動ログ: 推定常駐 MiB、hot 層、Expert リング、stream 数；番号付き `[1/5]…` フェーズ + カウンタ

#### 12.4 レイテンシ（MoE 活性経路）

- [x] 次 Top-K Expert の prefetch と計算の重ね合わせ（既存 `PrefetchRing`；Top-8 スロット）
- [x] materialize 済み Expert の RAM LRU（`--ram-mib` 余剰から容量決定；KV リセットを跨いで保持）
- [x] ホット層 + キャッシュ Expert の VRAM `warm_q4k`（Vulkan 小バッチ経路）
- [ ] 同一 `--ram-mib 16384` で `gemma3:27b` dense との TTFT / tok/s を計測（**release** バイナリ）→ **Phase 13.1 へ移管**
- [ ] （任意）Gemma 4 GGUF 遅延時は既存 Qwen-MoE 経路で `qwen3:30b-a3b` を経路探査 A/B → **Phase 13.5**

#### 12.5 検証

- [x] パック構築: Gemma 4 26B-A4B 向け `layers.pack` + `experts.pack`（experts=128、top-k=8、pack v3）
- [x] **`gemma4:26b-a4b --hybrid --ram-mib 16384` テキスト対話**（フェーズ完了条件；release 推奨）
- [ ] RSS / hot 選定が ~16 GiB 意図を満たすことを確認（計測方法を文書化）→ **Phase 13.1**
- [ ] （任意）Vision / マルチモーダル Gemma 4 — 後続フェーズ

**pull の独立性:** blobs は `~/.local/share/lpc-llm/blobs/<hf-repo>/`（XDG ユーザデータ）。カタログ各モデルは独立モジュールで、`lpc-llm pull` は `.part` レジュームし、別モデルの `run` 停止を必要としない。

**運用ヒント:** debug ビルドでは MoE prefill が分単位になりやすい → `./target/release/lpc-llm` を使う。stderr に prefill 層進捗と `expert cache hits/misses` を表示。

### Phase 13: デコードスループット（MoE + Vulkan 端到端） — **進行中**

Phase 12 で **正しさ**（`gemma4:26b-a4b` テキスト対話）は到達。release + Vulkan でも **デコード ≈ 1 文字/秒**は会話用途に耐えない。  
マイクロベンチでは Q4_K GEMV が GPU≈0.18 ms vs CPU≈0.59 ms で勝っていても端到端が遅い → ボトルネックは **パイプライン / 同期 / I/O**であり、「GEMV シェーダ不足」ではない。

**実測シグナル（2026-08-04、release、`--hybrid --ram-mib 16384 --device vulkan --max-tokens 128`）:**

| シグナル | 観測 | 示唆 |
|----------|------|------|
| Prefill | 16 prompt tok × 30 層で **18.7 s** | 層あたり ~1.2 s；第一到達としては可 |
| Prefill 後 Expert キャッシュ | **1177/2048**、hits=0 / misses=1177 | 1 通目は完全コールド；容量未満 |
| デコード体感 | **~1 文字/秒** | 日本語 1 文字≈1 tok なら ~1 tok/s 級；~3.8B 活性の潜在に遠く及ばない |
| `mlock` | 失敗（os error 12 / RLIMIT_MEMLOCK） | アンロック arena → 負荷時のページフォルトリスク |
| GEMV probe | VRAM キャッシュ時は GPU 勝ち | **カーネル自体は健全**；呼び出し単位の submit+fence が支配的と仮説 |

**原因仮説（期待レバレッジ順）:**

1. **Vulkan の QMatMul 単位同期** — GEMV ごとに activation の host↔device + `queue_submit` + `wait_for_fences`。デコード（m=1）は **トークンあたり数百回**の QMatMul（attn Q/K/V/O + shared FFN + Top-8×{gate,up,down} × 30 層）。fence コストが 0.18 ms カーネルを飲み込む。
2. **CPU↔GPU ピンポン** — 呼び出しごとに activation が GPU を離れ（`to_vec1` / `Tensor` 再構築）、attention / norm / Softmax は Candle CPU。
3. **MoE Expert I/O（ミス時）** — Top-8 × 30 = 最大 240 expert/トークン。コールドミス = NVMe DMA + materialize + `warm_q4k`（同期）。prefill で 1177 を暖機済みだが、デコード中の hit 率は未計測（フェーズ別カウンタが必要）。
4. **アンロック arena** — `mlock` 失敗が大 RSS 下のジッタを増幅。
5. **残存 non-Q4_K / 未キャッシュ重み** — 静かな `vulkan-skip:` → Candle CPU（Phase 10 残り）。

**目標:** 同一 ≤16 GiB 常駐テーマのまま、`gemma4:26b-a4b` の **2 通目以降デコードを会話速度**にする。  
**成功条件（release、1 回暖機後）:** TTFT / tok/s を計測・公開し、同一ホストの Phase 13 コールド基準比で **≥2×**（**到達 ~1.8–2.4×**）。ストレッチ **≥3× / ≥8–15 tok/s** → **Phase 14**。  
**成功条件外:** スマホ級 TTFT；フル 256K；vision；llama.cpp 絶対 tok/s 追従。  
**依存:** Phase 10 Q4_K；Phase 12 MoE hybrid。Phase 12.4 計測と Phase 10 磨きを本フェーズに吸収。

#### 13.0 運用ベースライン（コード不要・最先）

- [x] memlock: soft→hard のみ静かに上げる。**既定は `ulimit` 不要**（unlocked arena）
- [x] 速度判定は **2 通目**；stderr `[bench]` 行
- [ ] 同一 release で `--device cpu` vs `--device vulkan` のホスト A/B
- [ ] 1 通目後の RSS / expert キャッシュ充填を記録

#### 13.1 計測ハーネス（Phase 12.4 磨きを吸収）

- [x] stderr `[bench]`: **TTFT**、**decode tok/s**、prefill / decode 分割
- [x] Expert キャッシュの **デコード区間のみ** hits/misses
- [x] Vulkan submits / skip カウンタ；`vulkan-skip` 理由は初回のみログ
- [ ] `--ram-mib 16384` 下で `gemma3:27b` との数値を文書化（ホスト実行）
- [ ] RSS / hot 選定が ~16 GiB 意図を満たすことを確認

#### 13.2 Vulkan デコード経路 — 同期削減（最大レバレッジ）

- [x] **コマンドバッファ融合**: Q4_K/Q6_K GEMV を 1 submit（最大 16 ops）
- [x] **DeviceAct** + 融合グループで activation を一括 H2D；HOST barrier は最終 op のみ；multi-X downs API
- [x] staging / 複数 descriptor set 再利用
- [x] VRAM warm 済み重みは GPU（ホット pin + MoE Expert）；コールドは Candle CPU
- [x] **Q6_K Vulkan シェーダ**（candle `BlockQ6K::to_float` と一致）
- [x] **Expert GPU 経路**: Top-K gate/up を VRAM warm；gate+up を 1 submit（≤16）；**Q8_0 down → 並列 CPU**（k≈704 では sync GPU が不利）

#### 13.3 MoE デコード I/O

- [x] キャッシュヒット時 DMA スキップ；非同期 prefetch；ring スロット占有管理
- [x] 暖機後 hit 率 **~98%**（2026-08-05: hits=2359/misses=41）
- [x] README: 任意の `mlock`（`ulimit` 不要）；soft RLIMIT_MEMLOCK 引き上げ
- [ ] ring / LRU と `--ram-mib` 余剰の調整
- [x] 直前トークン Top-K からの投機 prefetch

#### 13.4 残計算（Phase 10 残り）

- [x] `vulkan-skip` 理由の初回ログ + カウンタ
- [x] Q8_0 expert-down: シェーダあり（`q8_0_gemv.wgsl`）だが **forward は Candle CPU 優先**（Gemma4 down では sync GPU が遅い）
- [ ] （任意）非ホット層の GPU 経路
- [x] attention Softmax / RoPE は当面 CPU

#### 13.5 検証

- [x] 計測方法を README に記載
- [x] 回帰スモーク: 短プロンプト + decode；2 通目 expert hits（2026-08-05: ~1.8 tok/s ≈2×；submits≈½）
- [ ] （任意）`qwen3:30b-a3b` A/B

**ホスト実測（2026-08-05、release、`--hybrid --ram-mib 16384 --device vulkan --max-tokens 16`）:**

| 通 | decode tok/s | expert hits/misses | 備考 |
|----|-------------|-------------------|------|
| Phase-13 ベースライン (2026-08-04) | **~1** | cold | ~1 文字/秒 |
| 融合後 1 通目 | **1.87** | 1350/90 | TTFT≈11s |
| 融合後 2 通目 | **1.81** | 2359/41 | ≈2×；vulkan submits≈½ |

ストレッチ **≥3×** は **Phase 14 へ移管**（Softmax GPU / 層パイプライン / 条件付き Q8_0）。

**pull 独立性 / 運用:** Phase 12 と同じ。計測は常に `./target/release/lpc-llm`。

### Phase 14: 精度維持のままデコード高速化（≥3×） — **計画**

Phase 13 で量子化・Top-K を変えずに暖機後 **~2×** を得た。残りレイテンシは依然 **パイプライン / 同期**であり、「重みが間違っている」ではない。Phase 14 は **ビット等価な MatMul**（同一 Q4_K / Q6_K / Q8_0、同一 Top-8）を保ち、ホスト待ちと CPU/GPU アイドルだけを削る。

**制約（厳守）:** より粗い量子化禁止、Top-K 削減禁止、回答が変わりうる speculative accept 禁止。数値経路は Phase 13 の CPU/GPU 参照と一致（既存単体 + スポット A/B logits）。

**ベースライン（ホスト、release、暖機 2 通目以降）:** Phase 13 持続 **~1.8 tok/s**；短答バースト ~2.4。Phase 13 文書化コールド基準 **~1 tok/s**。

**目標:** 同一 ≤16 GiB hybrid テーマのまま会話に耐えるデコード。  
**成功条件:** 暖機 2 通目以降で Phase 13 コールド基準比 **≥3×**（~1 → ≥3 tok/s）；`[bench]` と submits/token を公開。  
**ストレッチ:** 融合 attention+FFN グラフ + dual-scratch が乗れば **≥8 tok/s**（ホスト依存）。  
**成功条件外:** llama.cpp 絶対 tok/s 追従；粗い量子化での「勝ち」；vision；フル 256K。

**依存:** Phase 13 の DeviceAct / multi-X / gate/up 融合；Q8_0 シェーダはツリー内（forward は CPU 優先のまま）。

#### 14.0 計測ゲート（コード不要）

- [ ] Phase 14 マージ前に暖機 2 通目の `[bench]` を記録（tok/s、submits、expert hits、skips）
- [ ] 比較コマンドを固定:  
  `./target/release/lpc-llm run gemma4:26b-a4b --hybrid --ram-mib 16384 --device vulkan --max-tokens 48`
- [ ] （任意）同一プロンプトで logits A/B（max abs / KL）CPU 参照 vs 新経路

#### 14.1 Attention 側 fence 削減（最大レバレッジ）

- [ ] QKV 融合は維持（13 で済）；**O / Softmax 境界**の往復を減らす
- [ ] decode `m=1` で Softmax（必要なら QKᵀ / AV）を GPU 化 — **Candle と同じ f32 演算**
- [ ] DeviceAct（または dual Y scratch）で Softmax 入力を residual / 次 norm までホストに戻さない
- [ ] RoPE は検証済み GPU 実装が来るまで CPU；rope 表は変更しない

#### 14.2 層をまたぐ GPU ∥ CPU 重ね

- [ ] Dual fence + dual scratch（ping-pong）: バッファが異なれば次 submit が先行 D2H を待たない
- [ ] デコードパイプライン: **層 i の expert/shared Q8_0 CPU down** と **層 i+1 の QKV GPU** を重ねる（i+1 の residual が揃ってから）— または GPU アイドルを有用な CPU 作業で埋める
- [ ] 数値を変える層演算の並べ替えは禁止（独立演算の並列のみ）

#### 14.3 条件付き融合 Q8_0 down

- [ ] `q8_0_gemv.wgsl` を維持；**融合 multi-X マイクロベンチが並列 Candle CPU に勝つときだけ** forward を有効化（Gemma4 down `n≈2816`, `k≈704`, Top-8）
- [ ] 方針: 単独の小さな Q8_0 GEMV は CPU；**≥4 downs / 1 submit** なら GPU 可
- [ ] candle `BlockQ8_0::to_float` とのビット一致（Q6_K と同様の単体）

#### 14.4 MoE / I/O 磨き（精度非依存）

- [ ] 暖機後 short chat で expert RAM hit **≥95%**；gate/up の VRAM thrash を避ける
- [ ] キャッシュ済みの冗長 `warm_*` を避ける（expert は済；shared / attn を監査）
- [ ] RSS / `--ram-mib 16384` の計測方法を文書化（13.1 残りを吸収）

#### 14.5 範囲外（品質またはテーマを損なう）

- より粗い GGUF（Q3_K 等）、`expert_used_count` 削減、回答が分岐しうる draft/speculative
- Candle Vulkan `Device` 全面書き換え（あればよいが ≥3× の必須ではない）
- llama.cpp パリティ追従

#### 14.6 検証

- [ ] 暖機 2 通目 tok/s が ~1 基準比 ≥3×；submits/token 減少を記録
- [ ] 固定プロンプトで logits スポット A/B（ユーザー可視の回答ドリフトなし）
- [ ] 回帰: Gemma4 `>>>` UI + 裸の `thought` リークなし
- [ ] 本ファイル + README bench にホスト数値を追記

---

## 4. 仕様書セクション別の対応状況

### データレイアウト

| パス | 仕様 | 現状 |
|------|------|------|
| `blobs/` | ベース GGUF | 既存どおり |
| `adapters/` | 差分モジュール | **実装済**（ディレクトリ + json/bin）；**バックアップ対象** |
| `adapters/user_profile/` | 自動学習ユーザー LoRA | **実装済**（Phase 7 `adapter auto-train` + 自動アタッチ） |
| `paths.train_dir`（ホーム） | `adapter create` / `train` 用非公開コーパス | **実装済**（既定 `<data_dir>/train`；`config_lpcllm`） |
| `data/train/`（リポ内） | 開発用の残り場所のみ | **gitignore 保険** — 非公開データを置かない |
| `cache/packs/.../layers.pack` | ベース層パック | 既存（名称は `layers.pack`、仕様の `base_layers.pack` 改名は未実施） |
| `cache/packs/.../experts.pack` | MoE Expert パック | **実装済** |
| `cache/knowledge/` | Web 取得ナレッジ | **実装済**（Phase 7） |
| `cache/user_logs/` | 癖学習用ログ | **実装済**（Phase 7） |
| `cache/projects/<hash>/map.bin` | プロジェクト構造グラフ | **実装済**（Phase 8） |
| `manifest.json` | models + adapters | **adapters キー追加済** |

### CLI

| コマンド | 現状 |
|----------|------|
| `run … --adapter <name>` | **実装済** |
| `run … --agent` | **実装済**（`--agent-model` 付き） |
| `run … --project-map` | **実装済**（Phase 8；path または hash） |
| `run … --knowledge` / `--no-user-profile` | **実装済**（Phase 7） |
| `adapter list` | **実装済** |
| `adapter install-demo` | **実装済**（検証用） |
| `config show\|init\|get\|example` | **実装済**（`config_lpcllm` のパス / 導入レイアウト） |
| `adapter create …` | **実装済**（LoRA SFT → Phase 1 形式；`--from` は `train_dir` 解決） |
| `train scratch|sft|dpo|export` | **実装済**（Phase 5 超小型学習 → GGUF → `run`） |
| `job init|run|status|import|convert` | **実装済**（Phase 6 ブリッジ） |
| `adapter auto-train` | **実装済**（Phase 7） |
| `search` / `knowledge …` | **実装済**（Phase 7） |
| `project-map build|status|rebuild` | **実装済**（Phase 8） |
| `setup` / `config init --interactive` | **実装済**（Phase 9 i18n デバイスウィザード） |
| `run … --device <auto\|cpu\|cuda\|vulkan>` | **実装済**（Phase 9） |

### メモリ・I/O パイプライン

| 項目 | 現状 |
|------|------|
| 層単位 pack + ping-pong DMA | 既存 |
| LoRA サイドパス（計算時アタッチ） | **実装済**（DMA バッファは非破壊） |
| Expert 単位インデックス / 動的 DMA | **実装済**（`experts.pack` + `PrefetchRing`） |
| project-map ノード単位 `io_uring` プレフェッチ | **実装済**（`map.bin` + `PrefetchRing`；バッファド可） |
| Vulkan QMatMul オフロード（Candle スタック） | **実装済**（Phase 9；ash + SPIR-V；CPU フォールバック；デコードは CPU より遅いことが多い） |
| Vulkan Q4_K 系 dequant+GEMV + VRAM ホット重み | **進行中**（Phase 10；Q4_K + warm_q4k；端到端デコード → Phase 13） |
| Gemma 3 27B hybrid テキスト実行 | **完了**（Phase 11；テキスト hybrid 検証済） |
| Gemma 4 26B-A4B MoE hybrid テキスト実行 | **完了**（Phase 12；テキスト検証済） |
| デコードスループット（MoE + Vulkan fuse / MoE I/O） | **進行中**（Phase 13；~2× 到達；≥3× → Phase 14） |
| 精度維持のデコード高速化（≥3×） | **計画**（Phase 14；Softmax GPU · 層パイプライン · 条件付き Q8_0） |
| CQE 時の ΔW マージ（重み書き換え） | 採用せず（サイドパス方針） |

---

## 5. 推奨する次工程

1. **Phase 14（計画）** — 精度維持の ≥3× デコード: Softmax / activation 常駐、dual-scratch 層重ね、条件付き融合 Q8_0；暖機 2 通目で計測
2. **Phase 13 締め** — RSS / vs `gemma3:27b` 文書化（13.1）；~2× 経路を安定維持
3. **Phase 6 フォロー** — `job.remote` / `$LPC_LLM_CONVERT_CMD` に実クラスタ・CUDA 変換を接続
4. **（任意）** 経路探査カタログ `qwen3:30b-a3b`
5. **（任意）** プロセス内アダプタホットリロード / 会話途中ホットスワップ（Phase 1 + 7.3 残り）
6. **（任意）** `install.mode = "system"` の `/etc/lpc-llm/config_lpcllm` を同梱するパッケージ化
7. **（任意）** project-map 16GB 級回帰ベンチ / inotify 差分監視；Gemma vision

---

## 6. 補足（仕様外だが実施済み）

- [x] 日本語入力時の Backspace が UTF-8 バイト欠けする問題への対処（REPL を `rustyline` 化）
- [x] 開発用 PATH 導入 `scripts/install-dev.sh`（symlink → `target/debug`；`config_lpcllm` の `bin_dir` 対応）
- [x] 共有バイナリ導入 `scripts/install-system.sh`（バイナリのみ）
- [x] プライバシー: 非公開コーパスは `train_dir`（ホーム/XDG）；リポ `data/train/` は gitignore 保険；公開サンプルは `examples/`
- [x] hybrid 起動 / prefill の stderr 進捗（`progress` モジュール；番号付きフェーズ + カウンタ）
- [x] Gemma 4 MoE 修正: rms_norm 前 F16→F32；expert_count メタ優先；Candle 次元反転スライス；rope contiguous