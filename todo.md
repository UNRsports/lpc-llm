# lpc-llm 拡張ロードマップ（進捗）

仕様書「MoE 対応・差分アダプタ駆動・軽量エージェント統合」に対する実装状況。  
最終更新: 2026-07-30

## 総括

| 軸 | 内容 | 進捗 |
|----|------|------|
| 基盤（既存） | GGUF 層パック + io_uring ダブルバッファ hybrid | **完了**（本拡張の前提） |
| 軸2 / Phase 1 | 差分アダプタ管理・サイドパス LoRA・`--adapter` | **完了** |
| 軸1 / Phase 2 | MoE Expert 分割パック + 動的 DMA | **未着手** |
| 軸3 / Phase 3 | 超軽量ルーターエージェント + メモリ排他 | **未着手** |
| 軸2 / Phase 4 | `adapter create` 学習器プロトタイプ | **未着手**（CLI 案内のみ） |

**いま使えるもの:** `lpc-llm run <model> --adapter <name>`（Hybrid 経路で LoRA サイドパス）。  
**まだ使えないもの:** `--agent`、MoE Expert ストリーミング、実アダプタ学習（`adapter create`）。

---

## 工程チェックリスト

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

### Phase 2: MoE パック + Expert ストリーミング — **未着手**

- [ ] GGUF MoE テンソル解析（`ffn_gate_exps`, `ffn_down_exps` 等）
- [ ] 常駐（embeddings / norm / lm_head / router）とオンデマンド Expert の分離
- [ ] `cache/packs/.../experts.pack`（または同等）への再レイアウト
- [ ] `layers.pack.json` に Expert index / offset テーブル拡張
- [ ] Gating Network（ルーター）推論 + Top-K Expert 選抜
- [ ] 選抜 Expert の io_uring DMA 発行
- [ ] 2× バッファを Expert 単位の動的リングへ拡張
- [ ] DeepSeek / Mixtral / Qwen-MoE 等のアーキ分岐

### Phase 3: 超軽量ルーターエージェント — **未着手**

- [ ] `lpc-llm run … --agent` CLI
- [ ] SmolLM2 360M（または分級器）による意図分類プロンプト
- [ ] 判定結果 → `--adapter` / Expert prefetch の自動選択
- [ ] ルーター完了後にメインへコンテキスト引き継ぎ（タイムシェア）
- [ ] `--ram-mib` 内でルーター用 KV とメイン用 KV の排他管理

### Phase 4: アダプタ作成器 — **未着手**

- [ ] `lpc-llm adapter create --from … --out … --base …` の実装  
      （現状は Phase 4 案内メッセージのみ）
- [ ] 小規模テキストから数 MB 差分を数分で学習・保存する処理線
- [ ] 出力を Phase 1 形式（`adapter.json` + `weights.bin`）に合わせる
- [ ] （任意）独立クレート化 / Safetensors 出力

---

## 仕様書セクション別の対応状況

### データレイアウト

| パス | 仕様 | 現状 |
|------|------|------|
| `blobs/` | ベース GGUF | 既存どおり |
| `adapters/` | 差分モジュール | **実装済**（ディレクトリ + json/bin） |
| `cache/packs/.../layers.pack` | ベース層パック | 既存（名称は `layers.pack`、仕様の `base_layers.pack` 改名は未実施） |
| `cache/packs/.../experts.pack` | MoE Expert パック | **未実装** |
| `manifest.json` | models + adapters | **adapters キー追加済** |

### CLI

| コマンド | 現状 |
|----------|------|
| `run … --adapter <name>` | **実装済** |
| `run … --agent` | **未実装** |
| `adapter list` | **実装済** |
| `adapter install-demo` | **実装済**（検証用） |
| `adapter create …` | **スタブ**（未実装案内） |

### メモリ・I/O パイプライン

| 項目 | 現状 |
|------|------|
| 層単位 pack + ping-pong DMA | 既存 |
| LoRA サイドパス（計算時アタッチ） | **実装済**（DMA バッファは非破壊） |
| Expert 単位インデックス / 動的 DMA | **未実装** |
| CQE 時の ΔW マージ（重み書き換え） | 採用せず（サイドパス方針） |

---

## 推奨する次工程

1. **Phase 2** — MoE テンソルマップと `experts.pack` の設計・実装（メモリ爆発回避の本丸）
2. **Phase 3** — `--agent` と SmolLM2 タイムシェア（アダプタ自動選択は Phase 1 API を再利用）
3. **Phase 4** — `adapter create` で実ドメイン差分を生成できるようにする

---

## 補足（仕様外だが実施済み）

- [x] 日本語入力時の Backspace が UTF-8 バイト欠けする問題への対処（REPL を `rustyline` 化）
