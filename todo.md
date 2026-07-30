# lpc-llm 拡張ロードマップ（進捗）

仕様書「MoE 対応・差分アダプタ駆動・軽量エージェント統合」に対する実装状況。  
プロジェクトテーマ: **限定的リソース下での LLM 効率化実行とモデル作成**  
最終更新: 2026-07-30

## 総括

| 軸 | 内容 | 進捗 |
|----|------|------|
| 基盤（既存） | GGUF 層パック + io_uring ダブルバッファ hybrid | **完了**（本拡張の前提） |
| 軸2 / Phase 1 | 差分アダプタ管理・サイドパス LoRA・`--adapter` | **完了** |
| 軸1 / Phase 2 | MoE Expert 分割パック + 動的 DMA | **未着手** |
| 軸3 / Phase 3 | 超軽量ルーターエージェント + メモリ排他 | **未着手** |
| 軸2 / Phase 4 | `adapter create` 学習器プロトタイプ | **未着手**（CLI 案内のみ） |
| 長期 / Phase 5+ | 基盤フル学習・数十億 GGUF・SFT/RLHF | **未着手**（下記の実現可能性を参照） |

**いま使えるもの:** `lpc-llm run <model> --adapter <name>`（Hybrid 経路で LoRA サイドパス）。  
**まだ使えないもの:** `--agent`、MoE Expert ストリーミング、実アダプタ学習（`adapter create`）、基盤フル学習、本格 SFT/RLHF。

---

## テーマ追加要件の実現可能性

テーマ「効率化による限定リソース下での実行とモデル作成」に対し、次の 3 要件をどう扱うか。

| 要件 | 限定リソース下でそのまま？ | 判定 | 本リポでの現実的な落としどころ |
|------|---------------------------|------|--------------------------------|
| ゼロから基盤モデルをフル学習 | 数十億級を家庭用 CPU/少 RAM でフル学習は非現実（計算・データ・電力が桁違い） | **条件付き可能** | まず **超小型（数 M〜数百 M）の from-scratch 学習ループ** を純 Rust/Candle で持つ。大規模は外部計算資源へのジョブ投入 or チェックポイント取込 |
| 数十億パラメータ級の新規 GGUF を一から作る | 「一から学習して数十億 GGUF」は同上。**形式としての GGUF 出力パイプライン**は可能 | **条件付き可能** | (1) 小規模学習結果 → GGUF 書き出し (2) 既存重みの量子化・変換 → `blobs/` 登録。数十億の学習本体はクラスタ前提の別ステージ |
| 本格的な SFT / RLHF パイプライン全体 | フル RLHF（大規模報酬モデル + PPO 等）は GPU 多枚が前提。テーマとは緊張関係 | **条件付き可能** | ローカル向けに **SFT（LoRA/QLoRA）→ 嗜好最適化の軽量版（DPO/ORPO 等）** までをパイプライン化。「本格 RLHF」は段階的・外部アクセラレータ対応として残す |

**結論:** 3 要件とも「エンジニアリングとして追える」が、**現行マシンだけでフルスケール完遂**はテーマと矛盾する。todo には (A) 限定リソースで完結する中間成果物と (B) フルスケールを見据えた長期ステージの両方を載せる。

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

### Phase 5: 限定リソース向け「モデル作成」基盤 — **未着手**（テーマ直結・実行可能）

フルスケール 3 要件の **前段**。家庭用〜ワークステーション規模で完結させる。

- [ ] 超小型 Transformer の from-scratch 学習ループ（Candle、CPU/単一 GPU 想定）
- [ ] 学習チェックポイント → GGUF（または中間 Safetensors）書き出し
- [ ] 書き出した成果を `blobs/` + `manifest` に登録し `lpc-llm run` で推論
- [ ] ローカル SFT パイプライン（フル微調整または LoRA；Phase 4 と統合可）
- [ ] 軽量嗜好最適化（DPO / ORPO など）の最小実装 — 「本格 RLHF」への足場
- [ ] `--ram-mib` / 勾配チェックポイント等、学習時もメモリ上限を意識した設計

### Phase 6: 大規模化ブリッジ — **未着手**（条件付き・外部資源前提）

「数十億級」「本格 RLHF」を **このツールチェーンの延長**で扱うための橋。ローカル単機完結は求めない。

- [ ] **ゼロから基盤モデルをフル学習する**  
      - 分散/リモート学習ジョブの起動・再開・成果物取込インターフェース  
      - データセット仕様・トークナイザ・学習設定の宣言的定義  
      - 進捗・チェックポイントを `cache/` または外部ストアへ接続
- [ ] **数十億パラメータ級の新規 GGUF を一から作る**  
      - 大規模チェックポイント → 量子化 GGUF 変換パイプライン  
      - 変換結果の hybrid pack（Phase 2 連携）とカタログ登録  
      - ※学習計算そのものは Phase 6 のリモート/クラスタ側
- [ ] **本格的な SFT / RLHF パイプライン全体**  
      - SFT → 報酬モデル（または嗜好データ）→ PPO/類似アルゴリズムのステージ定義  
      - 評価・回帰テスト・成果アダプタ/マージ重みの `adapters/` or `blobs/` への出力  
      - アクセラレータ（CUDA 等）バックエンドの任意接続（Linux 本体の io_uring 推論パスとは分離）

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

1. **Phase 2** — MoE テンソルマップと `experts.pack`（限定リソースでの大型実行）
2. **Phase 3** — `--agent` と SmolLM2 タイムシェア
3. **Phase 4** — `adapter create`（差分による「モデル作成」の最短路）
4. **Phase 5** — 超小型 from-scratch + GGUF 出力 + ローカル SFT/DPO（テーマ内で完結）
5. **Phase 6** — フル基盤学習 / 数十億 GGUF / 本格 RLHF（外部計算とブリッジ）

---

## 補足（仕様外だが実施済み）

- [x] 日本語入力時の Backspace が UTF-8 バイト欠けする問題への対処（REPL を `rustyline` 化）
