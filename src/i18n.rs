//! Minimal UI locale strings (Phase 9 setup / Phase 15 rm help).

use crate::config::{AppConfig, UiLanguage};
use crate::error::Result;

#[derive(Debug, Clone, Copy)]
pub struct Locale {
    pub lang: UiLanguage,
}

impl Locale {
    pub fn new(lang: UiLanguage) -> Self {
        Self { lang }
    }

    /// Load UI language from `config_lpcllm` / env (`LPC_LLM_LANGUAGE`).
    pub fn load() -> Self {
        let lang = AppConfig::load()
            .map(|c| c.ui_language)
            .unwrap_or(UiLanguage::En);
        Self::new(lang)
    }

    pub fn t(&self, key: &str) -> &'static str {
        match self.lang {
            UiLanguage::Ja => ja(key).or_else(|| en(key)).unwrap_or("???"),
            UiLanguage::En => en(key).unwrap_or("???"),
        }
    }

    /// Translate and substitute `{name}`-style placeholders.
    pub fn tf(&self, key: &str, args: &[(&str, &str)]) -> String {
        let mut s = self.t(key).to_string();
        for (k, v) in args {
            let needle = format!("{{{k}}}");
            s = s.replace(&needle, v);
        }
        s
    }
}

/// If argv is `… rm (-h|--help)`, print localized rm help and return true.
pub fn try_print_command_help(args: &[String]) -> Result<bool> {
    let is_help = args.iter().any(|a| a == "-h" || a == "--help");
    if !is_help {
        return Ok(false);
    }
    // `lpc-llm rm --help` or `lpc-llm --help rm` (rare); require positional `rm`.
    let has_rm = args.iter().skip(1).any(|a| a == "rm");
    if !has_rm {
        return Ok(false);
    }
    // Do not steal help from nested flags after another subcommand.
    let first_cmd = args.iter().skip(1).find(|a| !a.starts_with('-'));
    if first_cmd.map(|s| s.as_str()) != Some("rm") {
        return Ok(false);
    }
    print_rm_help(&Locale::load());
    Ok(true)
}

pub fn print_rm_help(loc: &Locale) {
    println!("{}", loc.t("rm.help.title"));
    println!();
    println!("{}", loc.t("rm.help.usage"));
    println!();
    println!("{}", loc.t("rm.help.args_hdr"));
    println!("{}", loc.t("rm.help.arg_name"));
    println!();
    println!("{}", loc.t("rm.help.opts_hdr"));
    println!("{}", loc.t("rm.help.opt_purge"));
    println!("{}", loc.t("rm.help.opt_cache"));
    println!("{}", loc.t("rm.help.opt_with_adapters"));
    println!("{}", loc.t("rm.help.opt_yes"));
    println!("{}", loc.t("rm.help.opt_help"));
    println!();
    println!("{}", loc.t("rm.help.notes_hdr"));
    println!("{}", loc.t("rm.help.note_default"));
    println!("{}", loc.t("rm.help.note_purge"));
    println!("{}", loc.t("rm.help.note_adapters"));
    println!("{}", loc.t("rm.help.note_shared"));
}

fn en(key: &str) -> Option<&'static str> {
    Some(match key {
        // --- setup / gate (Phase 9) ---
        "setup.title" => "lpc-llm first-run setup",
        "setup.lang_prompt" => "UI language",
        "setup.device_prompt" => "Compute device for inference",
        "setup.device_auto" => "auto — prefer Vulkan, then CUDA, else CPU",
        "setup.device_cpu" => "cpu — Candle on CPU (always available)",
        "setup.device_cuda" => "cuda — NVIDIA GPU (requires --features cuda build)",
        "setup.device_vulkan" => "vulkan — GPU / iGPU via Vulkan (Intel/AMD/NVIDIA)",
        "setup.detect_vulkan_yes" => "Vulkan: detected",
        "setup.detect_vulkan_no" => "Vulkan: not available",
        "setup.detect_cuda_yes" => "CUDA: available in this build",
        "setup.detect_cuda_no" => "CUDA: not in this build",
        "setup.saved" => "Saved settings to",
        "setup.skip_hint" => "You can re-run with `lpc-llm setup` anytime.",
        "gate.prompt" => "Compute device is not configured. Run first-run setup now?",
        "gate.skip" => "Skipping setup for this session (using CPU). Run `lpc-llm setup` later.",

        // --- menu ---
        "menu.tagline" => "— local LLM runner (pure Rust / Candle)",
        "menu.prompt" => "What do you want to do?",
        "menu.act_run" => "Run a model (chat)",
        "menu.act_list" => "List models",
        "menu.act_pull" => "Pull a model",
        "menu.act_show" => "Show model info",
        "menu.act_rm" => "Remove / purge a local model",
        "menu.act_exit" => "Exit",
        "menu.pick_pull" => "Select a model to pull",
        "menu.pick_show" => "Select a local model to show",
        "menu.rm_empty" => "No locally registered models.",
        "menu.rm_pick" => "Select a model to remove",
        "menu.rm_mode" => "Removal mode",
        "menu.rm_mode_soft" => "Registry only (blobs kept)",
        "menu.rm_mode_purge" => "Purge blobs + pack cache",
        "menu.rm_mode_cache" => "Pack cache only",

        // --- list ---
        "list.none_local" => "No local models installed.",
        "list.none_hint" => "  tip: `lpc-llm pull <name>` to download, or `lpc-llm list --all` for the catalog",
        "list.catalog_hint" => "  tip: `lpc-llm list --all` shows the full catalog (including not installed)",
        "list.custom_desc" => "trained / imported",
        "list.blobs_hdr" => "Model module (durable blobs):",
        "list.data_root" => "Data root: {dir}  (blobs + engine cache under here)",
        "list.cache_root" => "Engine cache (regenerable): {dir}",

        // --- run ---
        "run.select" => "Select a local model to run",
        "run.no_local" =>
            "No local models installed. Run `lpc-llm pull <name>` first (see `lpc-llm list --all`).",

        // --- rm runtime ---
        "rm.err_with_adapters" =>
            "`--with-adapters` requires `--purge` (adapters are only removed with a full uninstall)",
        "rm.err_both_flags" => "use either `--purge` or `--cache`, not both",
        "rm.soft_ok" => "removed `{name}` from registry (model blobs kept under {dir})",
        "rm.soft_tip" =>
            "  tip: use `lpc-llm rm <name> --purge` to free disk (blobs + pack cache)",
        "rm.cache_confirm" =>
            "Delete pack cache for `{name}` under {dir}? (blobs kept)",
        "rm.cancelled" => "cancelled",
        "rm.cache_ok" => "wiped pack cache for `{name}` ({bytes})",
        "rm.nothing" => "  (nothing to delete)",
        "rm.purge_confirm" =>
            "Permanently delete `{name}` blobs + pack cache under {dir}?",
        "rm.purge_adapters" => "\n  Also delete {n} adapter(s): {list}",
        "rm.purge_adapters_none" => "\n  (--with-adapters: no matching adapters found)",
        "rm.purge_ok" => "purged `{name}` ({bytes})",
        "rm.registry_removed" => "  registry: removed",
        "rm.blob_line" => "  blob:  {path}",
        "rm.cache_line" => "  cache: {path}",
        "rm.adapter_line" => "  adapter: {name}",
        "rm.skipped_shared" => "  skipped shared path {path}",

        // --- rm --help (i18n) ---
        "rm.help.title" => "Remove a model from the local registry (use `--purge` to free disk)",
        "rm.help.usage" => "Usage: lpc-llm rm [OPTIONS] <NAME>",
        "rm.help.args_hdr" => "Arguments:",
        "rm.help.arg_name" => "  <NAME>  Catalog or local model name (e.g. smollm2:360m)",
        "rm.help.opts_hdr" => "Options:",
        "rm.help.opt_purge" =>
            "      --purge          Delete durable blobs + pack cache (full uninstall)",
        "rm.help.opt_cache" =>
            "      --cache          Delete pack cache only (blobs and registry kept)",
        "rm.help.opt_with_adapters" =>
            "      --with-adapters  With `--purge`, also delete LoRA adapters for this base",
        "rm.help.opt_yes" =>
            "  -y, --yes            Skip confirmation for `--purge` / `--cache`",
        "rm.help.opt_help" => "  -h, --help           Print help",
        "rm.help.notes_hdr" => "Notes:",
        "rm.help.note_default" =>
            "  • Default removes the soft registry entry only; blobs stay on disk.",
        "rm.help.note_purge" =>
            "  • `--purge` deletes GGUF/tokenizer blobs and cache/packs/<model>/.",
        "rm.help.note_adapters" =>
            "  • Trained adapters are kept unless `--purge --with-adapters`.",
        "rm.help.note_shared" =>
            "  • Shared blob paths used by another model are not deleted.",

        _ => return None,
    })
}

fn ja(key: &str) -> Option<&'static str> {
    Some(match key {
        "setup.title" => "lpc-llm 初期設定",
        "setup.lang_prompt" => "UI 言語",
        "setup.device_prompt" => "推論に使う計算デバイス",
        "setup.device_auto" => "auto — Vulkan 優先、次に CUDA、なければ CPU",
        "setup.device_cpu" => "cpu — Candle CPU（常に利用可）",
        "setup.device_cuda" => "cuda — NVIDIA GPU（`--features cuda` ビルドが必要）",
        "setup.device_vulkan" => "vulkan — Vulkan 経由の GPU / 内蔵 GPU（Intel/AMD/NVIDIA）",
        "setup.detect_vulkan_yes" => "Vulkan: 検出済み",
        "setup.detect_vulkan_no" => "Vulkan: 利用不可",
        "setup.detect_cuda_yes" => "CUDA: このビルドで利用可",
        "setup.detect_cuda_no" => "CUDA: このビルドには未含",
        "setup.saved" => "設定を保存しました:",
        "setup.skip_hint" => "いつでも `lpc-llm setup` で再設定できます。",
        "gate.prompt" => "計算デバイスが未設定です。初期設定を今実行しますか？",
        "gate.skip" => "今回はスキップします（CPU 使用）。後で `lpc-llm setup` を実行してください。",

        "menu.tagline" => "— ローカル LLM ランナー（純 Rust / Candle）",
        "menu.prompt" => "何をしますか？",
        "menu.act_run" => "モデルを実行（チャット）",
        "menu.act_list" => "モデル一覧",
        "menu.act_pull" => "モデルを取得（pull）",
        "menu.act_show" => "モデル情報を表示",
        "menu.act_rm" => "ローカルモデルを削除 / purge",
        "menu.act_exit" => "終了",
        "menu.pick_pull" => "取得するモデルを選択",
        "menu.pick_show" => "表示するローカルモデルを選択",
        "menu.rm_empty" => "ローカル登録済みのモデルはありません。",
        "menu.rm_pick" => "削除するモデルを選択",
        "menu.rm_mode" => "削除モード",
        "menu.rm_mode_soft" => "レジストリのみ（blobs は残す）",
        "menu.rm_mode_purge" => "blobs + pack cache を purge",
        "menu.rm_mode_cache" => "pack cache のみ削除",

        "list.none_local" => "ローカルに導入済みのモデルはありません。",
        "list.none_hint" =>
            "  ヒント: `lpc-llm pull <name>` で取得、カタログ全体は `lpc-llm list --all`",
        "list.catalog_hint" =>
            "  ヒント: 未導入を含むカタログ全体は `lpc-llm list --all`",
        "list.custom_desc" => "学習 / インポート",
        "list.blobs_hdr" => "モデルモジュール（永続 blobs）:",
        "list.data_root" => "データルート: {dir}  （blobs + エンジンキャッシュ）",
        "list.cache_root" => "エンジンキャッシュ（再生成可）: {dir}",

        "run.select" => "実行するローカルモデルを選択",
        "run.no_local" =>
            "ローカルモデルがありません。先に `lpc-llm pull <name>` してください（カタログは `lpc-llm list --all`）。",

        "rm.err_with_adapters" =>
            "`--with-adapters` には `--purge` が必要です（アダプタは完全削除時のみ消えます）",
        "rm.err_both_flags" => "`--purge` と `--cache` は同時に指定できません",
        "rm.soft_ok" =>
            "`{name}` をレジストリから削除しました（モデル blobs は {dir} に残っています）",
        "rm.soft_tip" =>
            "  ヒント: ディスクを空けるには `lpc-llm rm <name> --purge` を使ってください（blobs + pack cache）",
        "rm.cache_confirm" =>
            "`{name}` の pack cache を削除しますか？（場所: {dir}、blobs は残します）",
        "rm.cancelled" => "キャンセルしました",
        "rm.cache_ok" => "`{name}` の pack cache を削除しました（{bytes}）",
        "rm.nothing" => "  （削除対象なし）",
        "rm.purge_confirm" =>
            "`{name}` の blobs + pack cache を完全削除しますか？（場所: {dir}）",
        "rm.purge_adapters" => "\n  あわせてアダプタ {n} 個を削除: {list}",
        "rm.purge_adapters_none" => "\n  （--with-adapters: 一致するアダプタなし）",
        "rm.purge_ok" => "`{name}` を purge しました（{bytes}）",
        "rm.registry_removed" => "  レジストリ: 削除済み",
        "rm.blob_line" => "  blob:  {path}",
        "rm.cache_line" => "  cache: {path}",
        "rm.adapter_line" => "  adapter: {name}",
        "rm.skipped_shared" => "  共有パスのためスキップ: {path}",

        "rm.help.title" =>
            "ローカルレジストリからモデルを削除（ディスク解放には `--purge`）",
        "rm.help.usage" => "使い方: lpc-llm rm [OPTIONS] <NAME>",
        "rm.help.args_hdr" => "引数:",
        "rm.help.arg_name" => "  <NAME>  カタログ名またはローカル名（例: smollm2:360m）",
        "rm.help.opts_hdr" => "オプション:",
        "rm.help.opt_purge" =>
            "      --purge          永続 blobs + pack cache を削除（完全アンインストール）",
        "rm.help.opt_cache" =>
            "      --cache          pack cache のみ削除（blobs とレジストリは残す）",
        "rm.help.opt_with_adapters" =>
            "      --with-adapters  `--purge` 時、このベースの LoRA アダプタも削除",
        "rm.help.opt_yes" =>
            "  -y, --yes            `--purge` / `--cache` の確認をスキップ",
        "rm.help.opt_help" => "  -h, --help           ヘルプを表示",
        "rm.help.notes_hdr" => "補足:",
        "rm.help.note_default" =>
            "  • 既定はソフトなレジストリ削除のみ。blobs はディスクに残ります。",
        "rm.help.note_purge" =>
            "  • `--purge` は GGUF/tokenizer の blobs と cache/packs/<model>/ を削除します。",
        "rm.help.note_adapters" =>
            "  • 学習済みアダプタは `--purge --with-adapters` を付けない限り残します。",
        "rm.help.note_shared" =>
            "  • 他モデルが参照する共有 blob パスは削除しません。",

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf_replaces_placeholders() {
        let loc = Locale::new(UiLanguage::En);
        let s = loc.tf(
            "rm.soft_ok",
            &[("name", "smollm2:360m"), ("dir", "/tmp/blobs")],
        );
        assert!(s.contains("smollm2:360m"));
        assert!(s.contains("/tmp/blobs"));
    }

    #[test]
    fn ja_rm_help_title() {
        let loc = Locale::new(UiLanguage::Ja);
        assert!(loc.t("rm.help.title").contains("purge") || loc.t("rm.help.title").contains("削除"));
    }
}
