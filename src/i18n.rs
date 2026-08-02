//! Minimal UI locale strings (Phase 9 setup / prompts).

use crate::config::UiLanguage;

#[derive(Debug, Clone, Copy)]
pub struct Locale {
    pub lang: UiLanguage,
}

impl Locale {
    pub fn new(lang: UiLanguage) -> Self {
        Self { lang }
    }

    pub fn t(&self, key: &str) -> &'static str {
        match self.lang {
            UiLanguage::Ja => ja(key).or_else(|| en(key)).unwrap_or("???"),
            UiLanguage::En => en(key).unwrap_or("???"),
        }
    }
}

fn en(key: &str) -> Option<&'static str> {
    Some(match key {
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
        _ => return None,
    })
}
