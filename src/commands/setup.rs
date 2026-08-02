//! Interactive first-run setup (i18n Q&A → home `config_lpcllm`).

use console::style;
use dialoguer::{Confirm, Select};

use crate::config::{
    AppConfig, ComputeDevicePref, ConfigFile, RuntimeSection, UiLanguage, UiSection,
};
use crate::device::{detect_cuda, detect_vulkan};
use crate::error::{AppError, Result};
use crate::i18n::Locale;

/// Run the one-question-at-a-time setup wizard and persist to the user config.
pub fn run() -> Result<()> {
    let detected_lang = AppConfig::load()
        .map(|c| c.ui_language)
        .unwrap_or(UiLanguage::En);

    let lang_items = ["English", "日本語 (Japanese)"];
    let lang_default = match detected_lang {
        UiLanguage::En => 0,
        UiLanguage::Ja => 1,
    };
    // Language question uses bilingual prompt (locale not chosen yet).
    let lang_idx = Select::new()
        .with_prompt("UI language / UI 言語")
        .items(&lang_items)
        .default(lang_default)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;
    let language = if lang_idx == 1 {
        UiLanguage::Ja
    } else {
        UiLanguage::En
    };
    let loc = Locale::new(language);

    println!();
    println!("{}", style(loc.t("setup.title")).bold().cyan());

    let vk = detect_vulkan();
    let cuda = detect_cuda();
    println!(
        "  {} / {}",
        if vk {
            loc.t("setup.detect_vulkan_yes")
        } else {
            loc.t("setup.detect_vulkan_no")
        },
        if cuda {
            loc.t("setup.detect_cuda_yes")
        } else {
            loc.t("setup.detect_cuda_no")
        }
    );

    let device_labels = [
        loc.t("setup.device_auto"),
        loc.t("setup.device_cpu"),
        loc.t("setup.device_cuda"),
        loc.t("setup.device_vulkan"),
    ];
    let device_idx = Select::new()
        .with_prompt(loc.t("setup.device_prompt"))
        .items(&device_labels)
        .default(0)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;
    let device = match device_idx {
        1 => ComputeDevicePref::Cpu,
        2 => ComputeDevicePref::Cuda,
        3 => ComputeDevicePref::Vulkan,
        _ => ComputeDevicePref::Auto,
    };

    let patch = ConfigFile {
        ui: UiSection {
            language: Some(language),
        },
        runtime: RuntimeSection {
            device: Some(device),
        },
        ..ConfigFile::default()
    };
    let path = AppConfig::save_user_merged(&patch)?;
    println!(
        "{} {} {}",
        style("ok").green().bold(),
        loc.t("setup.saved"),
        path.display()
    );
    println!("{}", style(loc.t("setup.skip_hint")).dim());
    Ok(())
}

/// If setup is needed, ask whether to run the wizard. Returns preferred device for this session
/// when the user skips (CPU).
pub fn maybe_gate() -> Result<Option<ComputeDevicePref>> {
    if !AppConfig::needs_setup()? {
        return Ok(None);
    }
    let lang = AppConfig::load()
        .map(|c| c.ui_language)
        .unwrap_or(UiLanguage::En);
    let loc = Locale::new(lang);
    let ok = Confirm::new()
        .with_prompt(loc.t("gate.prompt"))
        .default(true)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;
    if ok {
        run()?;
        Ok(None)
    } else {
        println!("{}", style(loc.t("gate.skip")).yellow());
        Ok(Some(ComputeDevicePref::Cpu))
    }
}
