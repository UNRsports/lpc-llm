//! `lpc-llm config` — show / init / get path settings from `config_lpcllm`.

use console::style;

use crate::config::{
    system_config_path, user_config_path, AppConfig, InstallMode, CONFIG_FILE_NAME,
};
use crate::error::{AppError, Result};

pub fn show() -> Result<()> {
    let cfg = AppConfig::load()?;
    println!("{}", style("lpc-llm config (resolved)").bold());
    println!("  data_dir:      {}", cfg.data_dir.display());
    println!("  train_dir:     {}", cfg.train_dir.display());
    println!("  bin_dir:       {}", cfg.bin_dir.display());
    println!(
        "  install.mode:  {}",
        match cfg.install_mode {
            InstallMode::User => "user",
            InstallMode::System => "system",
        }
    );
    println!("  ui.language:   {}", cfg.ui_language.as_str());
    println!(
        "  runtime.device:{} ({})",
        cfg.compute_device.as_str(),
        if cfg.runtime_device_configured {
            "configured"
        } else {
            "default/auto"
        }
    );
    println!();
    println!("{}", style("config files").bold());
    println!("  user:          {}", user_config_path()?.display());
    println!("  system:        {}", system_config_path().display());
    if let Ok(explicit) = std::env::var("LPC_LLM_CONFIG") {
        println!("  LPC_LLM_CONFIG: {explicit}");
    }
    if cfg.loaded_from.is_empty() {
        println!("  loaded:        (defaults only)");
    } else {
        for p in &cfg.loaded_from {
            println!("  loaded:        {}", p.display());
        }
    }
    println!();
    println!(
        "{}",
        style(format!(
            "Privacy: put private corpora under train_dir; do not commit them. File name: {CONFIG_FILE_NAME}"
        ))
        .dim()
    );
    println!(
        "{}",
        style("First-run / device: `lpc-llm setup` or `lpc-llm config init --interactive`").dim()
    );
    Ok(())
}

pub fn init(force: bool, interactive: bool) -> Result<()> {
    if interactive {
        return crate::commands::setup::run();
    }
    let path = AppConfig::write_user_default(force)?;
    println!(
        "{} wrote {}",
        style("ok").green().bold(),
        path.display()
    );
    println!(
        "{}",
        style("Edit paths / [ui] / [runtime], or run `lpc-llm setup` for Q&A.")
            .dim()
    );
    Ok(())
}

pub fn get(key: &str) -> Result<()> {
    let cfg = AppConfig::load()?;
    let value = match key {
        "data_dir" | "paths.data_dir" => cfg.data_dir.display().to_string(),
        "train_dir" | "paths.train_dir" => cfg.train_dir.display().to_string(),
        "bin_dir" | "install.bin_dir" => cfg.bin_dir.display().to_string(),
        "mode" | "install.mode" => match cfg.install_mode {
            InstallMode::User => "user".into(),
            InstallMode::System => "system".into(),
        },
        "language" | "ui.language" => cfg.ui_language.as_str().into(),
        "device" | "runtime.device" => cfg.compute_device.as_str().into(),
        "user_config" => user_config_path()?.display().to_string(),
        "system_config" => system_config_path().display().to_string(),
        "config_file_name" => CONFIG_FILE_NAME.into(),
        other => {
            return Err(AppError::msg(format!(
                "unknown key `{other}` — try data_dir, train_dir, bin_dir, install.mode, ui.language, runtime.device, user_config"
            )));
        }
    };
    println!("{value}");
    Ok(())
}

pub fn print_example() -> Result<()> {
    print!("{}", AppConfig::default_toml());
    Ok(())
}
