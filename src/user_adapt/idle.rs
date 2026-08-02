//! Linux idle detection (X11 Screensaver / Wayland ext-idle / fallback timer).

use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{AppError, Result};

/// Best-effort idle time in seconds.
///
/// Order: `xprintidle` → GNOME Mutter IdleMonitor via `gdbus` → `None` if unavailable.
pub fn idle_seconds() -> Option<u64> {
    if let Some(ms) = xprintidle_ms() {
        return Some(ms / 1000);
    }
    if let Some(ms) = gnome_idle_ms() {
        return Some(ms / 1000);
    }
    None
}

/// Block until idle ≥ `min_idle_secs`, or until `max_wait` elapses.
///
/// When no system idle API is available, uses a simple wall-clock wait of
/// `min_idle_secs` (conservative “no user interaction assumed” for daemon mode).
pub fn wait_until_idle(min_idle_secs: u64, max_wait: Duration) -> Result<()> {
    let start = Instant::now();
    let poll = Duration::from_secs(5);
    loop {
        if start.elapsed() > max_wait {
            return Err(AppError::msg(format!(
                "idle wait timed out after {}s (need {min_idle_secs}s idle)",
                max_wait.as_secs()
            )));
        }
        match idle_seconds() {
            Some(idle) if idle >= min_idle_secs => return Ok(()),
            Some(_) => thread::sleep(poll),
            None => {
                // Fallback: sleep the required idle window once.
                let remain = Duration::from_secs(min_idle_secs).saturating_sub(start.elapsed());
                if remain.is_zero() {
                    return Ok(());
                }
                let slice = remain.min(max_wait.saturating_sub(start.elapsed()));
                thread::sleep(slice);
                return Ok(());
            }
        }
    }
}

fn xprintidle_ms() -> Option<u64> {
    let out = Command::new("xprintidle").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse::<u64>().ok()
}

fn gnome_idle_ms() -> Option<u64> {
    // org.gnome.Mutter.IdleMonitor GetIdletime → uint64 milliseconds
    let out = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.gnome.Mutter.IdleMonitor",
            "--object-path",
            "/org/gnome/Mutter/IdleMonitor/Core",
            "--method",
            "org.gnome.Mutter.IdleMonitor.GetIdletime",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // Typical: `(uint64 12345,)\n`
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u64>().ok()
}
