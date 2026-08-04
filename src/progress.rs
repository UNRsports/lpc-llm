//! Stderr progress lines for long hybrid load / pack / prefill steps.
//!
//! Uses `\r` updates so a quiet terminal still shows motion; final lines end
//! with newline. Safe when stderr is not a TTY (each update becomes a line).

use std::io::{self, IsTerminal, Write};
use std::time::Instant;

/// Numbered load phase, e.g. `[2/5] packing experts …`.
pub fn phase(step: u32, of: u32, msg: &str) {
    eprintln!("[{step}/{of}] {msg}");
    let _ = io::stderr().flush();
}

/// One-shot status (cache hit, done, etc.).
pub fn note(msg: &str) {
    eprintln!("  · {msg}");
    let _ = io::stderr().flush();
}

/// Counting progress with optional rate (items/s).
pub struct Counter {
    label: &'static str,
    total: usize,
    done: usize,
    start: Instant,
    last_emit: Instant,
    tty: bool,
}

impl Counter {
    pub fn start(label: &'static str, total: usize) -> Self {
        let now = Instant::now();
        let mut c = Self {
            label,
            total,
            done: 0,
            start: now,
            last_emit: now.checked_sub(std::time::Duration::from_secs(10)).unwrap_or(now),
            tty: io::stderr().is_terminal(),
        };
        c.emit(true);
        c
    }

    pub fn tick(&mut self) {
        self.done = self.done.saturating_add(1);
        let force = self.done >= self.total
            || self.done == 1
            || self.last_emit.elapsed().as_millis() >= 400
            || (self.total > 0 && self.done % progress_stride(self.total) == 0);
        if force {
            self.emit(false);
        }
    }

    fn emit(&mut self, initial: bool) {
        self.last_emit = Instant::now();
        let pct = if self.total == 0 {
            100
        } else {
            (100 * self.done) / self.total
        };
        let elapsed = self.start.elapsed().as_secs_f64().max(0.001);
        let rate = self.done as f64 / elapsed;
        let eta = if self.done > 0 && self.done < self.total {
            let rem = (self.total - self.done) as f64 / rate.max(0.001);
            format!(" ETA {rem:.0}s")
        } else {
            String::new()
        };
        let line = format!(
            "  {label}: {done}/{total} ({pct}%){eta}",
            label = self.label,
            done = self.done,
            total = self.total,
            pct = pct,
            eta = eta
        );
        if self.tty && !initial && self.done < self.total {
            eprint!("\r{line}          ");
        } else if self.tty && self.done >= self.total && self.total > 0 {
            eprintln!("\r{line}  ({elapsed:.1}s)          ");
        } else {
            eprintln!("{line}");
        }
        let _ = io::stderr().flush();
    }
}

fn progress_stride(total: usize) -> usize {
    if total <= 32 {
        1
    } else if total <= 256 {
        8
    } else {
        16
    }
}
