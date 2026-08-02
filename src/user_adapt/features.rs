//! Lightweight coding-style feature extraction from user text.

/// Extract human-readable style notes (indent, naming, comment density, etc.).
pub fn extract_style_features(user: &str, correction: Option<&str>) -> Vec<String> {
    let sample = correction.unwrap_or(user);
    let mut notes = Vec::new();

    if looks_like_code(sample) {
        if let Some(indent) = detect_indent(sample) {
            notes.push(format!("indent={indent}"));
        }
        let snake = count_ident_style(sample, IdentStyle::Snake);
        let camel = count_ident_style(sample, IdentStyle::Camel);
        let pascal = count_ident_style(sample, IdentStyle::Pascal);
        if snake + camel + pascal > 0 {
            let dominant = if snake >= camel && snake >= pascal {
                "snake_case"
            } else if camel >= pascal {
                "camelCase"
            } else {
                "PascalCase"
            };
            notes.push(format!("naming={dominant}"));
        }
        let comment_lines = sample
            .lines()
            .filter(|l| {
                let t = l.trim_start();
                t.starts_with("//") || t.starts_with('#') || t.starts_with("/*")
            })
            .count();
        let code_lines = sample.lines().filter(|l| !l.trim().is_empty()).count().max(1);
        let density = (comment_lines * 100) / code_lines;
        notes.push(format!("comment_density={density}%"));
    }

    if sample.chars().any(|c| ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{4e00}'..='\u{9fff}').contains(&c))
    {
        notes.push("locale=ja".into());
    } else if sample.is_ascii() {
        notes.push("locale=en".into());
    }

    notes
}

fn looks_like_code(s: &str) -> bool {
    let markers = ["fn ", "def ", "function ", "impl ", "class ", "=>", "{", "};", "import ", "use "];
    let lower = s.to_ascii_lowercase();
    markers.iter().any(|m| lower.contains(&m.to_ascii_lowercase()))
        || s.lines().filter(|l| l.starts_with("    ") || l.starts_with('\t')).count() >= 2
}

fn detect_indent(s: &str) -> Option<&'static str> {
    let mut spaces4 = 0usize;
    let mut spaces2 = 0usize;
    let mut tabs = 0usize;
    for line in s.lines() {
        if line.starts_with('\t') {
            tabs += 1;
        } else if line.starts_with("    ") {
            spaces4 += 1;
        } else if line.starts_with("  ") && !line.starts_with("   ") {
            spaces2 += 1;
        }
    }
    if tabs == 0 && spaces4 == 0 && spaces2 == 0 {
        return None;
    }
    if tabs >= spaces4 && tabs >= spaces2 {
        Some("tabs")
    } else if spaces4 >= spaces2 {
        Some("4-space")
    } else {
        Some("2-space")
    }
}

enum IdentStyle {
    Snake,
    Camel,
    Pascal,
}

fn count_ident_style(s: &str, style: IdentStyle) -> usize {
    let mut n = 0usize;
    for token in s.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if token.len() < 3 {
            continue;
        }
        match style {
            IdentStyle::Snake => {
                if token.contains('_')
                    && token.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                {
                    n += 1;
                }
            }
            IdentStyle::Camel => {
                if !token.contains('_')
                    && token.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                    && token.chars().any(|c| c.is_ascii_uppercase())
                {
                    n += 1;
                }
            }
            IdentStyle::Pascal => {
                if !token.contains('_')
                    && token.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    && token.chars().skip(1).any(|c| c.is_ascii_lowercase())
                    && token.chars().any(|c| c.is_ascii_uppercase())
                {
                    n += 1;
                }
            }
        }
    }
    n
}
