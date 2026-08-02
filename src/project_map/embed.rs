//! Lightweight node embeddings via hashed character n-grams (no neural model).

pub const EMBED_DIM: usize = 32;

/// Fixed-size hashed n-gram embedding in `[-1, 1]`-ish f32 space (stored as i16).
pub fn hash_embed(text: &str) -> [f32; EMBED_DIM] {
    let mut v = [0.0f32; EMBED_DIM];
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    if bytes.is_empty() {
        return v;
    }
    // unigrams + character trigrams
    for &b in bytes {
        let idx = (b as usize) % EMBED_DIM;
        v[idx] += 1.0;
    }
    if bytes.len() >= 3 {
        for w in bytes.windows(3) {
            let h = fnv1a_32(w) as usize % EMBED_DIM;
            v[h] += 1.0;
        }
    }
    // tokens
    for tok in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if tok.len() < 2 {
            continue;
        }
        let h = fnv1a_32(tok.as_bytes()) as usize % EMBED_DIM;
        v[h] += 2.0;
    }
    normalize(&mut v);
    v
}

pub fn cosine(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..EMBED_DIM {
        dot += a[i] * b[i];
    }
    dot
}

pub fn embed_to_i16(v: &[f32; EMBED_DIM]) -> [i16; EMBED_DIM] {
    let mut out = [0i16; EMBED_DIM];
    for i in 0..EMBED_DIM {
        let x = (v[i] * 32767.0).clamp(-32767.0, 32767.0);
        out[i] = x as i16;
    }
    out
}

pub fn embed_from_i16(v: &[i16; EMBED_DIM]) -> [f32; EMBED_DIM] {
    let mut out = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        out[i] = v[i] as f32 / 32767.0;
    }
    normalize(&mut out);
    out
}

fn normalize(v: &mut [f32; EMBED_DIM]) {
    let mut norm = 0.0f32;
    for x in v.iter() {
        norm += x * x;
    }
    norm = norm.sqrt();
    if norm < 1e-6 {
        return;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in data {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}
