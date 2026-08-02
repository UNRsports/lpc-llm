//! Quantized GGUF model load + greedy/temperature sampling (Candle, pure Rust).
//!
//! Supports the architectures used by the built-in catalog: llama/gemma/gemma2,
//! qwen2, phi3 (via candle-transformers quantized backends).

use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use tokenizers::Tokenizer;

use crate::device::ComputeContext;
use crate::error::{AppError, Result};

enum Weights {
    Llama(LlamaWeights),
    Qwen2(Qwen2Weights),
    Phi3(Phi3Weights),
}

impl Weights {
    fn forward(&mut self, x: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(w) => w.forward(x, offset),
            Self::Qwen2(w) => w.forward(x, offset),
            Self::Phi3(w) => w.forward(x, offset),
        }
    }
}

pub struct Engine {
    weights: Weights,
    device: Device,
    architecture: String,
    model_path: PathBuf,
    device_label: String,
    compute: ComputeContext,
}

impl Engine {
    pub fn load(path: impl AsRef<Path>, compute: ComputeContext) -> Result<Self> {
        let path = path.as_ref();
        let device = compute.device().clone();
        let device_label = compute.label().to_string();
        if matches!(
            compute.backend,
            crate::device::ResolvedBackend::Vulkan
        ) {
            eprintln!(
                "compute backend: Vulkan (eager path uses Candle CPU/CUDA tensors; hybrid preferred for Vulkan GEMM)"
            );
        } else {
            eprintln!("compute backend: {}", compute.label());
        }
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| AppError::msg(format!("GGUF read: {e}")))?;

        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(ToOwned::to_owned)
            .ok_or_else(|| AppError::msg("GGUF missing general.architecture"))?;

        let arch = architecture.to_ascii_lowercase();
        let weights = match arch.as_str() {
            "llama" | "mistral" | "gemma" | "gemma2" | "mixtral" => {
                Weights::Llama(
                    LlamaWeights::from_gguf(content, &mut file, &device)
                        .map_err(|e| AppError::msg(format!("load llama-family: {e}")))?,
                )
            }
            "qwen2" => Weights::Qwen2(
                Qwen2Weights::from_gguf(content, &mut file, &device)
                    .map_err(|e| AppError::msg(format!("load qwen2: {e}")))?,
            ),
            "phi3" => Weights::Phi3(
                Phi3Weights::from_gguf(false, content, &mut file, &device)
                    .map_err(|e| AppError::msg(format!("load phi3: {e}")))?,
            ),
            other => {
                return Err(AppError::msg(format!(
                    "unsupported GGUF architecture `{other}` \
                     (supported: llama/mistral/gemma/gemma2/qwen2/phi3)"
                )));
            }
        };

        Ok(Self {
            weights,
            device,
            architecture,
            model_path: path.to_path_buf(),
            device_label,
            compute,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn device_name(&self) -> &str {
        &self.device_label
    }

    /// Reload weights to clear KV cache between turns.
    pub fn reset_state(&mut self) -> Result<()> {
        let reloaded = Self::load(&self.model_path, self.compute.clone())?;
        self.weights = reloaded.weights;
        self.architecture = reloaded.architecture;
        self.device_label = reloaded.device_label;
        Ok(())
    }

    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| AppError::msg(format!("tokenize: {e}")))?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(AppError::msg("empty prompt after tokenization"));
        }

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), None);
        let eos = eos_token_ids(tokenizer);

        let input = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?;
        let mut logits = prepare_logits(self.weights.forward(&input, 0)?)?;

        let mut generated = String::new();
        for _ in 0..max_tokens {
            let next = logits_processor.sample(&logits)?;
            tokens.push(next);

            let piece = tokenizer
                .decode(&[next], true)
                .map_err(|e| AppError::msg(format!("decode: {e}")))?;
            on_token(&piece)?;
            generated.push_str(&piece);

            if eos.contains(&next) {
                break;
            }

            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = prepare_logits(self.weights.forward(&input, tokens.len() - 1)?)?;
        }

        Ok(generated)
    }
}

fn prepare_logits(logits: Tensor) -> Result<Tensor> {
    let mut logits = logits.squeeze(0)?;
    if logits.dims().len() > 1 {
        let last = logits.dim(0)? - 1;
        logits = logits.get(last)?;
    }
    Ok(logits.clamp(-100.0, 100.0)?)
}

fn eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    const CANDIDATES: &[&str] = &[
        "<|endoftext|>",
        "</s>",
        "<|eot_id|>",
        "<|end|>",
        "<end_of_turn>",
        "<|im_end|>",
    ];
    let mut ids = Vec::new();
    for c in CANDIDATES {
        if let Some(id) = tokenizer.token_to_id(c) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

impl From<candle_core::Error> for AppError {
    fn from(e: candle_core::Error) -> Self {
        AppError::msg(format!("candle: {e}"))
    }
}
