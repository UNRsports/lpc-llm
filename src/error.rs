//! Application-level errors.

use thiserror::Error;

use crate::io::error::IoError;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Msg(String),

    #[error("unknown model `{0}` — run `lpc-llm list` to see the catalog")]
    UnknownModel(String),

    #[error("model `{0}` is not installed locally — run `lpc-llm pull {0}`")]
    NotInstalled(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Prefetch(#[from] IoError),
}

impl AppError {
    pub fn msg(s: impl Into<String>) -> Self {
        Self::Msg(s.into())
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
