//! I/O module error types.

use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("mmap allocation failed ({0} bytes): {1}")]
    Mmap(usize, #[source] io::Error),

    #[error(
        "mlock failed ({0} bytes): {1}"
    )]
    Mlock(usize, #[source] io::Error),

    #[error("buffer size {0} is not aligned to {1}-byte O_DIRECT boundary")]
    Alignment(usize, usize),

    #[error("open weight file `{0}` with O_DIRECT: {1}")]
    Open(String, #[source] io::Error),

    #[error("io_uring setup failed: {0}")]
    UringSetup(#[source] io::Error),

    #[error("io_uring submit failed: {0}")]
    UringSubmit(#[source] io::Error),

    #[error("io_uring completion queue empty after submit_and_wait")]
    MissingCompletion,

    #[error("io_uring read failed (cqe.res={0}): {1}")]
    ReadFailed(i32, #[source] io::Error),

    #[error("short read: expected {expected} bytes, got {got}")]
    ShortRead { expected: usize, got: usize },

    #[error("no in-flight read to wait on")]
    NoInFlight,

    #[error("read already in flight (user_data={0}); wait before submit")]
    Busy(u64),

    #[error("I/O size {0} exceeds buffer capacity {1}")]
    BufferTooSmall(usize, usize),

    #[error("invalid buffer slot index {0}")]
    BadSlot(usize),

    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, IoError>;
