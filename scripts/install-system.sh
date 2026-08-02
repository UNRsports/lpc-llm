#!/usr/bin/env bash
# Install a shared machine binary only (no user data).
#
# Default destination: /usr/local/bin/lpc-llm
# Override via config_lpcllm [install] or LPC_LLM_BIN_DIR.
#
# Usage:
#   ./scripts/install-system.sh              # build --release, install to bin_dir
#   ./scripts/install-system.sh --no-build   # copy existing target/release/lpc-llm
#
# Each user's models / corpora stay under their own paths.data_dir
# (default ~/.local/share/lpc-llm). Never copy home data into /usr.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DO_BUILD=1

for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

TARGET_DIR="${LPC_LLM_TARGET_DIR:-$ROOT/target}"
TARGET="$TARGET_DIR/release/lpc-llm"

if [[ "$DO_BUILD" -eq 1 ]]; then
  (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release)
fi

if [[ ! -x "$TARGET" ]]; then
  echo "error: binary missing: $TARGET" >&2
  exit 1
fi

resolve_bin_dir() {
  if [[ -n "${LPC_LLM_BIN_DIR:-}" ]]; then
    echo "$LPC_LLM_BIN_DIR"
    return
  fi
  if [[ -x "$TARGET" ]]; then
    # Prefer resolved config from the freshly built binary (no install yet).
    if BIN_DIR="$("$TARGET" config get bin_dir 2>/dev/null)"; then
      # If mode is still "user", force system default unless config says system.
      MODE="$("$TARGET" config get install.mode 2>/dev/null || echo user)"
      if [[ "$MODE" == "system" ]]; then
        echo "$BIN_DIR"
        return
      fi
    fi
  fi
  echo "/usr/local/bin"
}

BIN_DIR="$(resolve_bin_dir)"
# Expand leading ~/
if [[ "$BIN_DIR" == ~* ]]; then
  BIN_DIR="${BIN_DIR/#\~/$HOME}"
fi

DEST="$BIN_DIR/lpc-llm"
echo "installing shared binary: $DEST"
echo "(user data is NOT installed; each account uses its own data_dir)"

install -d "$BIN_DIR"
install -m 755 "$TARGET" "$DEST"
echo "installed: $DEST"
command -v lpc-llm >/dev/null 2>&1 && echo "on PATH: $(command -v lpc-llm)" || true
