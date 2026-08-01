#!/usr/bin/env bash
# Install a PATH entry that always tracks this repo's build artifact.
#
# Default: ~/.local/bin/lpc-llm → <repo>/target/debug/lpc-llm
# So every `cargo build` / `cargo test` refreshes what `lpc-llm` runs.
#
# Usage:
#   ./scripts/install-dev.sh           # debug (dev default)
#   ./scripts/install-dev.sh --release # track target/release instead
#   ./scripts/install-dev.sh --no-build

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="debug"
DO_BUILD=1

for arg in "$@"; do
  case "$arg" in
    --release) PROFILE="release" ;;
    --debug) PROFILE="debug" ;;
    --no-build) DO_BUILD=0 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

BIN_DIR="${LPC_LLM_BIN_DIR:-$HOME/.local/bin}"
# Always prefer the repo-local target/ so the symlink stays stable across shells
# (ignore ambient CARGO_TARGET_DIR unless LPC_LLM_TARGET_DIR is set).
TARGET_DIR="${LPC_LLM_TARGET_DIR:-$ROOT/target}"
TARGET="$TARGET_DIR/$PROFILE/lpc-llm"
LINK="$BIN_DIR/lpc-llm"

if [[ "$DO_BUILD" -eq 1 ]]; then
  if [[ "$PROFILE" == "release" ]]; then
    (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release)
  else
    (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build)
  fi
fi

if [[ ! -x "$TARGET" ]]; then
  echo "error: binary missing: $TARGET (run without --no-build, or cargo build first)" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
ln -sfn "$TARGET" "$LINK"

echo "installed: $LINK -> $TARGET"
if command -v lpc-llm >/dev/null 2>&1; then
  echo "on PATH:   $(command -v lpc-llm)"
else
  echo "note:      add $BIN_DIR to PATH (e.g. export PATH=\"\$HOME/.local/bin:\$PATH\")"
fi
ls -l "$LINK"
