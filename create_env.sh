#!/bin/bash
# Create a fresh `cssm_uv` virtualenv at $HOME/envs/cssm_uv using `uv`,
# then install the project's requirements.
#
# Tested on hosts with CUDA 12.x toolchain available.
#
# Usage:
#   bash create_env.sh
#   # then in any new shell:
#   source activate.sh

set -euo pipefail

ENV_PATH="$HOME/envs/cssm_uv"
PY_VERSION="${PY_VERSION:-3.10}"
REQUIREMENTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "requirements.txt not found at $REQUIREMENTS" >&2
    exit 1
fi

echo "Creating venv at $ENV_PATH (python $PY_VERSION)"
mkdir -p "$(dirname "$ENV_PATH")"
uv venv --python "$PY_VERSION" "$ENV_PATH"

# shellcheck disable=SC1091
source "$ENV_PATH/bin/activate"

echo "Installing requirements from $REQUIREMENTS"
# JAX CUDA wheels live on the GCS find-links url declared inside requirements.txt.
uv pip install -r "$REQUIREMENTS"

# Optional fonts dir for the paper figures (Roboto). Best-effort: silent if
# already present or if the network is unavailable.
ROBOTO_DIR="$HOME/.local/share/fonts/roboto"
if [[ ! -d "$ROBOTO_DIR" ]]; then
    echo "Installing Roboto fonts to $ROBOTO_DIR"
    mkdir -p "$ROBOTO_DIR"
    for variant in Regular Bold Medium Italic BoldItalic; do
        target="$ROBOTO_DIR/Roboto-${variant}.ttf"
        if [[ ! -f "$target" ]]; then
            curl -sLo "$target" \
                "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-${variant}.ttf" || true
        fi
    done
fi

echo
echo "Done. To activate in a new shell:"
echo "  source activate.sh"
