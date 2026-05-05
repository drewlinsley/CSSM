#!/bin/bash
# Setup script for CSSM environment using uv
# Usage: source setup_env.sh

set -e

ENV_NAME="cssm_uv"
ENV_PATH="$HOME/envs/$ENV_NAME"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment with uv
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    uv venv "$ENV_PATH" --python 3.11
fi

# Activate environment
echo "Activating environment..."
source "$ENV_PATH/bin/activate"

# Set CUDA library path
export LD_LIBRARY_PATH=/oscar/rt/9.6/25/spack/x86_64_v4/cuda-12.6.0-wefccjmiz6tr2sj6lgcxj6ihwvcnqkuo/lib64:$LD_LIBRARY_PATH

# Install requirements with uv
echo "Installing requirements..."
uv pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate in future sessions, run:"
echo ""
echo "  source $ENV_PATH/bin/activate"
echo "  export LD_LIBRARY_PATH=/oscar/rt/9.6/25/spack/x86_64_v4/cuda-12.6.0-wefccjmiz6tr2sj6lgcxj6ihwvcnqkuo/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "Or just: source setup_env.sh"
