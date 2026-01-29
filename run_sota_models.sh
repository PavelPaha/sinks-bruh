#!/bin/bash
set -e
SCRIPT="$(cd "$(dirname "$0")" && pwd)/scripts/measure_accuracy.py"

# 1. Llama-3.1-70B (Latest Meta flagship) - 4bit
echo "=== Running Llama-3.1-70B ==="
uv run python "$SCRIPT" --model "meta-llama/Meta-Llama-3.1-70B-Instruct" --quantization 4bit || true

# 2. Qwen2.5-72B (Latest Qwen flagship) - 4bit
echo "=== Running Qwen2.5-72B ==="
uv run python "$SCRIPT" --model "Qwen/Qwen2.5-72B-Instruct" --quantization 4bit || true

# 3. Gemma-2-27B (Google SOTA mid-size)
echo "=== Running Gemma-2-27B ==="
uv run python "$SCRIPT" --model "google/gemma-2-27b-it" || true

# 4. Phi-3.5-medium (Microsoft SOTA small/mid)
echo "=== Running Phi-3.5-medium ==="
uv run python "$SCRIPT" --model "microsoft/Phi-3.5-medium-instruct" || true

echo "SOTA Models Done!"
