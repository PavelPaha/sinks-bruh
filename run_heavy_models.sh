#!/bin/bash
set -e
SCRIPT="$(cd "$(dirname "$0")" && pwd)/scripts/measure_accuracy.py"

# 1. Gemma-2-27B (Google)
echo "=== Running Gemma-2-27B ==="
uv run python "$SCRIPT" --model "google/gemma-2-27b-it" || true

# 2. Mixtral-8x7B (Mistral's big gun) - 4bit to save memory/speed
echo "=== Running Mixtral-8x7B ==="
uv run python "$SCRIPT" --model "mistralai/Mixtral-8x7B-Instruct-v0.1" --quantization 4bit || true

# 3. Llama-3-70B (The King) - 4bit required for single H100
echo "=== Running Llama-3-70B ==="
uv run python "$SCRIPT" --model "meta-llama/Meta-Llama-3-70B-Instruct" --quantization 4bit || true

# 4. Qwen1.5-32B (Comparison point for Qwen2.5) - just in case
echo "=== Running Qwen1.5-32B ==="
uv run python "$SCRIPT" --model "Qwen/Qwen1.5-32B-Chat" || true

echo "Heavy Models Done!"
