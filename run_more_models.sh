#!/bin/bash
set -e
SCRIPT="$(cd "$(dirname "$0")" && pwd)/scripts/measure_accuracy.py"

# 1. Qwen-7B (Small check)
echo "=== Running Qwen-7B ==="
uv run python "$SCRIPT" --model "Qwen/Qwen2.5-7B-Instruct" || true

# 2. Llama-3-8B (The standard small model)
echo "=== Running Llama-3-8B ==="
uv run python "$SCRIPT" --model "meta-llama/Meta-Llama-3-8B-Instruct" || true

# 3. Mistral-Nemo-12B (Mid-range check)
echo "=== Running Mistral-Nemo-12B ==="
uv run python "$SCRIPT" --model "mistralai/Mistral-Nemo-Instruct-2407" || true

# 4. Yi-34B (Large check)
echo "=== Running Yi-34B ==="
uv run python "$SCRIPT" --model "01-ai/Yi-1.5-34B-Chat" || true

echo "All Done!"
