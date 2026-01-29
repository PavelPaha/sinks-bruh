# Sink-Aware Decoding: Experimental Framework

This repository contains the code and experimental setup for analyzing **Attention Sinks** in Large Language Models and their correlation with **Hallucinations**, **Uncertainty (Entropy)**, and **Model Accuracy**.

The core hypothesis is that attention sink mass (attention to the initial token `<s>`) serves as a dynamic proxy for the model's reliance on parametric memory vs. context, and can predict generation quality.

## 🚀 Quick Start

### 1. Installation
We use `uv` for fast dependency management.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create venv
uv venv
source .venv/bin/activate
uv pip install torch transformers datasets matplotlib seaborn pandas scipy accelerate bitsandbytes
```

### 2. Project Structure
```
.
├── scripts/
│   ├── measure_accuracy.py      # Worker: MMLU correctness + sink_mass + entropy
│   ├── compare_accuracy_sink.py # Plotter: Accuracy vs sink_mass curves across models
│   └── analyze_mmlu_results.py  # Analyzer: correlations + per-subject summary
├── configs/                 # Experiment configs (model lists, defaults)
├── run_sota_models.sh       # Orchestrator: Runs accuracy experiments on SOTA models (Llama-3.1, Qwen2.5)
├── notebooks/               # Scratchpad for rapid hypothesis testing
└── artifacts/
    ├── results/             # Output json files (per-example measurements)
    └── plots/               # Output figures
```

## 🧪 Running Experiments

### Experiment: MMLU Accuracy vs Sink Mass (The "Sink-Confidence" Curve)
**Main Result:** Tests if sink mass predicts correctness on MMLU.

1. **Run SOTA Models:**
   ```bash
   bash run_sota_models.sh
   ```
   *Includes Llama-3.1-70B, Qwen2.5-72B, etc. Supports 4-bit quantization.*

2. **Visualize Comparison:**
   ```bash
   uv run python scripts/compare_accuracy_sink.py
   ```
   *Outputs:* `artifacts/plots/accuracy_sink_comparison_full.png` (The Grand Plot)

3. **Analyze correlations / entropy / per-subject breakdown:**
   ```bash
   uv run python scripts/analyze_mmlu_results.py --plots
   ```
   *Outputs:* console summaries + `artifacts/plots/mmlu_subject_sink_*.png`

> Note: The repository currently focuses on MMLU experiments. Some filenames mentioned in older notes (e.g. `measure_sink.py`, `visualize_entropy.py`) are not present in this snapshot.

## 📊 Key Plots Description

| Plot | Description | Interpretation |
|------|-------------|----------------|
| `accuracy_sink_comparison_full.png` | **Accuracy vs Sink Mass**. Curves for multiple models. | **Rising curve** (High Sink -> High Acc) indicates "Confidence". **Falling curve** indicates "Confusion". |
| `global_entropy_map.png` | **Entropy vs Sink Mass** (Hexbin). | Global trend showing how uncertainty drives attention to/from the sink. |
| `barplot_real_window_1.png` | **Sink Mass by Task**. | Shows which tasks trigger high sink attention (typically reasoning/grounding tasks). |
| `window_sensitivity.png` | **Sink Mass over Time**. | How sink attention aggregates over different context window sizes. |

## 🛠 Advanced Usage

**Run a single accuracy check:**
```bash
uv run python measure_accuracy.py --model "Qwen/Qwen2.5-32B-Instruct" --samples 2000
```

**Run with 4-bit quantization (for 70B+ models):**
```bash
uv run python measure_accuracy.py --model "meta-llama/Meta-Llama-3.1-70B-Instruct" --quantization 4bit
```
# sinks-bruh
# sinks-bruh
