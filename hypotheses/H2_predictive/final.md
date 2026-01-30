# H2 — Predictive (train/test)

## Claim being tested
`sink_mass` can be used to predict the positive class (hallucinated/incorrect).

## Protocol
- eval_mode: `holdout` (test_frac=0.3, repeats=5)

## What we fit (per model/run)
- baseline: AUROC/AUPRC of raw `sink_mass`
- classifier: logistic regression `y ~ zscore(sink_mass)` trained on train, evaluated on test

## Where the *actual* results are
- per-model/per-run results: `plots/metrics.json`
- per-model AUROC canvas: `plots/per_model_auroc.png` (if generated)
- example curves (last processed split): `plots/roc.png`, `plots/pr.png`

## Heterogeneity across runs (raw sink AUROC from run summaries)
Best:
- truthfulqa_mc / Qwen/Qwen2.5-0.5B-Instruct: AUROC=0.542 (n=415)
- truthfulqa_mc / EleutherAI/pythia-2.8b-deduped: AUROC=0.508 (n=415)
- truthfulqa_mc / TinyLlama/TinyLlama-1.1B-Chat-v1.0: AUROC=0.490 (n=415)

Worst:
- truthfulqa_mc / mistralai/Mistral-7B-Instruct-v0.3: AUROC=0.430 (n=415)
- truthfulqa_mc / mistralai/Mistral-Nemo-Instruct-2407: AUROC=0.454 (n=415)
- truthfulqa_mc / Qwen/Qwen2.5-7B-Instruct: AUROC=0.465 (n=415)

## Extra diagnostics
- per-run: `plots/runs/`
- aggregate: `plots/agg/` (table: `runs_summary.csv`)
