"""
Text-only Sink experiments runner.

Goals:
- Run ONE model on ONE dataset/task per invocation.
- Save per-example metrics + compact "attention map" (L x H sink-to-prefix mass) to jsonl.gz.
- Support multi-GPU via --cuda_device (sets CUDA_VISIBLE_DEVICES early).

We intentionally do NOT store full [S x S] attention matrices (too large).
Instead, we store per-layer/head sink mass to the first K tokens, for a chosen query set.
This is enough to build layer×head heatmaps (like LLaMA sink plots) and do H1–H4 analyses.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


def _early_set_cuda_visible_devices(argv: List[str]) -> None:
    """
    Must run before importing torch/transformers to make CUDA device selection reliable.
    """
    if "--cuda_device" not in argv:
        return
    try:
        idx = argv.index("--cuda_device")
        dev = argv[idx + 1]
    except Exception:
        return
    if dev is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)


_early_set_cuda_visible_devices(sys.argv)

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _single_token_id(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def _predict_choice_from_logits(tokenizer, next_token_logits: torch.Tensor) -> str:
    """
    Robust A/B/C/D scoring instead of argmax over whole vocab.
    """
    variants = {
        "A": ["A", " A", "\nA", "\n A"],
        "B": ["B", " B", "\nB", "\n B"],
        "C": ["C", " C", "\nC", "\n C"],
        "D": ["D", " D", "\nD", "\n D"],
    }

    scores: Dict[str, float] = {}
    for label, texts in variants.items():
        token_ids = [_single_token_id(tokenizer, t) for t in texts]
        token_ids = [tid for tid in token_ids if tid is not None]
        if not token_ids:
            continue
        scores[label] = max(float(next_token_logits[tid].item()) for tid in token_ids)

    if scores:
        return max(scores.items(), key=lambda kv: kv[1])[0]

    # Fallback: decode argmax token
    pred_token_id = int(torch.argmax(next_token_logits).item())
    pred_text = tokenizer.decode([pred_token_id]).strip().upper()
    for ch in ("A", "B", "C", "D"):
        if pred_text.startswith(ch):
            return ch
    return pred_text[:1] if pred_text else "?"


def format_prompt(tokenizer, model_name: str, raw_prompt: str, chat_mode: str) -> str:
    """
    chat_mode:
      - auto: use chat template if available and model looks like chat/instruct
      - on:   force chat template if available
      - off:  never use chat template
    """
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    looks_chat = any(k in (model_name or "") for k in ("Instruct", "Chat", "chat", "-it", "it"))

    if chat_mode == "off":
        use_chat = False
    elif chat_mode == "on":
        use_chat = has_chat
    else:
        use_chat = has_chat and looks_chat

    if not use_chat:
        return raw_prompt

    messages = [{"role": "user", "content": raw_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return raw_prompt


def compute_entropy_from_logits(next_token_logits: torch.Tensor) -> float:
    logits = next_token_logits.float()
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


@dataclass
class RunMeta:
    task: str
    dataset: str
    subset: Optional[str]
    split: str
    model: str
    revision: Optional[str]
    samples: int
    seed: int
    sink_tokens: int
    query_mode: str
    query_start: int
    chat: str
    quantization: str
    device: str
    cuda_visible_devices: Optional[str]
    transformers_version: str
    torch_version: str
    timestamp: float


def _safe_model_name(model_id: str) -> str:
    # stable + filesystem-friendly
    m = model_id.replace("/", "__").replace(":", "_")
    return m


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def load_model_and_tokenizer(model_id: str, revision: Optional[str], quantization: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=True)

    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: Dict[str, Any] = {
        "attn_implementation": "eager",  # required for output_attentions in many setups
        "trust_remote_code": True,
    }

    if device == "cpu":
        model_kwargs["device_map"] = None
        model_kwargs["torch_dtype"] = torch.float32
    else:
        # after CUDA_VISIBLE_DEVICES, "cuda:0" is the selected GPU
        model_kwargs["device_map"] = "auto"
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **model_kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def sink_map_from_attentions(
    attentions: Tuple[torch.Tensor, ...],
    sink_tokens: int,
    query_mode: Literal["last", "range"],
    query_start: int,
) -> Tuple[float, List[float], List[List[float]]]:
    """
    Returns:
      sink_mass_mean: scalar averaged over layers & heads
      sink_by_layer: [L] averaged over heads
      sink_by_layer_head: [L][H]
    """
    sink_by_layer_head: List[List[float]] = []
    sink_by_layer: List[float] = []

    # attentions[l]: [B,H,S,S]
    for a in attentions:
        a0 = a[0].float()  # [H,S,S]
        if query_mode == "last":
            # queries = last token only -> [H, sink_tokens]
            sink_h = a0[:, -1, :sink_tokens].sum(dim=-1)  # [H]
        else:
            start = max(int(query_start), 0)
            # queries = [start: ] average across query positions
            sink_h = a0[:, start:, :sink_tokens].mean(dim=-2).sum(dim=-1)  # [H]

        sink_h_list = [float(x.item()) for x in sink_h]
        sink_by_layer_head.append(sink_h_list)
        sink_by_layer.append(float(np.mean(sink_h_list)) if sink_h_list else float("nan"))

    sink_mass_mean = float(np.mean(sink_by_layer)) if sink_by_layer else float("nan")
    return sink_mass_mean, sink_by_layer, sink_by_layer_head


def iter_task_truthfulqa_mc(ds, samples: int, seed: int) -> Iterable[Dict[str, Any]]:
    """
    TruthfulQA has multiple variants; we use a robust loader with fallbacks.
    We treat this as a *hallucination* task: hallucinated = not correct (under MC label).
    """
    if len(ds) > samples:
        ds = ds.shuffle(seed=seed).select(range(samples))
    for ex in ds:
        # Typical fields: question, mc1_targets / mc2_targets etc. We'll prefer MC1 if present.
        q = ex.get("question") or ex.get("prompt") or ex.get("query")

        # Build options
        choices = None
        answer_idx = None

        # Newer datasets often provide 'mc1_targets' with 'choices' and 'labels'
        mc1 = ex.get("mc1_targets")
        if isinstance(mc1, dict):
            choices = mc1.get("choices")
            labels = mc1.get("labels")
            if isinstance(labels, (list, tuple)) and len(labels) == len(choices):
                # label 1 means correct, 0 incorrect
                try:
                    answer_idx = int(np.argmax(np.array(labels, dtype=float)))
                except Exception:
                    answer_idx = None

        # Another common format: 'choices' and 'answer'
        if choices is None:
            choices = ex.get("choices")
        if answer_idx is None:
            # could be int or string
            ans = ex.get("answer")
            if isinstance(ans, int):
                answer_idx = ans

        if not q or not choices or answer_idx is None:
            # skip unknown formats
            continue

        # Keep only first 4 options to fit A/B/C/D protocol.
        # TruthfulQA occasionally has <4 options; skip those examples for now
        # to keep a consistent A/B/C/D evaluation protocol.
        choices4 = list(choices)[:4]
        if len(choices4) < 4:
            continue
        if answer_idx >= len(choices4):
            continue

        int_to_char = {0: "A", 1: "B", 2: "C", 3: "D"}
        target = int_to_char[answer_idx]

        raw_prompt = (
            f"Question: {q}\n"
            f"A. {choices4[0]}\n"
            f"B. {choices4[1]}\n"
            f"C. {choices4[2]}\n"
            f"D. {choices4[3]}\n"
            f"Answer:"
        )

        yield {
            "raw_prompt": raw_prompt,
            "target": target,
            "label_hallucinated": None,
            "meta": {"question_id": ex.get("id") or _stable_id(str(q)), "source": "truthfulqa"},
        }


def iter_task_halueval(ds, samples: int, seed: int) -> Iterable[Dict[str, Any]]:
    """
    HaluEval provides a passage, a question, a proposed answer, and a PASS/FAIL label.
    We turn it into a binary classification:
      - A = PASS (supported)
      - B = FAIL (not supported / hallucinated)
    """
    if len(ds) > samples:
        ds = ds.shuffle(seed=seed).select(range(samples))

    for ex in ds:
        passage = ex.get("passage", "")
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        label = (ex.get("label") or "").strip().upper()
        if not passage or not question or not answer or label not in ("PASS", "FAIL"):
            continue

        label_h = bool(label == "FAIL")
        target = "A" if not label_h else "B"
        raw_prompt = (
            "You are given a passage, a question, and a proposed answer.\n"
            "Decide whether the answer is fully supported by the passage.\n"
            "Answer with:\n"
            "A. Supported by the passage (PASS)\n"
            "B. Not supported / hallucinated (FAIL)\n\n"
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n"
            "Choice:"
        )

        yield {
            "raw_prompt": raw_prompt,
            "target": target,
            "label_hallucinated": label_h,
            "meta": {"id": ex.get("id"), "source_ds": ex.get("source_ds"), "score": ex.get("score")},
        }


def iter_task_freshqa_false_premise(
    ds, samples: int, seed: int, language: str = "English"
) -> Iterable[Dict[str, Any]]:
    """
    FreshQA-multilingual provides a question + boolean false_premise.
    We turn it into a binary classification:
      - A = has a false premise (unanswerable as stated)
      - B = no false premise (plausibly answerable)
    """
    if len(ds) > samples:
        ds = ds.shuffle(seed=seed).select(range(samples))

    for ex in ds:
        q = ex.get(language) or ex.get("question")
        fp = ex.get("false_premise")
        if q is None or fp is None:
            continue

        label_h = bool(fp)
        target = "A" if label_h else "B"
        raw_prompt = (
            "Decide whether the following question contains a false premise.\n"
            "Answer with:\n"
            "A. Has a false premise (cannot be answered as stated)\n"
            "B. No false premise (plausibly answerable)\n\n"
            f"Question: {q}\n"
            "Choice:"
        )

        yield {
            "raw_prompt": raw_prompt,
            "target": target,
            "label_hallucinated": label_h,
            "meta": {
                "id": ex.get("id"),
                "fact_type": ex.get("fact_type"),
                "num_hops": ex.get("num_hops"),
                "language": language,
            },
        }


def iter_task_mmlu(ds, samples: int, seed: int) -> Iterable[Dict[str, Any]]:
    if len(ds) > samples:
        ds = ds.shuffle(seed=seed).select(range(samples))

    int_to_char = {0: "A", 1: "B", 2: "C", 3: "D"}
    for ex in ds:
        raw_prompt = (
            f"Question: {ex['question']}\n"
            f"A. {ex['choices'][0]}\n"
            f"B. {ex['choices'][1]}\n"
            f"C. {ex['choices'][2]}\n"
            f"D. {ex['choices'][3]}\n"
            f"Answer:"
        )
        target = int_to_char[int(ex["answer"])]
        yield {
            "raw_prompt": raw_prompt,
            "target": target,
            "label_hallucinated": None,
            "meta": {"subject": ex.get("subject"), "question_id": _stable_id(str(ex.get("question")))},
        }


def load_task_dataset(task: str, split: str) -> Tuple[Any, str, Optional[str]]:
    """
    Returns (dataset, dataset_id, subset)
    """
    if task == "mmlu":
        try:
            ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
        except Exception:
            ds = load_dataset("cais/mmlu", "all", split=split)
        return ds, "cais/mmlu", "all"

    if task == "truthfulqa_mc":
        # Try the commonly used variants
        # 1) truthful_qa multiple_choice
        try:
            ds = load_dataset("truthful_qa", "multiple_choice", split=split)
            return ds, "truthful_qa", "multiple_choice"
        except Exception:
            pass
        # 2) HF community mirror
        ds = load_dataset("truthfulqa/truthful_qa", split=split)
        return ds, "truthfulqa/truthful_qa", None

    if task == "halueval":
        ds = load_dataset("flowaicom/HaluEval", split=split)
        return ds, "flowaicom/HaluEval", None

    if task == "freshqa_false_premise":
        # Dataset itself is shipped as test-only (it has an internal column 'split' = TEST)
        ds = load_dataset("SeaLLMs/FreshQA-multilingual", split="test")
        return ds, "SeaLLMs/FreshQA-multilingual", None

    raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["mmlu", "truthfulqa_mc", "halueval", "freshqa_false_premise"])
    parser.add_argument("--split", type=str, default="test", help="Dataset split where applicable (TruthfulQA often uses validation).")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cuda_device", type=int, default=None, help="If set, assigns CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "8bit"])

    parser.add_argument("--sink_tokens", type=int, default=4, help="Sink window K: number of initial tokens.")
    parser.add_argument("--query_mode", type=str, default="last", choices=["last", "range"])
    parser.add_argument("--query_start", type=int, default=0, help="Used only for query_mode=range.")
    parser.add_argument("--chat", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--language", type=str, default="English", help="Used for multilingual datasets like FreshQA.")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_text", action="store_true", help="Store prompt text in outputs (bigger files).")
    args = parser.parse_args()

    # Determinism for dataset sampling
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds, dataset_id, subset = load_task_dataset(args.task, args.split)

    print(f"Loading model={args.model} quantization={args.quantization} device={args.device}...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.revision, args.quantization, args.device)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "sink_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = _safe_model_name(args.model)
    run_name = f"{args.task}__{safe_model}__K{args.sink_tokens}__q{args.query_mode}"
    if args.query_mode == "range":
        run_name += f"_start{args.query_start}"
    if args.chat != "auto":
        run_name += f"__chat{args.chat}"
    if args.quantization != "none":
        run_name += f"__{args.quantization}"

    results_path = out_dir / f"{run_name}.jsonl.gz"
    meta_path = out_dir / f"{run_name}.meta.json"

    meta = RunMeta(
        task=args.task,
        dataset=dataset_id,
        subset=subset,
        split=args.split,
        model=args.model,
        revision=args.revision,
        samples=args.samples,
        seed=args.seed,
        sink_tokens=args.sink_tokens,
        query_mode=args.query_mode,
        query_start=args.query_start,
        chat=args.chat,
        quantization=args.quantization,
        device=args.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        transformers_version=__import__("transformers").__version__,
        torch_version=torch.__version__,
        timestamp=time.time(),
    )

    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"Saved run meta to {meta_path}")

    if args.task == "mmlu":
        iterator = iter_task_mmlu(ds, args.samples, args.seed)
        label_hallucination_available = False
    elif args.task == "truthfulqa_mc":
        iterator = iter_task_truthfulqa_mc(ds, args.samples, args.seed)
        label_hallucination_available = False
    elif args.task == "halueval":
        iterator = iter_task_halueval(ds, args.samples, args.seed)
        label_hallucination_available = True
    else:
        iterator = iter_task_freshqa_false_premise(ds, args.samples, args.seed, language=args.language)
        label_hallucination_available = True

    # Materialize the iterator to know the *actual* number of valid examples after filtering.
    # This makes tqdm progress reporting accurate (some datasets have fewer usable samples).
    examples = list(iterator)
    print(f"Prepared {len(examples)} valid examples (requested up to {args.samples}).")

    device = "cpu" if args.device == "cpu" else "cuda"

    n = 0
    n_correct = 0
    n_total = 0

    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        for ex in tqdm(examples, total=len(examples)):
            raw_prompt = ex["raw_prompt"]
            target = ex["target"]
            prompt = format_prompt(tokenizer, args.model, raw_prompt, chat_mode=args.chat)

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)

            next_logits = out.logits[0, -1, :].float()
            entropy = compute_entropy_from_logits(next_logits)
            pred = _predict_choice_from_logits(tokenizer, next_logits)

            is_correct = str(pred).upper().startswith(str(target).upper())
            n_total += 1
            if is_correct:
                n_correct += 1

            if out.attentions is None:
                raise RuntimeError("Model did not return attentions (attentions=None).")

            sink_mass, sink_by_layer, sink_by_layer_head = sink_map_from_attentions(
                out.attentions,
                sink_tokens=args.sink_tokens,
                query_mode=args.query_mode,  # type: ignore[arg-type]
                query_start=args.query_start,
            )

            label_h = ex.get("label_hallucinated", None)
            if label_hallucination_available:
                hallucinated = bool(label_h)
            else:
                # operational definition for TruthfulQA-MC: hallucinated = not correct
                hallucinated = bool(not is_correct) if args.task == "truthfulqa_mc" else None

            record: Dict[str, Any] = {
                "idx": n,
                "task": args.task,
                "model": args.model,
                "target": target,
                "pred": pred,
                "correct": bool(is_correct),
                "hallucinated": hallucinated,
                "label_hallucinated": label_h,
                "entropy": float(entropy),
                "sink_mass": float(sink_mass),
                "sink_by_layer": sink_by_layer,
                "sink_by_layer_head": sink_by_layer_head,
                "seq_len": int(inputs["input_ids"].shape[-1]),
                "sink_tokens": int(args.sink_tokens),
                "query_mode": args.query_mode,
                "query_start": int(args.query_start),
                "chat_mode": args.chat,
                "quantization": args.quantization,
                "meta": ex.get("meta", {}),
            }
            if args.save_text:
                record["raw_prompt"] = raw_prompt
                record["prompt"] = prompt

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    acc = n_correct / max(1, n_total)
    print(f"Done. Wrote {n_total} examples to {results_path}")
    print(f"Accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()

