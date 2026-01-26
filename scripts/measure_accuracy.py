import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

def _single_token_id(tokenizer, text: str):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def _predict_choice_from_logits(tokenizer, next_token_logits: torch.Tensor):
    # Prefer a robust A/B/C/D scoring instead of argmax over whole vocab.
    # Try multiple surface forms because many tokenizers have a dedicated " A" token.
    variants = {
        "A": ["A", " A", "\nA", "\n A"],
        "B": ["B", " B", "\nB", "\n B"],
        "C": ["C", " C", "\nC", "\n C"],
        "D": ["D", " D", "\nD", "\n D"],
    }

    scores = {}
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


def get_prediction_sink_entropy(model, tokenizer, text, device, sink_tokens: int):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 1) Next-token logits + entropy
    next_token_logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(next_token_logits, dim=-1)
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    pred_choice = _predict_choice_from_logits(tokenizer, next_token_logits)

    # 2) Sink mass: attention from last token to first K tokens (default K=1)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions (outputs.attentions is None).")

    # attentions: tuple[num_layers] of [batch, heads, seq, seq]
    attn = torch.stack([a.float() for a in outputs.attentions], dim=0)  # [L,B,H,S,S]
    attn_mean = torch.nanmean(attn, dim=(0, 1, 2))  # [S,S]
    sink_mass = float(torch.nan_to_num(attn_mean[-1, :sink_tokens].sum(), nan=0.0).item())

    return pred_choice, sink_mass, entropy


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--sink_tokens", type=int, default=1, help="How many initial tokens count as sink (K).")
    parser.add_argument("--chat", type=str, default="auto", choices=["auto", "on", "off"], help="Use tokenizer chat template.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to write result json into.")
    args = parser.parse_args()

    print(f"Loading {args.model} with quantization={args.quantization}...")
    
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if args.device != "cpu" else None,
            "attn_implementation": "eager",
            "trust_remote_code": True
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(args.model, **{k: v for k, v in model_kwargs.items() if v is not None})
        if args.device == "cpu":
            model = model.to("cpu")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # MMLU Mapping
    int_to_char = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    results = []
    
    # Load Dataset
    print("Loading Dataset...")
    try:
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except:
         ds = load_dataset("cais/mmlu", "all", split="test")

    if len(ds) > args.samples:
        ds = ds.shuffle(seed=42).select(range(args.samples))
        
    print(f"Running MMLU Accuracy Check on {args.model}...")
    correct_count = 0
    
    for item in tqdm(ds):
        raw_prompt = f"Question: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\nAnswer:"
        prompt = format_prompt(tokenizer, args.model, raw_prompt, chat_mode=args.chat)
        target = int_to_char[item['answer']]
        
        try:
            pred, sink, entropy = get_prediction_sink_entropy(
                model, tokenizer, prompt, args.device, sink_tokens=args.sink_tokens
            )
            
            # Check correctness (starts with target letter)
            is_correct = str(pred).upper().startswith(target)
            
            results.append({
                "correct": is_correct,
                "sink_mass": sink,
                "entropy": entropy,
                "pred": pred,
                "target": target,
                "subject": item.get("subject"),
                # metadata for reproducibility (constant across a run, but stored per-row for backward compatibility)
                "sink_tokens": args.sink_tokens,
                "chat_mode": args.chat,
            })
            if is_correct: correct_count += 1
        except Exception as e:
            print(f"Error on sample: {e}")
            continue
        
    acc = correct_count / len(results) if results else 0
    print(f"Overall Accuracy: {acc:.2%}")
    
    # Save raw results
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "artifacts" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.split('/')[-1]
    filename = out_dir / f"mmlu_accuracy_sink_{safe_name}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    main()
