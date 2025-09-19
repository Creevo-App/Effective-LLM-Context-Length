#!/usr/bin/env python3
"""
Evaluate effective context length of gemini-2.5-flash on AIME 2025-style problems
using two padding strategies:

1) duplicate: duplicate the question until the prompt approaches a target token count
2) junk: fill the prompt with the word "junk" until the target token count, then append question

The script supports loading problems from:
- a Hugging Face dataset (via --hf-dataset-id, with configurable field names), or
- a local JSONL/CSV file (via --data-path, with configurable field names).

Outputs JSONL results with per-run metadata and scores.

Note: This script relies on google-generativeai >= 0.8.0.
Set your API key via environment variable GEMINI_API_KEY (or GOOGLE_API_KEY).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _import_optional_packages() -> Tuple[Any, Any]:
    """Import optional packages lazily to keep startup snappy.

    Returns:
        (genai, datasets)
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "google-generativeai is required. Install with: pip install google-generativeai"
        ) from exc

    try:
        import datasets  # type: ignore
    except Exception:
        datasets = None

    return genai, datasets


def configure_genai(api_key_env: str, model_name: str):
    genai, _ = _import_optional_packages()

    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {api_key_env} or GOOGLE_API_KEY or GEMINI_API_KEY in environment."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return genai, model


def load_hf_dataset(
    hf_dataset_id: str,
    split: str,
    question_field: str,
    answer_field: str,
    datasets_mod: Any,
) -> List[Dict[str, Any]]:
    if datasets_mod is None:
        raise RuntimeError("Hugging Face datasets not installed. pip install datasets")
    ds = datasets_mod.load_dataset(hf_dataset_id, split=split)
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        if question_field not in ex:
            raise KeyError(f"Missing field '{question_field}' in dataset example")
        if answer_field not in ex:
            raise KeyError(f"Missing field '{answer_field}' in dataset example")
        rows.append({
            "question": ex[question_field],
            "answer": ex[answer_field],
        })
    return rows


def load_local_data(
    data_path: str,
    question_field: str,
    answer_field: str,
) -> List[Dict[str, Any]]:
    ext = os.path.splitext(data_path)[1].lower()
    rows: List[Dict[str, Any]] = []
    if ext in {".jsonl", ".json"}:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({
                    "question": rec[question_field],
                    "answer": rec[answer_field],
                })
    elif ext in {".csv"}:
        import csv

        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                rows.append({
                    "question": rec[question_field],
                    "answer": rec[answer_field],
                })
    else:
        raise ValueError("Unsupported data format. Use .jsonl, .json, or .csv")

    return rows


def normalize_aime_answer(ans: Any) -> Optional[int]:
    """Normalize AIME answers to an integer in [0, 999], if possible.

    AIME official answers are integers 0-999. Attempt to extract first such integer.
    """
    if isinstance(ans, int):
        if 0 <= ans <= 999:
            return ans
        return None
    if isinstance(ans, float):
        if float(int(ans)) == ans and 0 <= int(ans) <= 999:
            return int(ans)
        return None
    if not isinstance(ans, str):
        ans = str(ans)

    m = re.search(r"(?<!\d)(\d{1,3})(?!\d)", ans)
    if not m:
        return None
    val = int(m.group(1))
    if 0 <= val <= 999:
        return val
    return None


def instruction_preamble() -> str:
    return (
        "You are solving an AIME problem. Provide only the final answer as an integer on the last line in the form: ANSWER: <int>\n"
        "Do not include units. Do not include any extra commentary after the final answer line.\n"
    )


def format_final_question_block(question: str) -> str:
    return f"\n\nQuestion:\n{question}\n\nRemember: respond with 'ANSWER: <int>' on the last line.\n"


def count_tokens(model: Any, text: str) -> int:
    try:
        # google-generativeai count_tokens returns a dict with 'total_tokens'
        res = model.count_tokens(text)
        if isinstance(res, dict) and "total_tokens" in res:
            return int(res["total_tokens"])  # type: ignore
        # Some SDK versions return an object with .total_tokens
        total = getattr(res, "total_tokens", None)
        if total is not None:
            return int(total)
    except Exception:
        # Fallback: rough approximation by words
        return max(1, len(text.split()))
    return max(1, len(text.split()))


def _grow_padding(base: str, pad_token: str, target_tokens: int, model: Any) -> str:
    """Exponential growth of padding until exceeding target, then return text."""
    text = base
    pad = pad_token
    while count_tokens(model, text) < target_tokens:
        text = text + pad
        pad = pad + pad  # exponential growth
        # Guard to avoid runaway memory when target is huge
        if len(text) > 5_000_000:
            break
    return text


def binary_refine(base: str, extra: str, target_tokens: int, model: Any, max_steps: int = 18) -> str:
    """Binary search add portions of `extra` to approach target token count."""
    low = 0
    high = len(extra)
    best = base
    for _ in range(max_steps):
        mid = (low + high) // 2
        candidate = base + extra[:mid]
        tok = count_tokens(model, candidate)
        if tok <= target_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def build_padded_prompt(
    model: Any,
    question: str,
    method: str,
    target_tokens: int,
    tolerance: int,
) -> Tuple[str, int]:
    """Build prompt approximately hitting target_tokens with the specified method.

    Returns (prompt_text, measured_tokens).
    """
    pre = instruction_preamble()
    final_q = format_final_question_block(question)

    if method == "duplicate":
        # Start with preamble; then grow by duplicating the question block; ensure the final question is present at the end
        base = pre
        repeated = ("\n\n" + question)
        grown = _grow_padding(base, repeated, target_tokens - count_tokens(model, final_q), model)
        # Refine with a slice of repeated
        grown = binary_refine(grown, repeated * 2, target_tokens - count_tokens(model, final_q), model)
        prompt = grown + final_q
    elif method == "junk":
        base = pre
        repeated = " junk"  # small unit
        # grow many junks; account for the final question at the end
        grown = _grow_padding(base, repeated, target_tokens - count_tokens(model, final_q), model)
        # refine
        grown = binary_refine(grown, "junk " * 1024, target_tokens - count_tokens(model, final_q), model)
        prompt = grown + final_q
    else:
        raise ValueError("method must be 'duplicate' or 'junk'")

    measured = count_tokens(model, prompt)
    # If we are under target - tolerance, append some small filler
    if measured + 4 <= target_tokens - tolerance:
        small = " junk"
        while measured + 4 <= target_tokens - tolerance:
            prompt += small
            measured = count_tokens(model, prompt)
            if len(prompt) > 5_000_000:
                break
    return prompt, measured


def call_model(
    model: Any,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[str, float, Optional[str]]:
    start = time.time()
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(max_output_tokens),
            },
        )
        latency = time.time() - start
        # SDK returns a response with .text
        text = getattr(resp, "text", None)
        if text is None:
            text = ""  # fallback
        return str(text), latency, None
    except Exception as exc:
        latency = time.time() - start
        return "", latency, str(exc)


@dataclass
class EvalRecord:
    problem_index: int
    method: str
    target_tokens: int
    measured_tokens: int
    model_name: str
    latency_s: float
    error: Optional[str]
    gold_answer: Optional[int]
    predicted_answer: Optional[int]
    correct: Optional[bool]
    raw_response: str


def eval_once(
    model: Any,
    model_name: str,
    question: str,
    gold: Optional[int],
    method: str,
    target_tokens: int,
    tolerance: int,
    max_output_tokens: int,
    temperature: float,
    problem_index: int,
) -> EvalRecord:
    prompt, measured = build_padded_prompt(model, question, method, target_tokens, tolerance)
    raw, latency, err = call_model(model, prompt, max_output_tokens, temperature)
    pred = normalize_aime_answer(raw)
    correct = (pred == gold) if (gold is not None and pred is not None) else None
    return EvalRecord(
        problem_index=problem_index,
        method=method,
        target_tokens=target_tokens,
        measured_tokens=measured,
        model_name=model_name,
        latency_s=latency,
        error=err,
        gold_answer=gold,
        predicted_answer=pred,
        correct=correct,
        raw_response=raw,
    )


def save_jsonl(path: str, records: Iterable[EvalRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def parse_lengths(arg: str) -> List[int]:
    if os.path.exists(arg):
        # file containing one length per line
        vals: List[int] = []
        with open(arg, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals.append(int(line))
        return vals
    # comma-separated list
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="AIME effective context evaluation on Gemini 2.5 Flash")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-dataset-id", type=str, help="Hugging Face dataset id (e.g., lighteval/AIME_2025)")
    src.add_argument("--data-path", type=str, help="Local JSONL/CSV with fields")

    parser.add_argument("--split", type=str, default="test", help="Dataset split if using HF")
    parser.add_argument("--question-field", type=str, default="question", help="Question field name")
    parser.add_argument("--answer-field", type=str, default="answer", help="Answer field name")

    parser.add_argument("--lengths", type=str, required=True, help="Comma list or file of target token lengths")
    parser.add_argument("--method", type=str, default="both", choices=["duplicate", "junk", "both"], help="Padding method")
    parser.add_argument("--tolerance", type=int, default=32, help="Allowed token deviation from target")

    parser.add_argument("--limit", type=int, default=15, help="Max number of problems to run")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle problems before limiting")

    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY", help="Env var for API key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="aime_gemini_results.jsonl")

    args = parser.parse_args()

    random.seed(args.seed)

    lengths = parse_lengths(args.lengths)
    genai, datasets_mod = _import_optional_packages()
    genai, model = configure_genai(args.api_key_env, args.model)

    # Load data
    if args.hf_dataset_id:
        rows = load_hf_dataset(args.hf_dataset_id, args.split, args.question_field, args.answer_field, datasets_mod)
    else:
        rows = load_local_data(args.data_path, args.question_field, args.answer_field)

    if args.shuffle:
        random.shuffle(rows)
    rows = rows[: max(1, int(args.limit))]

    methods: List[str]
    if args.method == "both":
        methods = ["duplicate", "junk"]
    else:
        methods = [args.method]

    # Run
    all_records: List[EvalRecord] = []
    for idx, row in enumerate(rows):
        question = str(row["question"]).strip()
        gold = normalize_aime_answer(row.get("answer"))
        for target in lengths:
            for method in methods:
                rec = eval_once(
                    model=model,
                    model_name=args.model,
                    question=question,
                    gold=gold,
                    method=method,
                    target_tokens=int(target),
                    tolerance=int(args.tolerance),
                    max_output_tokens=int(args.max_output_tokens),
                    temperature=float(args.temperature),
                    problem_index=idx,
                )
                all_records.append(rec)
                # Stream to disk to avoid losing progress
                save_jsonl(args.output, [rec])
                # Small delay to be gentle on rate limits
                time.sleep(0.2)

    # Quick aggregate print
    total_with_gold = [r for r in all_records if r.gold_answer is not None]
    if total_with_gold:
        acc = sum(1 for r in total_with_gold if r.correct) / float(len(total_with_gold))
        print(f"Ran {len(all_records)} evals across {len(rows)} problems; accuracy (where gold present): {acc:.3f}")
    else:
        print(f"Ran {len(all_records)} evals across {len(rows)} problems; no gold answers to score.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

