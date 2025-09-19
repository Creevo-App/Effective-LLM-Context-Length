#!/usr/bin/env python3
"""
Evaluate effective context length for gemini-2.5-flash on AIME-style questions.

Two context scaling methods:
- duplicate: duplicate the question until reaching target length
- junk: pad with the word "junk" until reaching target length

Rate limited to <= 10 requests per minute.

Usage examples:
  python aime_context_eval.py \
    --input /path/to/aime2025.csv \
    --lengths 2000 8000 32000 \
    --units chars \
    --methods duplicate junk \
    --max-examples 30 \
    --output results.csv

  # Optional: load from Hugging Face if available
  python aime_context_eval.py --hf-dataset lighteval/AIME_2025 --split test --output results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


DEFAULT_API_KEY = "AIzaSyAxJNYd-_WjwtteS44gzTW5PUnCPMQj7Aw"


def _eprint(message: str) -> None:
    print(message, file=sys.stderr)


class RateLimiter:
    """Simple rate limiter for max N requests per minute.

    Enforces a minimum spacing between calls. Also handles 429 backoff.
    """

    def __init__(self, max_per_minute: int = 10) -> None:
        if max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive")
        self.max_per_minute = max_per_minute
        # Conservative fixed spacing
        self.min_interval_seconds = 60.0 / float(max_per_minute)
        self._last_call_ts: float = 0.0

    def wait_turn(self) -> None:
        now = time.time()
        next_allowed = self._last_call_ts + self.min_interval_seconds
        if now < next_allowed:
            time.sleep(next_allowed - now)
        # Do not set last until after the API call completes

    def mark_call(self) -> None:
        self._last_call_ts = time.time()


def load_questions_from_csv(path: str) -> List[Tuple[str, Optional[str]]]:
    questions: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect columns: question, answer (answer optional)
        for row in reader:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            ans = (row.get("answer") or "").strip() or None
            questions.append((q, ans))
    return questions


def load_questions_from_jsonl(path: str) -> List[Tuple[str, Optional[str]]]:
    questions: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            if not q:
                continue
            ans_raw = obj.get("answer")
            ans = (ans_raw.strip() if isinstance(ans_raw, str) else None)
            questions.append((q, ans))
    return questions


def load_questions_from_hf(dataset_name: str, split: str) -> List[Tuple[str, Optional[str]]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "datasets library not installed. Install with `pip install datasets`."
        ) from e
    ds = load_dataset(dataset_name, split=split)
    questions: List[Tuple[str, Optional[str]]] = []
    # Try common field names
    candidate_question_fields = ["question", "problem", "prompt", "input"]
    candidate_answer_fields = ["answer", "solution", "target", "output"]
    q_field = next((f for f in candidate_question_fields if f in ds.features), None)
    if q_field is None:
        raise RuntimeError(
            f"Could not find a question field in dataset {dataset_name}:{split}."
        )
    a_field = next((f for f in candidate_answer_fields if f in ds.features), None)
    for ex in ds:
        q = (ex.get(q_field) or "").strip()
        if not q:
            continue
        ans_val = ex.get(a_field) if a_field is not None else None
        ans = (ans_val.strip() if isinstance(ans_val, str) else None)
        questions.append((q, ans))
    return questions


def ensure_length_chars(text: str, target_chars: int) -> str:
    if target_chars <= 0:
        return text
    if len(text) >= target_chars:
        return text[:target_chars]
    # Pad spaces to reach target length exactly
    return (text + (" " * (target_chars - len(text))))[:target_chars]


def ensure_length_words(text: str, target_words: int) -> str:
    if target_words <= 0:
        return text
    words = text.split()
    if len(words) >= target_words:
        return " ".join(words[:target_words])
    # Pad with empty words (spaces) to reach exact count is not meaningful; just return as is
    return text


def build_duplicate_context(question: str, target: int, units: str) -> str:
    if units == "chars":
        buf = question
        while len(buf) < target:
            buf += "\n\n" + question
        return ensure_length_chars(buf, target)
    if units == "words":
        q_words = question.split()
        if not q_words:
            return ""
        buf_words: List[str] = []
        while len(buf_words) < target:
            buf_words.extend(q_words)
        return ensure_length_words(" ".join(buf_words), target)
    raise ValueError("units must be 'chars' or 'words'")


def build_junk_context(question: str, target: int, units: str) -> str:
    header = question.strip() + "\n\n"
    if units == "chars":
        # Fill with 'junk ' until reaching target
        buf = header
        filler = "junk "
        while len(buf) < target:
            buf += filler
        return ensure_length_chars(buf, target)
    if units == "words":
        header_words = header.split()
        buf_words = list(header_words)
        while len(buf_words) < target:
            buf_words.append("junk")
        return ensure_length_words(" ".join(buf_words), target)
    raise ValueError("units must be 'chars' or 'words'")


def call_gemini(prompt: str, model_name: str, api_key: str, temperature: float, top_p: float, max_output_tokens: int) -> Tuple[bool, str]:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "google-generativeai is required. Install with `pip install google-generativeai`."
        ) from e

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        t0 = time.time()
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
            },
        )
        latency_s = time.time() - t0
        text = getattr(response, "text", None)
        if not text:
            # Some SDK versions expose candidates
            try:
                candidates = response.candidates or []
                text = candidates[0].content.parts[0].text if candidates else ""
            except Exception:
                text = ""
        return True, json.dumps({
            "latency_seconds": latency_s,
            "text": text,
        })
    except Exception as e:
        return False, str(e)


@dataclass
class EvalConfig:
    input_path: Optional[str]
    hf_dataset: Optional[str]
    hf_split: str
    output_path: str
    methods: List[str]
    lengths: List[int]
    units: str
    max_examples: Optional[int]
    api_key: str
    model_name: str
    temperature: float
    top_p: float
    max_output_tokens: int
    max_requests_per_minute: int


def parse_args(argv: Optional[Iterable[str]] = None) -> EvalConfig:
    ap = argparse.ArgumentParser(description="Test effective context length for Gemini on AIME questions")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", dest="input_path", type=str, default=None,
                     help="Path to CSV (question,answer) or JSONL with {'question','answer'}")
    src.add_argument("--hf-dataset", dest="hf_dataset", type=str, default=None,
                     help="Hugging Face dataset name, e.g., lighteval/AIME_2025")
    ap.add_argument("--split", dest="hf_split", type=str, default="test", help="HF split (default: test)")
    ap.add_argument("--output", dest="output_path", type=str, required=True, help="Output CSV path")
    ap.add_argument("--methods", nargs="+", default=["duplicate", "junk"], choices=["duplicate", "junk"],
                    help="Context length construction methods")
    ap.add_argument("--lengths", nargs="+", type=int, required=True,
                    help="Target context lengths (integers)")
    ap.add_argument("--units", choices=["chars", "words"], default="chars", help="Units for lengths")
    ap.add_argument("--max-examples", type=int, default=None, help="Limit number of questions")
    ap.add_argument("--api-key", type=str, default=os.environ.get("GENAI_API_KEY", DEFAULT_API_KEY),
                    help="Google Generative AI API key (env GENAI_API_KEY takes precedence if set)")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--max-requests-per-minute", type=int, default=10,
                    help="Hard cap on requests per minute (default 10)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    return EvalConfig(
        input_path=args.input_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        output_path=args.output_path,
        methods=args.methods,
        lengths=args.lengths,
        units=args.units,
        max_examples=args.max_examples,
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        max_requests_per_minute=args.max_requests_per_minute,
    )


def load_questions(cfg: EvalConfig) -> List[Tuple[str, Optional[str]]]:
    if cfg.input_path:
        if not os.path.exists(cfg.input_path):
            raise FileNotFoundError(cfg.input_path)
        lower = cfg.input_path.lower()
        if lower.endswith(".csv"):
            return load_questions_from_csv(cfg.input_path)
        if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
            return load_questions_from_jsonl(cfg.input_path)
        raise RuntimeError("Unsupported input format. Use .csv or .jsonl")
    if cfg.hf_dataset:
        return load_questions_from_hf(cfg.hf_dataset, cfg.hf_split)
    raise RuntimeError("Must provide --input or --hf-dataset")


def build_context(question: str, target: int, method: str, units: str) -> str:
    if method == "duplicate":
        return build_duplicate_context(question, target, units)
    if method == "junk":
        return build_junk_context(question, target, units)
    raise ValueError(f"Unknown method: {method}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)

    questions = load_questions(cfg)
    if cfg.max_examples is not None:
        questions = questions[: cfg.max_examples]
    if not questions:
        _eprint("No questions loaded.")
        return 1

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    rate = RateLimiter(cfg.max_requests_per_minute)

    # Prepare CSV output
    fieldnames = [
        "index",
        "method",
        "units",
        "target_length",
        "actual_prompt_chars",
        "actual_prompt_words",
        "latency_seconds",
        "success",
        "error",
        "response_text",
    ]
    with open(cfg.output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (question, _answer) in enumerate(questions):
            for length in cfg.lengths:
                for method in cfg.methods:
                    prompt = build_context(question, length, method, cfg.units)
                    # Respect RPM limit
                    rate.wait_turn()
                    ok, payload = call_gemini(
                        prompt=prompt,
                        model_name=cfg.model_name,
                        api_key=cfg.api_key,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_output_tokens=cfg.max_output_tokens,
                    )
                    rate.mark_call()

                    latency_seconds: Optional[float] = None
                    response_text: str = ""
                    error_msg: str = ""
                    if ok:
                        try:
                            obj = json.loads(payload)
                            latency_seconds = float(obj.get("latency_seconds", 0.0))
                            response_text = obj.get("text", "")
                        except Exception:
                            response_text = payload
                    else:
                        error_msg = payload

                    writer.writerow({
                        "index": idx,
                        "method": method,
                        "units": cfg.units,
                        "target_length": length,
                        "actual_prompt_chars": len(prompt),
                        "actual_prompt_words": len(prompt.split()),
                        "latency_seconds": ("" if latency_seconds is None else f"{latency_seconds:.3f}"),
                        "success": ok,
                        "error": error_msg,
                        "response_text": response_text,
                    })
                    out_f.flush()

                    # If we hit an explicit 429 in the error message, back off a bit more
                    if not ok and ("429" in error_msg or "rate" in error_msg.lower()):
                        time.sleep(10.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

