#!/usr/bin/env python3
"""Evaluate Gemini 2.5 Pro context robustness on the AIME 2025 benchmark.

This script approximates the effective context length of the ``gemini-2.5-pro``
model on the AIME 2025 benchmark in two stress-test settings:

1. ``duplicate`` mode repeats the full question until the prompt reaches a
   target context length.
2. ``junk`` mode fills the prefix of the prompt with the word ``"junk"`` until
   the prompt reaches a target context length.

For each target context length, the script submits the resulting prompt to the
Gemini API and measures accuracy against the gold answers.

Example usage::

    python scripts/gemini_aime_context.py \
        --api-key $GEMINI_API_KEY \
        --context-lengths 4096 8192 16384 \
        --num-questions 15 \
        --output results.json

The script expects the AIME 2025 dataset to be available either via Hugging
Face Datasets (``lighteval/aime`` with config ``2025``) or as a local JSON/JSONL
file provided via ``--dataset-path``. The local file must contain objects with
``question`` (or ``problem``/``prompt``) and ``answer`` fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Approximate number of characters per token for planning prompt lengths.
DEFAULT_CHARS_PER_TOKEN = 4
# Default context lengths (in tokens) to probe.
DEFAULT_CONTEXT_LENGTHS = (4096, 8192, 16384, 32768, 65536)
# Fields that may contain the question text in the dataset rows.
QUESTION_FIELD_CANDIDATES = ("question", "problem", "prompt", "input", "text")
# Fields that may contain the answer text in the dataset rows.
ANSWER_FIELD_CANDIDATES = ("answer", "target", "solution", "output")


@dataclass
class PromptMetadata:
    """Tracks metadata associated with a constructed prompt."""

    mode: str
    target_tokens: int
    prompt_length_chars: int
    approx_prompt_tokens: float
    filler_length_chars: int
    duplicate_count: Optional[int] = None
    junk_token_count: Optional[int] = None
    approx_char_target: Optional[int] = None


@dataclass
class QuestionRecord:
    """Stores a single question/answer pair from the dataset."""

    question: str
    answer: str
    metadata: Dict[str, str] = field(default_factory=dict)


def configure_logging(verbosity: int) -> None:
    """Configure module-level logging according to the desired verbosity."""

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset_from_file(path: str) -> List[QuestionRecord]:
    """Load AIME questions from a local JSON or JSONL file."""

    logging.info("Loading dataset from file: %s", path)
    records: List[QuestionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Dataset file '{path}' is empty")
        if content[0] == "[":
            data = json.loads(content)
            iterable: Iterable[dict] = data
        else:
            iterable = (json.loads(line) for line in content.splitlines())
        for idx, row in enumerate(iterable):
            question = extract_field(row, QUESTION_FIELD_CANDIDATES)
            answer = extract_field(row, ANSWER_FIELD_CANDIDATES)
            if question is None or answer is None:
                raise KeyError(
                    "Each dataset row must contain question and answer fields. "
                    f"Row {idx} keys: {sorted(row.keys())}"
                )
            records.append(
                QuestionRecord(
                    question=str(question).strip(),
                    answer=str(answer).strip(),
                    metadata={"source_index": str(idx)},
                )
            )
    if not records:
        raise ValueError(f"No records loaded from '{path}'")
    logging.info("Loaded %d questions from local file", len(records))
    return records


def load_dataset_from_hub(
    name: str,
    config: str,
    split: str,
    *,
    hf_token: Optional[str] = None,
) -> List[QuestionRecord]:
    """Load the dataset from Hugging Face Hub using ``datasets``."""

    logging.info(
        "Loading dataset '%s' (config=%s, split=%s) from Hugging Face", name, config, split
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "datasets library is required to load the benchmark from the hub. "
            "Install it with `pip install datasets`."
        ) from exc

    # Allow passing a token for gated/private datasets via env or CLI.
    # Prefer HF_TOKEN or HUGGINGFACE_TOKEN if present. CLI can override via --hf-token.
    # Prefer explicit token passed from CLI; else check environment variables.
    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    load_kwargs = {"split": split}
    if hf_token:
        # Newer datasets versions accept `token`; older accept `use_auth_token`.
        load_kwargs["token"] = hf_token
        load_kwargs["use_auth_token"] = hf_token

    try:
        dataset = load_dataset(name, config, **load_kwargs)
    except TypeError:
        # Fallback if only one of the parameters is supported
        try:
            dataset = load_dataset(name, config, split=split, token=hf_token)
        except TypeError:
            dataset = load_dataset(name, config, split=split, use_auth_token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset from Hugging Face. "
            "If the dataset is gated/private, set HF_TOKEN or HUGGINGFACE_TOKEN, "
            "or pass a local file via --dataset-path. Original error: " + str(exc)
        )
    records: List[QuestionRecord] = []
    for idx, row in enumerate(dataset):
        question = extract_field(row, QUESTION_FIELD_CANDIDATES)
        answer = extract_field(row, ANSWER_FIELD_CANDIDATES)
        if question is None or answer is None:
            raise KeyError(
                "Dataset row does not contain recognizable question/answer fields. "
                f"Row {idx} keys: {sorted(row.keys())}"
            )
        records.append(
            QuestionRecord(
                question=str(question).strip(),
                answer=str(answer).strip(),
                metadata={"split_index": str(idx)},
            )
        )
    if not records:
        raise ValueError(f"Dataset '{name}' returned no rows")
    logging.info("Loaded %d questions from Hugging Face dataset", len(records))
    return records


def extract_field(mapping: dict, candidates: Sequence[str]) -> Optional[str]:
    """Return the value of the first existing key from ``candidates``."""

    for key in candidates:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def build_prompt(
    question: str,
    mode: str,
    target_tokens: int,
    *,
    approx_chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> Tuple[str, PromptMetadata]:
    """Construct a prompt containing the question surrounded by filler text."""

    instructions = (
        "You will receive filler text meant to stress your context window. "
        "Ignore the filler and answer only the final AIME 2025 math question. "
        "Respond with a single 3-digit integer representing the final answer."
    )
    sanitized_question = question.strip()
    approx_char_target = int(target_tokens * approx_chars_per_token)

    filler_text = ""
    duplicate_count: Optional[int] = None
    junk_token_count: Optional[int] = None

    if mode == "duplicate":
        # Duplicate the question repeatedly until the filler roughly matches the target length.
        pieces: List[str] = []
        filler_len = 0
        while filler_len < approx_char_target:
            pieces.append(sanitized_question)
            filler_len += len(sanitized_question) + 2  # Account for separators.
        filler_text = "\n\n".join(pieces)
        duplicate_count = len(pieces)
    elif mode == "junk":
        # Fill with the word "junk" repeated until reaching the target length.
        junk_unit = "junk "
        desired = max(0, approx_char_target - len(instructions) - len(sanitized_question))
        repeats = (desired // len(junk_unit)) + 2
        filler_text = (junk_unit * repeats)[:desired]
        junk_token_count = filler_text.count("junk")
    else:  # pragma: no cover - protected by argument parser
        raise ValueError(f"Unsupported mode: {mode}")

    prompt = (
        f"{instructions}\n\n"
        f"{filler_text}\n\n"
        f"Final question (answer this one only):\n{sanitized_question}\n\n"
        "Provide only the numeric answer without explanation."
    )

    prompt_length_chars = len(prompt)
    approx_prompt_tokens = prompt_length_chars / float(approx_chars_per_token)

    metadata = PromptMetadata(
        mode=mode,
        target_tokens=target_tokens,
        prompt_length_chars=prompt_length_chars,
        approx_prompt_tokens=approx_prompt_tokens,
        filler_length_chars=len(filler_text),
        duplicate_count=duplicate_count,
        junk_token_count=junk_token_count,
        approx_char_target=approx_char_target,
    )
    return prompt, metadata


def parse_model_answer(text: str) -> Optional[str]:
    """Extract a potential AIME answer (3-digit integer) from the model output."""

    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    value = match.group(0)
    if value.startswith("-"):
        return value  # allow negative for diagnostics even if incorrect
    # Normalize to at most three digits by stripping leading zeros.
    normalized = value.lstrip("0")
    return normalized if normalized else "0"


def normalize_gold_answer(answer: str) -> str:
    """Normalize gold answers for consistent comparison."""

    cleaned = re.sub(r"[^0-9-]+", "", answer)
    if not cleaned:
        raise ValueError(f"Unable to parse gold answer from '{answer}'")
    normalized = cleaned.lstrip("0")
    return normalized if normalized else "0"


def call_gemini(
    model,
    prompt: str,
    *,
    max_output_tokens: int,
    temperature: float,
    top_p: Optional[float] = None,
) -> str:
    """Call the Gemini model and return the text response."""

    logging.debug("Submitting prompt of %d characters", len(prompt))
    generation_config = {"max_output_tokens": max_output_tokens, "temperature": temperature}
    if top_p is not None:
        generation_config["top_p"] = top_p
    response = model.generate_content(prompt, generation_config=generation_config)
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    # Fall back to aggregating candidates.
    if hasattr(response, "candidates"):
        texts = []
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
        if texts:
            return "\n".join(texts).strip()
    raise RuntimeError("Gemini response did not contain any text content")


def initialize_gemini_model(api_key: str, model_name: str):
    """Create a Gemini GenerativeModel instance using the supplied API key."""

    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "google-generativeai package is required. Install it with "
            "`pip install google-generativeai`."
        ) from exc

    genai.configure(api_key=api_key)
    logging.info("Initialized Gemini client for model '%s'", model_name)
    return genai.GenerativeModel(model_name)


def evaluate_context_lengths(
    records: Sequence[QuestionRecord],
    model,
    context_lengths: Sequence[int],
    modes: Sequence[str],
    *,
    sleep: float,
    max_output_tokens: int,
    temperature: float,
    top_p: Optional[float],
) -> Dict[str, List[dict]]:
    """Evaluate the model across context lengths and filler modes."""

    results: Dict[str, List[dict]] = {mode: [] for mode in modes}
    total = len(records)
    logging.info("Evaluating %d questions across %d modes", total, len(modes))

    normalized_answers = [normalize_gold_answer(r.answer) for r in records]

    for mode in modes:
        for target_tokens in context_lengths:
            logging.info("Mode=%s target_tokens=%d", mode, target_tokens)
            mode_results = {
                "mode": mode,
                "target_tokens": target_tokens,
                "per_question": [],
                "num_questions": total,
            }
            correct = 0
            start_time = time.time()
            for idx, record in enumerate(records):
                prompt, prompt_meta = build_prompt(
                    record.question, mode, target_tokens
                )
                try:
                    response_text = call_gemini(
                        model,
                        prompt,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except Exception as exc:  # pragma: no cover - runtime reporting
                    logging.error(
                        "Model call failed for index %d (mode=%s tokens=%d): %s",
                        idx,
                        mode,
                        target_tokens,
                        exc,
                    )
                    response_text = ""
                prediction = parse_model_answer(response_text)
                gold = normalized_answers[idx]
                is_correct = prediction == gold
                if is_correct:
                    correct += 1
                mode_results["per_question"].append(
                    {
                        "index": idx,
                        "question_metadata": record.metadata,
                        "prompt_metadata": prompt_meta.__dict__,
                        "response": response_text,
                        "prediction": prediction,
                        "gold": gold,
                        "correct": is_correct,
                    }
                )
                if sleep > 0:
                    time.sleep(sleep)
            duration = time.time() - start_time
            accuracy = correct / float(total) if total else math.nan
            mode_results.update(
                {
                    "correct": correct,
                    "accuracy": accuracy,
                    "duration_sec": duration,
                }
            )
            results[mode].append(mode_results)
            logging.info(
                "Completed mode=%s tokens=%d accuracy=%.3f (%.1fs)",
                mode,
                target_tokens,
                accuracy,
                duration,
            )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key. Defaults to the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        default="models/gemini-2.5-pro",
        help="Gemini model name to query (default: models/gemini-2.5-pro).",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help="Sequence of target context lengths (in tokens) to probe.",
    )
    parser.add_argument(
        "--mode",
        choices=["duplicate", "junk", "both"],
        default="both",
        help="Which filler mode(s) to evaluate.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Optional limit on the number of AIME questions to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling a subset of questions.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional local JSON/JSONL file with the AIME dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lighteval/aime",
        help="Hugging Face dataset name (default: lighteval/aime).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="2025",
        help="Hugging Face dataset configuration (default: 2025).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to use when loading from Hugging Face (default: test).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help=(
            "Optional Hugging Face access token for gated/private datasets. "
            "Defaults to HF_TOKEN or HUGGINGFACE_TOKEN environment variables if set."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write raw results as JSON.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) between consecutive API calls.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Gemini responses.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling parameter.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug logging).",
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if not args.api_key:
        parser.error("Gemini API key must be provided via --api-key or GEMINI_API_KEY")

    if args.mode == "both":
        modes = ["duplicate", "junk"]
    else:
        modes = [args.mode]

    if args.dataset_path:
        records = load_dataset_from_file(args.dataset_path)
    else:
        records = load_dataset_from_hub(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            hf_token=args.hf_token,
        )

    if args.num_questions is not None:
        if args.num_questions <= 0:
            parser.error("--num-questions must be positive when provided")
        if args.num_questions < len(records):
            rng = random.Random(args.seed)
            rng.shuffle(records)
            records = records[: args.num_questions]
            logging.info("Sampled %d questions (seed=%d)", len(records), args.seed)

    model = initialize_gemini_model(args.api_key, args.model)

    results = evaluate_context_lengths(
        records,
        model,
        args.context_lengths,
        modes,
        sleep=args.sleep,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Pretty print a summary table to stdout.
    print("Mode\tTokens\tAccuracy\tCorrect/Total\tAvg Prompt Tokens")
    for mode, mode_results in results.items():
        for entry in mode_results:
            avg_tokens = sum(
                item["prompt_metadata"]["approx_prompt_tokens"]
                for item in entry["per_question"]
            ) / float(entry["num_questions"])
            print(
                f"{mode}\t{entry['target_tokens']}\t{entry['accuracy']:.3f}\t"
                f"{entry['correct']}/{entry['num_questions']}\t{avg_tokens:.0f}"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info("Wrote detailed results to %s", args.output)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
