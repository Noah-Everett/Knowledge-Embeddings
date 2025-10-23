#!/usr/bin/env python3
"""
Score arXiv papers (title + abstract) with a local Ollama model.

Reads a JSONL file with objects containing at least {"id", "title", "abstract"}
and writes a JSONL file with an object per input id containing model-generated
scores across several criteria plus a brief rationale.

Requirements:
  - Ollama running locally (default: http://localhost:11434)
  - Python packages: requests, tqdm (already in requirements.txt)

Example:
  ./score_papers_with_ollama.py \
    --input arxiv-2007-2019-hepex.jsonl \
    --output hepex-ollama-scores.jsonl \
    --model llama3.1:8b \
    --limit 1000 --skip-existing

The output JSONL line schema (one per paper id):
{
  "id": "oai:arXiv.org:XXXX.YYYYY",
  "title": "...",
  "scores": {
    "progress": 1-5,
    "creativity": 1-5,
    "novelty": 1-5,
    "technical_rigor": 1-5,
    "clarity": 1-5,
    "potential_impact": 1-5,
    "overall": 1-5
  },
  "rationale": "1-3 sentence justification"
}
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional

import requests
import importlib.util as _importlib_util

# Optional tqdm progress bar without hard dependency for static analyzers
_tqdm_spec = _importlib_util.find_spec("tqdm")
if _tqdm_spec is not None:
    from tqdm import tqdm as tqdm  # type: ignore
else:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Scoring rubric (1-5) used in the prompt
CATEGORIES = [
    ("progress", "Contribution to advancing the field; tangible step forward."),
    ("creativity", "Originality of ideas, approaches, or problem framing."),
    ("novelty", "Newness relative to existing literature and prior art."),
    ("technical_rigor", "Methodological soundness, correctness, and thoroughness."),
    ("clarity", "Clarity of problem statement, methods, and results."),
    ("potential_impact", "Potential to influence future research or applications."),
]


def build_prompt(title: str, abstract: str) -> str:
    rubric_lines = []
    for key, desc in CATEGORIES:
        rubric_lines.append(f"- {key}: {desc}")
    rubric = "\n".join(rubric_lines)

    prompt = f"""
You are a careful, quantitative reviewer. Read the paper metadata and score it on a 1-5 integer scale for each rubric below, then produce STRICT JSON only.

Rubric (1 = very low, 3 = medium, 5 = very high):
{rubric}

Instructions:
- Consider only the information provided (title and abstract).
- Use integers 1, 2, 3, 4, or 5 only.
- Provide a brief, single-paragraph rationale.
- Respond with ONLY a single JSON object with this exact structure and keys:
  {{
    "scores": {{
      "progress": int,
      "creativity": int,
      "novelty": int,
      "technical_rigor": int,
      "clarity": int,
      "potential_impact": int,
      "overall": int
    }},
    "rationale": str
  }}

Paper:
Title: {title.strip()}
Abstract: {abstract.strip()}
""".strip()
    return prompt


def ensure_json(s: str) -> Optional[Dict[str, Any]]:
    """Best-effort to parse a JSON object from a string response.

    Some models may include stray text; this attempts to recover a JSON object
    by trimming to the outermost braces if direct parsing fails.
    """
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to find first '{' and last '}'
    try:
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        return None
    return None


def call_ollama(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    retries: int = 3,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call Ollama /api/generate with JSON formatting requested.

    Returns parsed JSON object from the model's response under the assumption
    that we asked for strict JSON output.
    """
    url = base_url.rstrip('/') + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        # Ask Ollama/runtime to enforce JSON output if supported by the model
        "format": "json",
        "stream": False,
        # Conservative context to fit long abstracts; adjust as needed
        "options": {"num_ctx": 8192},
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # For /api/generate, the response text is in data["response"].
                raw = data.get("response", "")
                obj = ensure_json(raw)
                if obj is None:
                    raise ValueError("Model response was not valid JSON")
                return obj
            # Retry on transient errors
            if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 10))
                continue
            # Non-retryable: raise
            resp.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last_exc}")


def normalize_scores(obj: Dict[str, Any]) -> Dict[str, int]:
    """Coerce scores to ints in [1,5] and ensure required keys exist."""
    scores = obj.get("scores", {}) if isinstance(obj, dict) else {}
    def coerce(v: Any) -> int:
        try:
            x = int(round(float(v)))
        except Exception:
            x = 3
        return max(1, min(5, x))

    out = {}
    for key, _desc in CATEGORIES:
        out[key] = coerce(scores.get(key, 3))
    # overall: if provided, coerce; else average round
    if "overall" in scores:
        out_overall = coerce(scores["overall"])
    else:
        vals = list(out.values())
        out_overall = coerce(sum(vals) / len(vals) if vals else 3)
    out["overall"] = out_overall
    return out


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def load_processed_ids(path: str) -> set:
    seen = set()
    if not os.path.exists(path):
        return seen
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    pid = obj.get("id")
                    if pid:
                        seen.add(pid)
                except Exception:
                    continue
    except Exception:
        pass
    return seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score papers with a local Ollama model")
    p.add_argument("--input", "-i", type=str, default="arxiv-2007-2019-hepex.jsonl",
                   help="Path to input JSONL file with fields id, title, abstract")
    p.add_argument("--output", "-o", type=str, default="ollama-scores.jsonl",
                   help="Path to output JSONL file to write scores")
    p.add_argument("--model", "-m", type=str, default="llama3.1:8b",
                   help="Ollama model name (e.g., 'llama3.1:8b', 'llama3.2:3b', 'qwen2.5:7b')")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, stop after scoring this many records")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip records whose id already exists in the output file")
    p.add_argument("--start", type=int, default=0,
                   help="Start from this 0-based index (useful for chunked runs)")
    p.add_argument("--end", type=int, default=0,
                   help="End at this 0-based index (exclusive); 0 means process to end")
    p.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL,
                   help="Base URL for Ollama (default: http://localhost:11434)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine progress skip set if requested
    seen_ids = load_processed_ids(args.output) if args.skip_existing else set()

    # Prepare output file handle (append mode to support resuming)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fout = open(args.output, "a", encoding="utf-8")

    # Iterate inputs with optional slicing
    it = iter_jsonl(args.input)
    start_i = max(0, int(args.start or 0))
    end_i = int(args.end or 0)

    processed = 0
    total_seen = 0
    failed = 0

    # We don't know the total count cheaply without scanning; tqdm will be unbounded
    for idx, rec in enumerate(it):
        if idx < start_i:
            continue
        if end_i and idx >= end_i:
            break

        pid = rec.get("id") or rec.get("arxiv_id") or rec.get("paper_id")
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or rec.get("summary") or ""

        total_seen += 1

        if not pid or not title or not abstract:
            # Skip records without required fields
            continue

        if seen_ids and pid in seen_ids:
            continue

        prompt = build_prompt(title, abstract)
        try:
            obj = call_ollama(
                model=args.model,
                prompt=prompt,
                base_url=args.ollama_url,
            )
            scores = normalize_scores(obj)
            rationale = obj.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                rationale = ""  # keep empty if model omitted

            out_obj = {
                "id": pid,
                "title": title,
                "scores": scores,
                "rationale": rationale.strip(),
                "model": args.model,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1
        except Exception as e:
            failed += 1
            # Log minimal error and continue
            sys.stderr.write(f"[warn] failed on {pid}: {e}\n")

        if args.limit and processed >= args.limit:
            break

    fout.close()

    print(
        json.dumps(
            {
                "processed": processed,
                "skipped_existing": len(seen_ids) if seen_ids else 0,
                "failed": failed,
                "seen": total_seen,
                "start": start_i,
                "end": end_i,
                "output": os.path.abspath(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
