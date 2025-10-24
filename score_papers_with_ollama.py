#!/usr/bin/env python3
"""
Score arXiv papers (title + abstract) with a local Ollama model.

Optimized for speed on Apple Silicon (M1 Pro MBP):
- Uses the official `ollama` Python client (low overhead) and one-paper-per-call.
- Supports parallel jobs via a thread pool (configurable with --workers).
- Avoids extra passes over data and keeps overhead minimal.

Reads a JSONL file containing at least {"id", "title", "abstract"} and writes a
JSONL file with an object per input id containing model-generated scores across
several criteria plus a brief rationale.

Example (lean only):
    ./score_papers_with_ollama.py \
        --input arxiv-2007-2019-hepex.jsonl \
        --output hepex-ollama-scores.jsonl \
        --model llama3.1:8b \
        --workers 2 --limit 1000 --skip-existing

Output JSONL schema per line:
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
    "rationale": "1-3 sentence justification",
    "model": "..."
}

    Full-output mode:
    - Pass --full-output <path> to write records that include ALL original input fields
        plus appended {scores, rationale, model}.
        You can write both lean and full outputs in one pass by providing both --output
        and --full-output.
"""

import argparse
import json
import os
import sys
import time
import threading
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Optional, Tuple

from ollama import Client
from tqdm import tqdm


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


_THREAD_LOCAL = threading.local()
_STOP_EVENT = threading.Event()


def _get_client(base_url: str) -> Client:
    cl: Optional[Client] = getattr(_THREAD_LOCAL, "client", None)
    if cl is None:
        cl = Client(host=base_url)
        setattr(_THREAD_LOCAL, "client", cl)
    return cl


def call_ollama(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    retries: int = 3,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    num_predict: int = 256,
) -> Dict[str, Any]:
    """Call Ollama via the Python client (one paper per call)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if _STOP_EVENT.is_set():
            raise RuntimeError("shutdown requested")
        try:
            client = _get_client(base_url)
            data = client.generate(
                model=model,
                prompt=prompt,
                format="json",
                stream=False,
                options={
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "seed": 42,
                },
            )
            # ollama client returns dict with key 'response'
            raw = data.get("response", "")
            obj = ensure_json(raw)
            if obj is None:
                raise ValueError("Model response was not valid JSON")
            return obj
        except Exception as e:
            last_exc = e
            # brief exponential backoff that can be interrupted by shutdown
            wait_s = min(2 ** attempt, 10)
            if _STOP_EVENT.wait(wait_s):
                raise RuntimeError("shutdown requested")
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


def load_processed_ids(paths: Iterable[str]) -> set:
    seen: set = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
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
            continue
    return seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score papers with a local Ollama model")
    p.add_argument("--input", "-i", type=str, default="arxiv-2007-2019-hepex.jsonl",
                   help="Path to input JSONL file with fields id, title, abstract")
    p.add_argument("--output", "-o", type=str, default="ollama-scores.jsonl",
                   help="Path to output JSONL file to write scores")
    p.add_argument("--full-output", type=str, default="",
                   help="Optional: path to write full JSONL (original fields + appended scores)")
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
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel jobs (each scores one paper at a time)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (lower is more deterministic)")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="Context window tokens for the model")
    p.add_argument("--num-predict", type=int, default=256,
                   help="Max tokens to generate (JSON is short; keep small for speed)")
    return p.parse_args()


def _process_record(
    rec: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (paper_id, lean_json_line, full_json_line, error_message)."""
    pid = rec.get("id") or rec.get("arxiv_id") or rec.get("paper_id")
    title = rec.get("title") or ""
    abstract = rec.get("abstract") or rec.get("summary") or ""

    if not pid or not title or not abstract:
        return None, None, None, "missing required fields"

    prompt = build_prompt(title, abstract)
    obj = call_ollama(
        model=args.model,
        prompt=prompt,
        base_url=args.ollama_url,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    scores = normalize_scores(obj)
    rationale = obj.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    lean_obj = {
        "id": pid,
        "title": title,
        "scores": scores,
        "rationale": rationale.strip(),
        "model": args.model,
    }
    full_obj = dict(rec)
    # Ensure we don't overwrite existing fields unintentionally; append under new keys
    full_obj["scores"] = scores
    full_obj["rationale"] = rationale.strip()
    full_obj["model"] = args.model

    return (
        pid,
        json.dumps(lean_obj, ensure_ascii=False),
        json.dumps(full_obj, ensure_ascii=False),
        None,
    )


def main() -> None:
    args = parse_args()
    
    # Handle SIGTERM gracefully (SIGINT raises KeyboardInterrupt by default)
    def _on_sigterm(signum, frame):
        _STOP_EVENT.set()
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:
        pass

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine progress skip set if requested (union of output files)
    out_paths_for_seen = []
    if args.output:
        out_paths_for_seen.append(args.output)
    if args.full_output:
        out_paths_for_seen.append(args.full_output)
    seen_ids = load_processed_ids(out_paths_for_seen) if args.skip_existing else set()

    # Prepare output file handle (append mode to support resuming)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fout_lean = None
    if args.output:
        out_dir_lean = os.path.dirname(os.path.abspath(args.output)) or "."
        os.makedirs(out_dir_lean, exist_ok=True)
        fout_lean = open(args.output, "a", encoding="utf-8")
    fout_full = None
    if args.full_output:
        out_dir_full = os.path.dirname(os.path.abspath(args.full_output)) or "."
        os.makedirs(out_dir_full, exist_ok=True)
        fout_full = open(args.full_output, "a", encoding="utf-8")

    # Iterate inputs with optional slicing, dispatch to worker pool
    it = iter_jsonl(args.input)
    start_i = max(0, int(args.start or 0))
    end_i = int(args.end or 0)

    processed = 0
    failed = 0
    seen_total = 0

    max_workers = max(1, int(args.workers or 1))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = set()

    def maybe_submit(rec: Dict[str, Any]):
        nonlocal processed, failed
        pid = rec.get("id") or rec.get("arxiv_id") or rec.get("paper_id")
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or rec.get("summary") or ""
        if not pid or not title or not abstract:
            return
        if seen_ids and pid in seen_ids:
            return
        futures.add(executor.submit(_process_record, rec, args))

    interrupted = False
    try:
        with tqdm(desc="Scoring", unit="paper") as pbar:
            for idx, rec in enumerate(it):
                if _STOP_EVENT.is_set():
                    interrupted = True
                    break
                if idx < start_i:
                    continue
                if end_i and idx >= end_i:
                    break
                seen_total += 1

                # Keep at most max_workers tasks in-flight
                while not _STOP_EVENT.is_set() and len(futures) >= max_workers:
                    done = [f for f in list(futures) if f.done()]
                    if not done:
                        if _STOP_EVENT.wait(0.01):
                            interrupted = True
                            break
                        continue
                    for f in done:
                        futures.remove(f)
                        pid, lean_line, full_line, err = f.result()
                        if err is None:
                            if fout_lean is not None and lean_line is not None:
                                fout_lean.write(lean_line + "\n")
                            if fout_full is not None and full_line is not None:
                                fout_full.write(full_line + "\n")
                            processed += 1
                        else:
                            failed += 1
                            if pid:
                                sys.stderr.write(f"[warn] failed on {pid}: {err}\n")
                        pbar.update(1)
                        if args.limit and processed >= args.limit:
                            break
                    if args.limit and processed >= args.limit:
                        break
                if args.limit and processed >= args.limit:
                    break
                if _STOP_EVENT.is_set():
                    interrupted = True
                    break

                maybe_submit(rec)

            # Drain remaining futures
            for f in as_completed(list(futures)):
                if _STOP_EVENT.is_set():
                    interrupted = True
                    break
                pid, lean_line, full_line, err = f.result()
                if err is None:
                    if fout_lean is not None and lean_line is not None:
                        fout_lean.write(lean_line + "\n")
                    if fout_full is not None and full_line is not None:
                        fout_full.write(full_line + "\n")
                    processed += 1
                else:
                    failed += 1
                    if pid:
                        sys.stderr.write(f"[warn] failed on {pid}: {err}\n")
                pbar.update(1)
    except KeyboardInterrupt:
        interrupted = True
        _STOP_EVENT.set()
    finally:
        # Cancel pending tasks to stop quickly
        try:
            executor.shutdown(wait=False, cancel_futures=True)  # type: ignore[arg-type]
        except TypeError:
            executor.shutdown(wait=False)

    if fout_lean is not None:
        fout_lean.close()
    if fout_full is not None:
        fout_full.close()

    summary = {
        "processed": processed,
        "skipped_existing": len(seen_ids) if seen_ids else 0,
        "failed": failed,
        "seen": seen_total,
        "start": start_i,
        "end": end_i,
        "workers": max_workers,
        "interrupted": interrupted,
        "output": os.path.abspath(args.output) if args.output else None,
        "full_output": os.path.abspath(args.full_output) if args.full_output else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
