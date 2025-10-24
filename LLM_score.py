#!/usr/bin/env python3
"""
LLM_scores.py

High-throughput scorer for arXiv-style JSONL (id, title, abstract) that:
- Launches and manages its own Ollama server with true parallelism (OLLAMA_NUM_PARALLEL).
- Supports "thinking" models (e.g., DeepSeek-R1, Qwen3 Thinking) via Ollama's `think` parameter.
- Streams requests via a pooled HTTP client (httpx) or official client, capturing robust telemetry.
- Writes lean and/or full JSONL outputs; optionally saves the reasoning trace.
- Uses fast JSON (orjson) and careful normalization to keep the loop tight and stable.

Assumptions: All imports exist in the environment; do not guard for import errors.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import signal
import atexit
import socket
import subprocess
import threading
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Optional, Tuple, Callable

# ---- Fast JSON (assumed present) ----
import orjson as _orjson
def jloads(s: str | bytes) -> Any: return _orjson.loads(s)
def jdumps(obj: Any) -> str: return _orjson.dumps(obj).decode("utf-8")

# ---- HTTP + official client + progress (assumed present) ----
import httpx
from ollama import Client as OllamaClient
from tqdm import tqdm

# =========================
# Logging setup utilities
# =========================
LOGGER = logging.getLogger("LLM_scores")

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a compact, high-signal formatter."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noisy third-party loggers if any:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

@contextmanager
def log_timing(name: str, extra: Optional[Dict[str, Any]] = None):
    """Context manager that logs the elapsed time for a named operation."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = (time.time() - t0) * 1000.0
        LOGGER.debug("timing | %s | %.2f ms%s",
                     name, dt,
                     f" | extra={extra}" if extra else "")

# =========================
# Constants & Globals
# =========================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_URL = "http://{host}:{port}"
CATEGORIES = (
    ("progress", "Contribution to advancing the field; tangible step forward."),
    ("creativity", "Originality of ideas, approaches, or problem framing."),
    ("novelty", "Newness relative to existing literature and prior art."),
    ("technical_rigor", "Methodological soundness, correctness, and thoroughness."),
    ("clarity", "Clarity of problem statement, methods, and results."),
    ("potential_impact", "Potential to influence future research or applications."),
)
_RUBRIC = "\n".join(f"- {k}: {d}" for k, d in CATEGORIES)

_STOP = threading.Event()
_TL = threading.local()

# =========================
# Utility functions
# =========================
def find_free_port(host: str) -> int:
    """Find and return a free TCP port bound to the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port = s.getsockname()[1]
    LOGGER.debug("network | found free port | host=%s port=%d", host, port)
    return port

def wait_for_http(url: str, timeout: float = 60.0) -> None:
    """Poll Ollama's /api/version until the server is ready or a timeout occurs."""
    LOGGER.info("server | waiting for readiness | url=%s timeout=%.1fs", url, timeout)
    t0 = time.time()
    with httpx.Client(timeout=2.0) as c:
        while True:
            try:
                r = c.get(f"{url}/api/version")
                if r.status_code == 200:
                    LOGGER.info("server | ready | url=%s version=%s", url, r.text.strip())
                    return
            except Exception:
                pass
            if (time.time() - t0) > timeout:
                raise RuntimeError(f"Ollama did not become ready at {url} in {timeout:.1f}s")
            time.sleep(0.1)

# =========================
# Ollama server manager
# =========================
class OllamaServer:
    """Manage a dedicated Ollama server subprocess with parallelism enabled."""

    def __init__(
        self,
        host: str,
        port: int,
        num_parallel: int,
        max_loaded_models: int,
        gpu_layers: Optional[int],
        bin_name: str = "ollama",
        extra_env: Optional[Dict[str, str]] = None,
        shutdown_timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.url = DEFAULT_URL.format(host=host, port=port)
        self.num_parallel = int(num_parallel)
        self.max_loaded_models = int(max_loaded_models)
        self.gpu_layers = gpu_layers
        self.bin_name = bin_name
        self.extra_env = dict(extra_env or {})
        self.shutdown_timeout = shutdown_timeout
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """Launch the Ollama server with the requested parallelism and wait for readiness."""
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        env["OLLAMA_MAX_LOADED_MODELS"] = str(self.max_loaded_models)
        if self.gpu_layers is not None:
            env["OLLAMA_GPU_LAYERS"] = str(int(self.gpu_layers))
        env.update(self.extra_env)

        LOGGER.info("server | starting | bin=%s host=%s port=%d parallel=%d max_models=%d gpu_layers=%s",
                    self.bin_name, self.host, self.port, self.num_parallel,
                    self.max_loaded_models, self.gpu_layers)

        # Start server as its own process group to manage signals cleanly.
        self.proc = subprocess.Popen(
            [self.bin_name, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atexit.register(self.stop)

        # Wait for HTTP readiness.
        wait_for_http(self.url, timeout=60.0)

    def stop(self) -> None:
        """Attempt graceful shutdown; force-kill if it doesn’t exit in time."""
        if not self.proc:
            return
        proc = self.proc
        LOGGER.info("server | stopping | pid=%s", proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        t0 = time.time()
        while proc.poll() is None and (time.time() - t0) < self.shutdown_timeout:
            time.sleep(0.1)

        if proc.poll() is None:
            LOGGER.warning("server | force kill | pid=%s", proc.pid)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.proc = None
        LOGGER.info("server | stopped")

# =========================
# Prompt & JSON helpers
# =========================
def build_prompt(title: str, abstract: str, include_rationale: bool) -> str:
    """Build the scoring prompt with a compact JSON schema and rubric."""
    if include_rationale:
        schema = (
            '{ "scores": { "progress": int, "creativity": int, "novelty": int, '
            '"technical_rigor": int, "clarity": int, "potential_impact": int, "overall": int }, '
            '"rationale": str }'
        )
    else:
        schema = (
            '{ "scores": { "progress": int, "creativity": int, "novelty": int, '
            '"technical_rigor": int, "clarity": int, "potential_impact": int, "overall": int } }'
        )

    prompt = (
        "You are a careful, quantitative reviewer. Read the paper metadata and score it on a 1–5 "
        "integer scale for each rubric below, then output STRICT JSON only.\n\n"
        "Rubric (1=very low, 3=medium, 5=very high):\n"
        f"{_RUBRIC}\n\n"
        "Instructions:\n"
        "- Consider ONLY the title and abstract.\n"
        "- Use integers 1, 2, 3, 4, or 5 only.\n"
        f"- Respond with ONLY a single JSON object with this exact structure:\n{schema}\n\n"
        f"Paper:\nTitle: {title.strip()}\nAbstract: {abstract.strip()}"
    )
    LOGGER.debug("prompt | built | title_len=%d abstract_len=%d rationale=%s",
                 len(title), len(abstract), include_rationale)
    return prompt

def ensure_json(s: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object from a string; trims to outer braces on failure."""
    try:
        return jloads(s)
    except Exception:
        pass
    try:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            return jloads(s[i:j+1])
    except Exception:
        return None
    return None

def normalize_scores(obj: Dict[str, Any]) -> Dict[str, int]:
    """Coerce score fields to int in [1,5], ensure presence, and compute overall if missing."""
    scores = obj.get("scores") if isinstance(obj, dict) else None
    if not isinstance(scores, dict):
        scores = {}
    def coerce(v: Any) -> int:
        try:
            x = int(round(float(v)))
        except Exception:
            x = 3
        return 1 if x < 1 else (5 if x > 5 else x)
    out = {k: coerce(scores.get(k, 3)) for k, _ in CATEGORIES}
    overall = coerce(scores.get("overall", sum(out.values()) / len(out)))
    out["overall"] = overall
    return out

# =========================
# I/O helpers
# =========================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects per non-empty line, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield jloads(s)
            except Exception:
                LOGGER.warning("io | malformed input line skipped")
                continue

def load_processed_ids(paths: Iterable[str]) -> set[str]:
    """Collect a set of already-processed ids from provided JSONL files."""
    seen: set[str] = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = jloads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    for k in ("id", "arxiv_id", "paper_id"):
                        v = obj.get(k)
                        if isinstance(v, str):
                            seen.add(v)
                            break
    LOGGER.info("io | loaded processed ids | count=%d from=%d files", len(seen), len(list(paths)))
    return seen

# =========================
# HTTP clients (thread-local)
# =========================
def _get_httpx_client(base_url: str, timeout: float) -> httpx.Client:
    """Return a per-thread HTTPX client with keep-alive pooling."""
    c = getattr(_TL, "httpx_client", None)
    if c is None:
        c = httpx.Client(
            base_url=base_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
        )
        _TL.httpx_client = c
        LOGGER.debug("httpx | new client | base_url=%s", base_url)
    return c

def _get_ollama_client(base_url: str) -> OllamaClient:
    """Return a per-thread official Ollama client."""
    cl = getattr(_TL, "ollama_client", None)
    if cl is None:
        cl = OllamaClient(host=base_url)
        _TL.ollama_client = cl
        LOGGER.debug("ollama_client | new client | base_url=%s", base_url)
    return cl

# =========================
# Thinking support
# =========================
def negotiate_thinking_support(base_url: str, model: str, timeout: float) -> bool:
    """
    Probe /api/generate with think=True. If server returns an object containing
    a top-level 'thinking' field, treat thinking as supported for this model.
    """
    with log_timing("negotiate_thinking", {"model": model}):
        c = _get_httpx_client(base_url, timeout)
        r = c.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": 'Respond with JSON: {"ok":1}',
                "stream": False,
                "think": True,
                "options": {"temperature": 0.0, "num_ctx": 256, "num_predict": 16, "seed": 42},
            },
        )
        r.raise_for_status()
        data = r.json()
        supported = "thinking" in data
        LOGGER.info("think | negotiated | model=%s supported=%s", model, supported)
        return supported

# =========================
# Core request path
# =========================
def call_ollama(
    model: str,
    prompt: str,
    base_url: str,
    retries: int,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    timeout: float,
    backend: str,
    keep_alive: str | int | float,
    think_value: Any,         # False/True/"low"/"medium"/"high"/None
    allow_format_json: bool,  # only True when thinking is OFF
) -> Dict[str, Any]:
    """
    Execute a single generation against the Ollama server and return the parsed JSON object.
    When 'think_value' is provided, Ollama may return a 'thinking' trace separate from 'response'.
    """
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if _STOP.is_set():
            raise RuntimeError("shutdown requested")
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "seed": 42,
                },
            }
            if think_value is not None:
                payload["think"] = think_value
            if allow_format_json:
                payload["format"] = "json"

            with log_timing("ollama_generate", {"attempt": attempt}):
                if backend == "httpx":
                    c = _get_httpx_client(base_url, timeout)
                    r = c.post("/api/generate", json=payload)
                    r.raise_for_status()
                    data = r.json()
                else:
                    cl = _get_ollama_client(base_url)
                    data = cl.generate(**payload)

            raw = data.get("response", "")
            obj = ensure_json(raw)
            if obj is None:
                raise ValueError("model did not return valid JSON in the response field")

            # Attach trace when requested and available
            if "thinking" in data:
                obj["_thinking_trace"] = data["thinking"]

            LOGGER.debug("ollama | success | tokens=%s", data.get("eval_count"))
            return obj

        except Exception as e:
            last = e
            LOGGER.warning("ollama | attempt failed | attempt=%d error=%s", attempt, e)
            if _STOP.wait(min(2 ** attempt, 8)):
                raise RuntimeError("shutdown requested")
    LOGGER.error("ollama | all retries failed | error=%s", last)
    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last}")

# =========================
# Worker
# =========================
def process_record(
    rec: Dict[str, Any],
    args: argparse.Namespace,
    prompt_fn: Callable[[str, str, bool], str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Process a single input record: build prompt, call model, normalize scores,
    and return serialized lean/full JSON lines (or an error message).
    """
    pid = rec.get("id") or rec.get("arxiv_id") or rec.get("paper_id")
    title = rec.get("title") or ""
    abstract = rec.get("abstract") or rec.get("summary") or ""
    if not (isinstance(pid, str) and title and abstract):
        return None, None, None, "missing required fields"

    prompt = prompt_fn(title, abstract, not args.no_rationale)
    obj = call_ollama(
        model=args.model,
        prompt=prompt,
        base_url=args.ollama_url,
        retries=args.retries,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        timeout=args.timeout,
        backend=args.backend,
        keep_alive=args.keep_alive,
        think_value=args._think_value,
        allow_format_json=(args._think_value in (None, False)),
    )

    scores = normalize_scores(obj)
    lean = {"id": pid, "title": title, "scores": scores, "model": args.model}
    if not args.no_rationale:
        r = obj.get("rationale")
        if isinstance(r, str):
            lean["rationale"] = r.strip()
    if args.save_thinking and isinstance(obj.get("_thinking_trace"), str):
        lean["thinking"] = obj["_thinking_trace"]

    full = dict(rec)
    full["scores"] = scores
    full["model"] = args.model
    if not args.no_rationale and isinstance(obj.get("rationale"), str):
        full["rationale"] = obj["rationale"].strip()
    if args.save_thinking and isinstance(obj.get("_thinking_trace"), str):
        full["thinking"] = obj["_thinking_trace"]

    LOGGER.debug("worker | processed | id=%s", pid)
    return pid, jdumps(lean), jdumps(full), None

# =========================
# Warmup
# =========================
def warmup(args: argparse.Namespace) -> None:
    """Run a tiny generation once to load weights and establish caches."""
    LOGGER.info("warmup | start | model=%s", args.model)
    try:
        _ = call_ollama(
            model=args.model,
            prompt='{"ping":1}',
            base_url=args.ollama_url,
            retries=1,
            temperature=0.0,
            num_ctx=256,
            num_predict=8,
            timeout=min(30.0, args.timeout),
            backend=args.backend,
            keep_alive=args.keep_alive,
            think_value=args._think_value,
            allow_format_json=(args._think_value in (None, False)),
        )
    except Exception as e:
        LOGGER.warning("warmup | failed | %s", e)
    else:
        LOGGER.info("warmup | done")

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for the scoring run."""
    p = argparse.ArgumentParser(description="Score papers with a parallel Ollama server (thinking-ready)")
    # Logging
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # I/O
    p.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL (id,title,abstract)")
    p.add_argument("--out", dest="out", type=str, default="", help="Output JSONL (lean)")
    p.add_argument("--full-out", dest="full_out", type=str, default="", help="Output JSONL (full record + scores)")
    p.add_argument("--skip-existing", action="store_true", help="Skip ids already present in outputs")
    p.add_argument("--limit", type=int, default=0, help="Stop after N processed (>0)")
    p.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    p.add_argument("--end", type=int, default=0, help="End index (exclusive; 0 = to end)")
    # Model/gen
    p.add_argument("--model", "-m", type=str, default="llama3.1:8b", help="Model name loaded by Ollama")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--num-predict", type=int, default=256)
    p.add_argument("--no-rationale", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--keep-alive", type=str, default="5m", help="Keep model hot (e.g., 5m, 0, -1)")
    # Thinking controls
    p.add_argument("--think", choices=("auto", "on", "off", "low", "medium", "high"), default="auto",
                   help="Enable/disable thinking (auto probes support; levels for compatible models)")
    p.add_argument("--save-thinking", action="store_true", help="Include reasoning trace in outputs")
    # Concurrency
    p.add_argument("--workers", "-j", type=int, default=1, help="Client threads to dispatch requests")
    p.add_argument("--backend", choices=("httpx", "ollama"), default="httpx")
    # Server management
    p.add_argument("--host", type=str, default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=0, help="Managed server port; 0 = free port")
    p.add_argument("--parallel", type=int, default=1, help="OLLAMA_NUM_PARALLEL for managed server")
    p.add_argument("--max-models", type=int, default=2, help="OLLAMA_MAX_LOADED_MODELS")
    p.add_argument("--gpu-layers", type=int, default=None, help="Set OLLAMA_GPU_LAYERS (optional)")
    p.add_argument("--ollama-bin", type=str, default="ollama", help="Path to `ollama` binary")
    p.add_argument("--no-warmup", action="store_true", help="Skip the initial warmup generation")
    return p.parse_args()

# =========================
# Main
# =========================
def main() -> None:
    """Entry point: configure logging, start server, run scoring loop, emit summary, and shut down."""
    args = parse_args()
    setup_logging(args.log_level)

    if not args.out and not args.full_out:
        LOGGER.error("config | missing outputs | specify --out and/or --full-out")
        sys.exit(2)
    if not os.path.exists(args.inp):
        LOGGER.error("io | input not found | %s", args.inp)
        sys.exit(1)

    # Start dedicated Ollama server with requested parallelism.
    port = args.port or find_free_port(args.host)
    server = OllamaServer(
        host=args.host,
        port=port,
        num_parallel=args.parallel,
        max_loaded_models=args.max_models,
        gpu_layers=args.gpu_layers,
        bin_name=args.ollama_bin,
    )
    with log_timing("server_start"):
        server.start()
    args.ollama_url = DEFAULT_URL.format(host=args.host, port=port)

    # Resolve think behavior.
    if args.think == "on":
        args._think_value = True
    elif args.think == "off":
        args._think_value = False
    elif args.think in ("low", "medium", "high"):
        args._think_value = args.think
    else:
        with log_timing("negotiate_thinking_auto"):
            supported = negotiate_thinking_support(args.ollama_url, args.model, args.timeout)
        args._think_value = True if supported else False

    # Signal handling.
    def _stop_handler(signum, frame):
        LOGGER.warning("signal | received | %s", signum)
        _STOP.set()
    try:
        signal.signal(signal.SIGTERM, _stop_handler)
        signal.signal(signal.SIGINT, _stop_handler)
    except Exception:
        pass

    # Warmup.
    if not args.no_warmup:
        warmup(args)

    # Prepare outputs.
    f_out = f_full = None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        f_out = open(args.out, "a", encoding="utf-8", buffering=1)
    if args.full_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.full_out)) or ".", exist_ok=True)
        f_full = open(args.full_out, "a", encoding="utf-8", buffering=1)

    # Skip set for resume.
    seen_ids: set[str] = set()
    if args.skip_existing:
        seen_ids = load_processed_ids([p for p in (args.out, args.full_out) if p])

    # Iterate + dispatch.
    start_i = max(0, int(args.start or 0))
    end_i = int(args.end or 0)
    limit = int(args.limit or 0)
    processed = failed = seen_total = 0
    max_workers = max(1, int(args.workers or 1))
    futs = set()
    interrupted = False

    def submit(executor, rec):
        """Submit one record for asynchronous processing."""
        futs.add(executor.submit(process_record, rec, args, build_prompt))

    it = iter_jsonl(args.inp)
    pbar = tqdm(desc="Scoring", unit="paper")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, rec in enumerate(it):
                if _STOP.is_set():
                    interrupted = True
                    break
                if idx < start_i:
                    continue
                if end_i and idx >= end_i:
                    break
                seen_total += 1

                pid = rec.get("id") or rec.get("arxiv_id") or rec.get("paper_id")
                title = rec.get("title")
                abstract = rec.get("abstract") or rec.get("summary")
                if not (pid and title and abstract):
                    pbar.update(1)
                    failed += 1
                    LOGGER.debug("worker | skipped malformed record at idx=%d", idx)
                    continue

                if seen_ids and pid in seen_ids:
                    pbar.update(1)
                    LOGGER.debug("worker | skipped existing | id=%s", pid)
                    continue

                # Keep at most max_workers in flight; drain as needed.
                while len(futs) >= max_workers:
                    done = [f for f in list(futs) if f.done()]
                    if not done:
                        if _STOP.wait(0.01):
                            interrupted = True
                            break
                        continue
                    for f in done:
                        futs.remove(f)
                        pid2, lean_line, full_line, err = f.result()
                        if err is None:
                            if f_out and lean_line: f_out.write(lean_line + "\n")
                            if f_full and full_line: f_full.write(full_line + "\n")
                            processed += 1
                            LOGGER.debug("worker | wrote | id=%s", pid2)
                        else:
                            failed += 1
                            LOGGER.warning("worker | failed | id=%s err=%s", pid2, err)
                        pbar.update(1)
                        if limit and processed >= limit:
                            break
                if limit and processed >= limit:
                    break
                if _STOP.is_set():
                    interrupted = True
                    break

                submit(ex, rec)

            # Drain remaining futures.
            for f in as_completed(list(futs)):
                if _STOP.is_set():
                    interrupted = True
                    break
                pid2, lean_line, full_line, err = f.result()
                if err is None:
                    if f_out and lean_line: f_out.write(lean_line + "\n")
                    if f_full and full_line: f_full.write(full_line + "\n")
                    processed += 1
                    LOGGER.debug("worker | wrote | id=%s", pid2)
                else:
                    failed += 1
                    LOGGER.warning("worker | failed | id=%s err=%s", pid2, err)
                pbar.update(1)

    finally:
        pbar.close()
        if f_out: f_out.close()
        if f_full: f_full.close()
        # Always stop the server.
        with log_timing("server_stop"):
            server.stop()

    # Summary is both logged (telemetry) and printed (program output).
    summary = {
        "processed": processed,
        "failed": failed,
        "seen": seen_total,
        "skipped_existing": len(seen_ids) if seen_ids else 0,
        "start": start_i,
        "end": end_i,
        "workers": max_workers,
        "interrupted": interrupted,
        "url": args.ollama_url,
        "parallel": args.parallel,
        "model": args.model,
        "think": args._think_value,
        "out": os.path.abspath(args.out) if args.out else None,
        "full_out": os.path.abspath(args.full_out) if args.full_out else None,
    }
    LOGGER.info("run_summary | %s", summary)
    print(jdumps(summary))


if __name__ == "__main__":
    main()