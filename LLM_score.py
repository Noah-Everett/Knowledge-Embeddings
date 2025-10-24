#!/usr/bin/env python3
"""
arxiv_score_ollama_parallel.py

Ultra-fast scorer for arXiv-style JSONL (id, title, abstract) using a dedicated,
self-managed Ollama server that the script launches with parallel execution enabled
and cleanly shuts down afterward.

Key features:
- Spins up its own `ollama serve` on a chosen host:port with OLLAMA_NUM_PARALLEL=N.
- Waits for readiness, warms the model, processes inputs, then stops the server.
- High-throughput request loop; robust JSON; optional lean/full outputs.
- Auto-picks httpx backend if available (keep-alive pooled connections).

Assumptions (per request): all imports exist; do not check/guard for them.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Optional, Tuple, Callable

# ---------- Fast JSON (no dependency checks) ----------
import orjson as _orjson
def jloads(s: str | bytes) -> Any: return _orjson.loads(s)
def jdumps(obj: Any) -> str: return _orjson.dumps(obj).decode("utf-8")

# ---------- Net client ----------
import httpx
from ollama import Client as OllamaClient
from tqdm import tqdm

# ---------- Constants ----------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_URL_TEMPLATE = "http://{host}:{port}"
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

# ---------- Utilities ----------
def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

def wait_for_http(url: str, timeout: float = 30.0) -> None:
    t0 = time.time()
    with httpx.Client(timeout=2.0) as c:
        while True:
            try:
                r = c.get(f"{url}/api/version")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Ollama did not become ready at {url} within {timeout:.1f}s")
            time.sleep(0.1)

# ---------- Ollama server manager ----------
class OllamaServer:
    def __init__(
        self,
        host: str,
        port: int,
        num_parallel: int,
        max_loaded_models: int,
        gpu_layers: Optional[int],
        extra_env: Dict[str, str] | None = None,
        bin_name: str = "ollama",
        shutdown_timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.url = DEFAULT_URL_TEMPLATE.format(host=host, port=port)
        self.num_parallel = int(num_parallel)
        self.max_loaded_models = int(max_loaded_models)
        self.gpu_layers = gpu_layers
        self.extra_env = dict(extra_env or {})
        self.bin_name = bin_name
        self.shutdown_timeout = shutdown_timeout
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        env["OLLAMA_MAX_LOADED_MODELS"] = str(self.max_loaded_models)
        if self.gpu_layers is not None:
            env["OLLAMA_GPU_LAYERS"] = str(int(self.gpu_layers))
        env.update(self.extra_env)

        # Start dedicated server
        self.proc = subprocess.Popen(
            [self.bin_name, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # avoid inheriting signals from parent
        )
        atexit.register(self.stop)
        # Wait until ready
        wait_for_http(self.url, timeout=60.0)

    def stop(self) -> None:
        proc = self.proc
        if not proc:
            return
        # Try graceful shutdown via signal then kill
        try:
            # Attempt API shutdown if available (not always exposed)
            # Fallback to SIGTERM
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
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.proc = None

# ---------- Prompt / JSON helpers ----------
def build_prompt(title: str, abstract: str, include_rationale: bool) -> str:
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
    return (
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

def ensure_json(s: str) -> Optional[Dict[str, Any]]:
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

# ---------- IO ----------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield jloads(line)
            except Exception:
                continue

def load_processed_ids(paths: Iterable[str]) -> set[str]:
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
    return seen

# ---------- Clients ----------
def _get_httpx_client(base_url: str, timeout: float) -> httpx.Client:
    c = getattr(_TL, "httpx_client", None)
    if c is None:
        c = httpx.Client(
            base_url=base_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
        )
        _TL.httpx_client = c
    return c

def _get_ollama_client(base_url: str) -> OllamaClient:
    cl = getattr(_TL, "ollama_client", None)
    if cl is None:
        cl = OllamaClient(host=base_url)
        _TL.ollama_client = cl
    return cl

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
) -> Dict[str, Any]:
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if _STOP.is_set():
            raise RuntimeError("shutdown requested")
        try:
            if backend == "httpx":
                c = _get_httpx_client(base_url, timeout)
                r = c.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "keep_alive": keep_alive,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": num_ctx,
                            "num_predict": num_predict,
                            "seed": 42,
                        },
                    },
                )
                r.raise_for_status()
                raw = r.json().get("response", "")
            else:
                cl = _get_ollama_client(base_url)
                data = cl.generate(
                    model=model,
                    prompt=prompt,
                    format="json",
                    stream=False,
                    keep_alive=keep_alive,
                    options={
                        "temperature": temperature,
                        "num_ctx": num_ctx,
                        "num_predict": num_predict,
                        "seed": 42,
                    },
                )
                raw = data.get("response", "")
            obj = ensure_json(raw)
            if obj is None:
                raise ValueError("model did not return valid JSON")
            return obj
        except Exception as e:
            last = e
            if _STOP.wait(min(2 ** attempt, 8)):
                raise RuntimeError("shutdown requested")
    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last}")

# ---------- Worker ----------
def process_record(
    rec: Dict[str, Any],
    args: argparse.Namespace,
    prompt_fn: Callable[[str, str, bool], str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
    )
    scores = normalize_scores(obj)
    lean = {"id": pid, "title": title, "scores": scores, "model": args.model}
    if not args.no_rationale:
        r = obj.get("rationale")
        if isinstance(r, str):
            lean["rationale"] = r.strip()

    full = dict(rec)
    full["scores"] = scores
    full["model"] = args.model
    if not args.no_rationale and isinstance(obj.get("rationale"), str):
        full["rationale"] = obj["rationale"].strip()

    return pid, jdumps(lean), jdumps(full), None

# ---------- Warmup ----------
def warmup(args: argparse.Namespace) -> None:
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
        )
    except Exception:
        pass

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score papers with a self-managed parallel Ollama server")
    # I/O
    p.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL (id,title,abstract)")
    p.add_argument("--out", dest="out", type=str, default="", help="Output JSONL (lean)")
    p.add_argument("--full-out", dest="full_out", type=str, default="", help="Output JSONL (full record + scores)")
    p.add_argument("--skip-existing", action="store_true", help="Skip ids already present in outputs")
    p.add_argument("--limit", type=int, default=0, help="Stop after N processed records (>0)")
    p.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    p.add_argument("--end", type=int, default=0, help="End index (exclusive); 0 = to end)")
    # Model/gen
    p.add_argument("--model", "-m", type=str, default="llama3.1:8b", help="Model name loaded by Ollama")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--num-predict", type=int, default=256)
    p.add_argument("--no-rationale", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--keep-alive", type=str, default="5m", help="Keep model hot between calls (e.g., 5m, 0, -1)")
    # Concurrency
    p.add_argument("--workers", "-j", type=int, default=1, help="Client threads to dispatch requests")
    p.add_argument("--backend", choices=("httpx", "ollama"), default="httpx")
    # Server management
    p.add_argument("--host", type=str, default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=0, help="Port for the managed server; 0 = pick a free port")
    p.add_argument("--parallel", type=int, default=1, help="OLLAMA_NUM_PARALLEL for the managed server")
    p.add_argument("--max-models", type=int, default=2, help="OLLAMA_MAX_LOADED_MODELS")
    p.add_argument("--gpu-layers", type=int, default=None, help="Set OLLAMA_GPU_LAYERS (optional)")
    p.add_argument("--ollama-bin", type=str, default="ollama", help="Path to `ollama` binary")
    p.add_argument("--no-warmup", action="store_true", help="Skip the initial warmup generation")
    return p.parse_args()

# ---------- Main ----------
def main() -> None:
    args = parse_args()
    if not args.out and not args.full_out:
        print("At least one of --out or --full-out must be provided.", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.inp):
        print(f"Input not found: {args.inp}", file=sys.stderr)
        sys.exit(1)

    # Start dedicated Ollama server with requested parallelism
    port = args.port or find_free_port(args.host)
    server = OllamaServer(
        host=args.host,
        port=port,
        num_parallel=args.parallel,
        max_loaded_models=args.max_models,
        gpu_layers=args.gpu_layers,
        bin_name=args.ollama_bin,
    )
    server.start()

    # Point all clients at our dedicated server
    args.ollama_url = DEFAULT_URL_TEMPLATE.format(host=args.host, port=port)

    # Warmup
    if not args.no_warmup:
        warmup(args)

    # Prepare outputs
    f_out = f_full = None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        f_out = open(args.out, "a", encoding="utf-8", buffering=1)
    if args.full_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.full_out)) or ".", exist_ok=True)
        f_full = open(args.full_out, "a", encoding="utf-8", buffering=1)

    # Skip set
    seen_ids: set[str] = set()
    if args.skip_existing:
        seen_ids = load_processed_ids([p for p in (args.out, args.full_out) if p])

    # Signal handling
    def _stop_handler(signum, frame): _STOP.set()
    try:
        signal.signal(signal.SIGTERM, _stop_handler)
        signal.signal(signal.SIGINT, _stop_handler)
    except Exception:
        pass

    # Iterate + dispatch
    start_i = max(0, int(args.start or 0))
    end_i = int(args.end or 0)
    limit = int(args.limit or 0)

    processed = failed = seen_total = 0
    max_workers = max(1, int(args.workers or 1))
    futs = set()

    def submit(executor, rec):
        futs.add(executor.submit(process_record, rec, args, build_prompt))

    interrupted = False
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
                    continue

                if seen_ids and pid in seen_ids:
                    pbar.update(1)
                    continue

                # keep at most max_workers in flight
                while len(futs) >= max_workers:
                    done = [f for f in list(futs) if f.done()]
                    if not done:
                        if _STOP.wait(0.01):
                            interrupted = True
                            break
                        continue
                    for f in done:
                        futs.remove(f)
                        pid, lean_line, full_line, err = f.result()
                        if err is None:
                            if f_out and lean_line: f_out.write(lean_line + "\n")
                            if f_full and full_line: f_full.write(full_line + "\n")
                            processed += 1
                        else:
                            failed += 1
                            if pid:
                                sys.stderr.write(f"[warn] failed on {pid}: {err}\n")
                        pbar.update(1)
                        if limit and processed >= limit:
                            break
                if limit and processed >= limit:
                    break
                if _STOP.is_set():
                    interrupted = True
                    break

                submit(ex, rec)

            for f in as_completed(list(futs)):
                if _STOP.is_set():
                    interrupted = True
                    break
                pid, lean_line, full_line, err = f.result()
                if err is None:
                    if f_out and lean_line: f_out.write(lean_line + "\n")
                    if f_full and full_line: f_full.write(full_line + "\n")
                    processed += 1
                else:
                    failed += 1
                    if pid:
                        sys.stderr.write(f"[warn] failed on {pid}: {err}\n")
                pbar.update(1)
    finally:
        pbar.close()
        if f_out: f_out.close()
        if f_full: f_full.close()
        # Stop server no matter what
        server.stop()

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
        "out": os.path.abspath(args.out) if args.out else None,
        "full_out": os.path.abspath(args.full_out) if args.full_out else None,
    }
    print(jdumps(summary))


if __name__ == "__main__":
    main()