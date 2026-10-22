#!/usr/bin/env python3
"""Baseline harness for an OpenAI-compatible server (sglang, ours, anything).

Measures what the plan's exit criteria need, at the prompt lengths that matter:

  - TTFT, time to first token, per request
  - TPOT, time per output token after the first
  - per-request tok/s over the whole request
  - aggregate tok/s across concurrent requests

The prompt length is controlled by repetition and the *actual* token count is
read back from the server's `usage` block, so the recorded number is measured,
not assumed. Streaming is required: TTFT is meaningless without it.

Usage:
    python bench/10_openai_bench.py --base-url http://localhost:30000 \
        --model qwen3.8-27b --prompt-tokens 4096 --max-tokens 128 --concurrency 1

    --self-test   start a mock streaming server with known delays, measure it,
                  and assert the harness reports them correctly. Run this first.

Results append to bench/results/openai_<tag>_<stamp>.json with provenance.
Only the standard library is used, so it runs anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"


def find_repo() -> Path:
    """The repo root, or the working directory if this file was copied out of it."""
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand
    return Path.cwd()

FILLER = ("The quick brown fox jumps over the lazy dog while the system measures "
          "tokens per second and writes the result to disk. ")


def build_prompt(target_tokens: int) -> str:
    words = FILLER.split()
    need = max(1, int(target_tokens))
    out = []
    while len(out) < need:
        out.extend(words)
    return " ".join(out[:need])


def one_request(base_url: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
    t_end = time.perf_counter()

    out_tokens = usage.get("completion_tokens") or chunks
    in_tokens = usage.get("prompt_tokens")
    ttft = (t_first - t0) if t_first else None
    decode_s = (t_end - t_first) if t_first else None
    return {
        "ttft_s": ttft,
        "total_s": t_end - t0,
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        "tpot_s": (decode_s / (out_tokens - 1)) if (decode_s and out_tokens > 1) else None,
        "req_tok_s": (out_tokens / (t_end - t0)) if out_tokens else None,
        "stream_chunks": chunks,
    }


def run_batch(base_url: str, model: str, prompt: str, max_tokens: int,
              concurrency: int, timeout: float) -> dict:
    rows: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker():
        try:
            r = one_request(base_url, model, prompt, max_tokens, timeout)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            r = {"error": repr(e)[:300]}
        with lock:
            rows.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            errors.append(r["error"])
    out_tokens = sum(r["out_tokens"] for r in ok)
    return {
        "wall_s": wall,
        "requests": len(rows),
        "errors": errors,
        "aggregate_tok_s": (out_tokens / wall) if ok else None,
        "ttft_s": [r["ttft_s"] for r in ok if r["ttft_s"]],
        "tpot_s": [r["tpot_s"] for r in ok if r["tpot_s"]],
        "req_tok_s": [r["req_tok_s"] for r in ok if r["req_tok_s"]],
        "in_tokens": [r["in_tokens"] for r in ok if r["in_tokens"]],
        "rows": rows,
    }


def summarise(batch: dict) -> dict:
    def med(xs):
        return statistics.median(xs) if xs else None

    return {
        "wall_s": round(batch["wall_s"], 3),
        "requests": batch["requests"],
        "errors": batch["errors"],
        "aggregate_tok_s": round(batch["aggregate_tok_s"], 2) if batch["aggregate_tok_s"] else None,
        "ttft_median_s": med(batch["ttft_s"]),
        "tpot_median_s": med(batch["tpot_s"]),
        "req_tok_s_median": med(batch["req_tok_s"]),
        "prompt_tokens_median": med(batch["in_tokens"]),
    }


# ---------------------------------------------------------------- self-test

class _MockHandler(BaseHTTPRequestHandler):
    first_delay = 0.10
    per_chunk = 0.005
    n_chunks = 40

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(self.first_delay)
        for i in range(self.n_chunks):
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(self.per_chunk)
        done = {"choices": [], "usage": {"prompt_tokens": 1234,
                                        "completion_tokens": self.n_chunks}}
        self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):  # noqa: A002 - name matches the base class
        return


def self_test() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    try:
        r = one_request(url, "mock", "hello", 32, 30)
        # Second pass with no artificial delay: what is left is this harness's
        # own floor (HTTP chunking, JSON parse, python loop). Any claim about a
        # few percent must be read against this number.
        _MockHandler.per_chunk = 0.0
        _MockHandler.first_delay = 0.0
        floor = one_request(url, "mock", "hello", 32, 30)
    finally:
        srv.shutdown()

    checks = [
        ("prompt tokens from usage", r["in_tokens"] == 1234),
        ("output tokens from usage", r["out_tokens"] == 40),
        ("ttft near 100 ms", r["ttft_s"] is not None and 0.08 < r["ttft_s"] < 0.30),
        ("tpot near 5 ms", r["tpot_s"] is not None and 0.002 < r["tpot_s"] < 0.02),
        ("positive request rate", bool(r["req_tok_s"])),
        ("floor below 3 ms/token", floor["tpot_s"] is not None and floor["tpot_s"] < 0.003),
    ]
    ok = True
    print("self-test against a mock streaming server:")
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print(f"  measured ttft={r['ttft_s']:.3f}s tpot={r['tpot_s']:.4f}s "
          f"tokens={r['out_tokens']}")
    print(f"  harness floor: {(floor['tpot_s'] or 0) * 1000:.2f} ms/token, "
          f"{(floor['ttft_s'] or 0) * 1000:.2f} ms ttft")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def _run(cmd: list[str]) -> str | None:
    """Run a tool that may not exist. Never raise from provenance collection."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance(base_url: str) -> dict:
    meta: dict = {"stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "base_url": base_url, "host": os.uname().nodename}
    meta["repo_sha"] = (_run(["git", "-C", str(find_repo()), "rev-parse", "--short", "HEAD"])
                        or None)
    for tool, args in (("rocm-smi", ["--showclocks", "--csv"]),
                       ("amd-smi", ["list"])):
        if shutil.which(tool):
            out = _run([tool, *args])
            meta[f"{tool}_available"] = out is not None
            if out:
                meta[f"{tool}_tail"] = out.splitlines()[-2:]
            break
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tag", default="sglang")
    ap.add_argument("--results-dir", default=None,
                    help="where to write the JSON; default is <repo>/bench/results")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    global RESULTS
    if args.results_dir:
        RESULTS = Path(args.results_dir).expanduser()
    else:
        RESULTS = find_repo() / "bench" / "results"

    if args.self_test:
        return self_test()

    prompt = build_prompt(args.prompt_tokens)
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        "meta": provenance(args.base_url),
        "model": args.model,
        "requested_prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "runs": [],
    }

    for rep in range(args.repeats):
        print(f"run {rep + 1}/{args.repeats}: prompt~{args.prompt_tokens} "
              f"max_tokens={args.max_tokens} concurrency={args.concurrency}")
        batch = run_batch(args.base_url, args.model, prompt, args.max_tokens,
                          args.concurrency, args.timeout)
        s = summarise(batch)
        record["runs"].append({"summary": s, "raw": batch})
        print(f"  ttft={s['ttft_median_s']} tpot={s['tpot_median_s']} "
              f"req_tok_s={s['req_tok_s_median']} aggregate={s['aggregate_tok_s']} "
              f"prompt_tokens={s['prompt_tokens_median']}")
        if s["errors"]:
            print(f"  errors: {s['errors'][:2]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"openai_{args.tag}_{stamp}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
