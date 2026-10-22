#!/usr/bin/env python
"""2-GPU RCCL all-reduce microbenchmark on gfx1100, PCIe-only (no P2P DMA).

Measures the transport sglang uses for --tp 2:
  * torch.distributed NCCL backend  -> on ROCm this IS RCCL (librccl.so)
  * sglang PyNcclCommunicator       -> ctypes ncclAllReduce on librccl.so;
                                       parallel_state.py _all_reduce_in_place()
                                       takes this path, not torch.distributed.

Reports per size: cold (first call, nothing warm) / per-op median / min /
amortized back-to-back. GB/s = bytes moved by one rank / amortized time.

Usage: HIP_VISIBLE_DEVICES=0,1 python rccl_ar_bench.py OUT.json
Env:   BENCH_PYNCCL=0 skip pynccl; BENCH_SIZES="5120,8192" override element counts.
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 5120
DEFAULT_SIZES = [1024, 4096, HIDDEN, 8192, 32768, 131072, 524288, 2097152, 8388608]
SIZES = [int(s) for s in os.environ.get("BENCH_SIZES", "").split(",") if s] or DEFAULT_SIZES
DTYPE = torch.bfloat16
ITERS = 300
ITERS_SINGLE = 50


def _bench_fn(fn, tensor, iters):
    """Back-to-back: one event pair around `iters` calls -> amortized per-op."""
    for _ in range(50):
        fn(tensor)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(tensor)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us/op


def _bench_single(fn, tensor, iters):
    """Per-op latency: event pair around each call, median + min."""
    for _ in range(50):
        fn(tensor)
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn(tensor)
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1e3)
    return statistics.median(out), min(out)


def _cold(fn, tensor):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    fn(tensor)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3


def worker(rank, world, port, results_path, use_pynccl):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.init()

    t0 = time.time()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    init_s = time.time() - t0
    torch.cuda.synchronize()
    gro = torch.distributed.new_group(backend="gloo")

    pynccl = None
    if use_pynccl:
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        pynccl = PyNcclCommunicator(
            group=gro, device=dev, is_symmetric_memory_enabled=lambda: False
        )
        # change_state() is a @contextmanager: keep the object alive for the
        # whole run, otherwise GC runs the finally clause and re-disables it.
        ctx = pynccl.change_state(enable=True)
        ctx.__enter__()
        assert pynccl.disabled is False, "pynccl did not enable"
        assert pynccl.available, "pynccl unavailable (librccl not loaded)"

    expect = world * (world - 1) / 2.0
    # correctness first, before any timing noise
    xc = torch.full((HIDDEN,), float(rank), dtype=DTYPE, device=dev)
    dist.all_reduce(xc)
    assert torch.allclose(xc.float(), torch.full_like(xc, expect).float()), "torch AR wrong"
    if pynccl is not None:
        yc = torch.full((HIDDEN,), float(rank), dtype=DTYPE, device=dev)
        pynccl.all_reduce(yc)
        torch.cuda.synchronize()
        assert torch.allclose(yc.float(), torch.full_like(yc, expect).float()), "pynccl AR wrong"

    rows = []
    for n in SIZES:
        nbytes = n * 2
        x = torch.ones(n, dtype=DTYPE, device=dev)
        if pynccl is not None:
            fn_p = lambda t: pynccl.all_reduce(t)
            cold_p = _cold(fn_p, x)
            med, mn = _bench_single(fn_p, x, ITERS_SINGLE)
            rows.append(dict(size=n, bytes=nbytes, path="pynccl", cold_us=cold_p,
                             amort_us=_bench_fn(fn_p, x, ITERS), median_us=med, min_us=mn))
        fn_t = lambda t: dist.all_reduce(t)
        cold_t = _cold(fn_t, x)
        med, mn = _bench_single(fn_t, x, ITERS_SINGLE)
        rows.append(dict(size=n, bytes=nbytes, path="torch", cold_us=cold_t,
                         amort_us=_bench_fn(fn_t, x, ITERS), median_us=med, min_us=mn))

    if rank == 0:
        meta = dict(
            torch=torch.__version__, backend=str(dist.get_backend()), world=world,
            gcn=torch.cuda.get_device_properties(0).gcnArchName,
            visible=os.environ.get("HIP_VISIBLE_DEVICES"), init_s=init_s,
            nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
            rows=rows, all_reduce_correct=True,
        )
        with open(results_path, "w") as f:
            json.dump(meta, f, indent=1)

    dist.barrier()
    dist.destroy_process_group()


def main():
    world = 2
    port = int(os.environ.get("BENCH_PORT", "29511"))
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rccl_ar_bench.json"
    use_pynccl = os.environ.get("BENCH_PYNCCL", "1") == "1"
    print(f"world={world} pynccl={use_pynccl} device_count={torch.cuda.device_count()}")
    mp.spawn(worker, args=(world, port, out, use_pynccl), nprocs=world, join=True)

    d = json.load(open(out))
    print(f"torch {d['torch']} backend={d['backend']} {d['gcn']} "
          f"nccl_env={d['nccl_env'] or '{}'} pg_init={d['init_s']:.2f}s")
    print(f"all_reduce correctness verified: {d['all_reduce_correct']}")
    hdr = (f"{'tensor':>9} {'bytes':>9} {'path':>7} {'cold us':>8} "
           f"{'amort us':>9} {'med us':>8} {'min us':>8} {'GB/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in d["rows"]:
        gbs = r["bytes"] / (r["amort_us"] * 1e-6) / 1e9
        print(f"{r['size']:>9} {r['bytes']:>9} {r['path']:>7} {r['cold_us']:>8.1f} "
              f"{r['amort_us']:>9.2f} {r['median_us']:>8.2f} {r['min_us']:>8.2f} {gbs:>8.2f}")


if __name__ == "__main__":
    main()
