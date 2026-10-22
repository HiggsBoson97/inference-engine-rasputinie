#!/usr/bin/env python3
"""Hardware envelope probe for rasputinie.

Measures the numbers that decide engine architecture:
  - device inventory and P2P capability
  - per-device HBM bandwidth, and aggregate bandwidth with both GPUs busy
  - PCIe cross-device copy bandwidth
  - host<->device pinned bandwidth
  - BF16 GEMM throughput at decode and prefill shapes, plus per-expert MoE GEMM
  - CPU DRAM bandwidth (warm)
  - NVMe sequential read bandwidth

Run:  ~/venv/sglang-rocm/bin/python bench/00_hardware_probe.py
Writes JSON to bench/results/hardware_probe.json
"""

import json
import os
import subprocess
import threading
import time

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
RESULTS = {}


def timed(fn, iters=5, warmup=2, dev=0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / iters


def inventory():
    rows = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        rows.append(
            {
                "index": i,
                "name": p.name,
                "gcn_arch": getattr(p, "gcnArchName", "?"),
                "total_gib": round(p.total_memory / 1024**3, 2),
                "cu_count": p.multi_processor_count,
                "pcie_bus": f"{p.pci_bus_id:04x}:{p.pci_device_id:02x}",
            }
        )
    RESULTS["devices"] = rows
    return rows


def p2p():
    n = torch.cuda.device_count()
    matrix = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                matrix[f"{i}->{j}"] = bool(torch.cuda.can_device_access_peer(i, j))
            except Exception as exc:  # noqa: BLE001
                matrix[f"{i}->{j}"] = f"error: {exc}"
    RESULTS["p2p"] = matrix


def hbm_bw(dev, gb=2.0):
    n = int(gb * 1024**3) // 2
    src = torch.empty(n, dtype=torch.float16, device=dev)
    dst = torch.empty_like(src)
    t = timed(lambda: dst.copy_(src), dev=dev)
    return 2 * src.numel() * 2 / t / 1e9


def hbm_bw_aggregate(gb=1.5):
    """Both GPUs stream at once: the real ceiling for TP=2."""
    out = {}

    def worker(i):
        out[i] = hbm_bw(i, gb)

    threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {str(k): round(v, 1) for k, v in out.items()}, round(sum(out.values()), 1)


def cross_bw(a, b, gb=0.5):
    n = int(gb * 1024**3) // 2
    src = torch.empty(n, dtype=torch.float16, device=a)
    dst = torch.empty(n, dtype=torch.float16, device=b)
    nbytes = n * 2

    def op():
        dst.copy_(src, non_blocking=False)

    op()
    torch.cuda.synchronize(a)
    torch.cuda.synchronize(b)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        op()
        torch.cuda.synchronize(b)
        best = min(best, time.perf_counter() - t0)
    return nbytes / best / 1e9


def host_bw(dev, gb=0.5):
    n = int(gb * 1024**3) // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    gpu = torch.empty(n, dtype=torch.float16, device=dev)
    nbytes = n * 2

    def up():
        gpu.copy_(host, non_blocking=False)

    def down():
        host.copy_(gpu, non_blocking=False)

    up()
    down()
    torch.cuda.synchronize(dev)
    best_up, best_down = 1e9, 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        up()
        torch.cuda.synchronize(dev)
        best_up = min(best_up, time.perf_counter() - t0)
        t0 = time.perf_counter()
        down()
        torch.cuda.synchronize(dev)
        best_down = min(best_down, time.perf_counter() - t0)
    return nbytes / best_up / 1e9, nbytes / best_down / 1e9


def gemm(dev, m, k, n, dtype=torch.bfloat16, iters=20):
    a = torch.randn(m, k, dtype=dtype, device=dev)
    b = torch.randn(k, n, dtype=dtype, device=dev)
    out = torch.empty(m, n, dtype=dtype, device=dev)
    t = timed(lambda: torch.mm(a, b, out=out), iters=iters, warmup=3, dev=dev)
    flops = 2 * m * k * n
    weight_bytes = k * n * torch.finfo(dtype).bits // 8
    return {
        "ms": round(t * 1e3, 4),
        "tflops": round(flops / t / 1e12, 2),
        "weight_gib_per_s": round(weight_bytes / t / 1024**3, 2),
    }


def cpu_bw(gb=1.5):
    import numpy as np

    n = int(gb * 1024**3) // 4
    a = np.ones(n, dtype=np.float32)
    b = np.empty_like(a)
    b[:] = a[:]  # first touch
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        np.copyto(b, a)
        best = min(best, time.perf_counter() - t0)
    return round(2 * n * 4 / best / 1e9, 2)


def nvme_bw(dev="/dev/nvme0n1", gb=4):
    res = {}
    for bs, count in (("1M", gb * 1024), ("4M", gb * 256)):
        cmd = ["sudo", "-n", "dd", f"if={dev}", "of=/dev/null", f"bs={bs}",
               f"count={count}", "iflag=direct", "status=none"]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        res[bs] = round(gb * 1024**3 / dt / 1e9, 2) if r.returncode == 0 else r.stderr.strip()[:120]
    cmd = ("for i in 1 2 3 4; do sudo -n dd if=/dev/nvme0n1 of=/dev/null bs=1M count=512 "
           "skip=$((i*2048)) iflag=direct status=none & done; wait")
    t0 = time.perf_counter()
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    res["4x_1M_parallel"] = round(4 * 512 * 1024**2 / dt / 1e9, 2) if r.returncode == 0 else "error"
    return res


def main():
    inventory()
    p2p()

    hbm = {}
    for d in RESULTS["devices"]:
        if d["total_gib"] < 1.0:  # skip the 512 MB iGPU
            continue
        try:
            hbm[str(d["index"])] = round(hbm_bw(d["index"]), 1)
        except Exception as exc:  # noqa: BLE001
            hbm[str(d["index"])] = f"error: {exc}"
    RESULTS["hbm_gib_per_s"] = hbm

    if torch.cuda.device_count() >= 2:
        try:
            per, total = hbm_bw_aggregate()
            RESULTS["hbm_aggregate_gib_per_s"] = {"per_gpu": per, "sum": total}
        except Exception as exc:  # noqa: BLE001
            RESULTS["hbm_aggregate_gib_per_s"] = f"error: {exc}"
        try:
            RESULTS["cross_pcie_gib_per_s"] = {
                "0->1": round(cross_bw(0, 1), 2),
                "1->0": round(cross_bw(1, 0), 2),
            }
        except Exception as exc:  # noqa: BLE001
            RESULTS["cross_pcie_gib_per_s"] = f"error: {exc}"

    try:
        up, down = host_bw(0)
        RESULTS["host_pcie_gib_per_s"] = {"h2d": round(up, 2), "d2h": round(down, 2)}
    except Exception as exc:  # noqa: BLE001
        RESULTS["host_pcie_gib_per_s"] = f"error: {exc}"

    for m in (1, 8, 32):
        try:
            RESULTS[f"gemm_bf16_m{m}_k5120_n17408"] = gemm(0, m, 5120, 17408)
        except Exception as exc:  # noqa: BLE001
            RESULTS[f"gemm_bf16_m{m}_k5120_n17408"] = f"error: {exc}"

    try:
        RESULTS["gemm_bf16_m2048_k5120_n17408"] = gemm(0, 2048, 5120, 17408, iters=5)
    except Exception as exc:  # noqa: BLE001
        RESULTS["gemm_bf16_m2048_k5120_n17408"] = f"error: {exc}"

    # Flash-Next expert shape: hidden 2560 -> moe_intermediate 640
    try:
        RESULTS["expert_gemm_bf16_m1_k2560_n640"] = gemm(0, 1, 2560, 640)
        RESULTS["expert_gemm_bf16_m32_k2560_n640"] = gemm(0, 32, 2560, 640)
    except Exception as exc:  # noqa: BLE001
        RESULTS["expert_gemm"] = f"error: {exc}"

    try:
        RESULTS["gemm_fp8_e4m3_m1_k5120_n17408"] = gemm(
            0, 1, 5120, 17408, dtype=torch.float8_e4m3fn
        )
    except Exception as exc:  # noqa: BLE001
        RESULTS["gemm_fp8_e4m3_m1_k5120_n17408"] = f"error: {type(exc).__name__}: {str(exc)[:160]}"

    RESULTS["cpu_dram_gib_per_s"] = cpu_bw()
    RESULTS["nvme_gib_per_s"] = nvme_bw()

    path = os.path.join(OUT, "hardware_probe.json")
    with open(path, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(json.dumps(RESULTS, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
