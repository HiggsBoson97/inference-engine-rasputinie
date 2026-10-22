#!/usr/bin/env python3
"""03-bandwidth-calc.py -- the throughput table in 03-bandwidth.md, re-derivable.

Inputs are the measured numbers from research/bench/ (see 03-bandwidth.md §2).
Nothing here is a hardware spec; every bandwidth value is a measurement on
rasputin card 1, 2026-10-22.

    python3 03-bandwidth-calc.py            # the table
    python3 03-bandwidth-calc.py --sweep    # tok/s vs sustained GB/s
"""
import argparse

# weight bytes per token at the target quant: 11.76 GB (docs/00-facts.md:114)
BYTES_PER_TOKEN_GB = 11.76

# name -> sustained GB/s of weight traffic, from 03-bandwidth.md
MEASURED = [
    ("torch bf16 GEMM M=1 (1x5120x17408)",      472.0),
    ("old 'wall' 619 (1.6 ms kernel)",          619.0),
    ("read wall, 6 GiB, 4 passes",              775.0),
    ("fused dequant GEMV, isolated, steady",    735.0),
    ("engine-realistic sustained",              650.0),
    ("fused GEMV, single 1.6 ms launch",        635.0),
]


def row(gbs: float) -> tuple[float, float]:
    ms = BYTES_PER_TOKEN_GB / gbs * 1e3      # GB / (GB/s) = s
    return ms, 1e3 / ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    print(f"weight bytes/token = {BYTES_PER_TOKEN_GB} GB (3.48 bpw, 27B)\n")
    w = max(len(n) for n, _ in MEASURED)
    print(f"{'scenario':<{w}}  {'GB/s':>6}  {'ms/token':>8}  {'tok/s':>7}")
    for name, gbs in MEASURED:
        ms, tps = row(gbs)
        print(f"{name:<{w}}  {gbs:6.0f}  {ms:8.1f}  {tps:7.1f}")

    if args.sweep:
        print("\nsustained GB/s -> tok/s")
        for gbs in range(400, 801, 25):
            ms, tps = row(gbs)
            print(f"  {gbs:4d} GB/s  {ms:6.1f} ms  {tps:6.1f} tok/s")

    print("\nchecks")
    # the two reconciliations the note leans on
    print(f"  592.9 GiB/s copy = {592.9 * 2**30 / 1e9:.1f} GB/s "
          f"({592.9 * 2**30 / 1e9 / 800 * 100:.1f} % of 800)")
    print(f"  619 GB/s = {619 / 800 * 100:.1f} % of spec, {619 / 775 * 100:.1f} % of the measured wall")
    print(f"  735 GB/s = {735 / 800 * 100:.1f} % of spec, {735 / 775 * 100:.1f} % of the measured wall")
    print(f"  472 GB/s = {472 / 800 * 100:.1f} % of spec, {472 / 775 * 100:.1f} % of the measured wall")
    # duration sensitivity: the same bytes in one kernel that runs 4x longer
    # (measured, 03-bandwidth.md §1.1). NOT a per-launch model -- see the caveat.
    print("\nduration sensitivity (measured, 1 GiB footprint, same kernel)")
    for label, gbs in (("1 pass  (1.6 ms)", 635.0), ("2 passes (3.3 ms)", 696.0),
                       ("4 passes (6.4 ms)", 746.0), ("8 passes (13 ms)", 738.0)):
        ms, tps = row(gbs)
        print(f"  {label:<18} {gbs:5.0f} GB/s  {ms:5.1f} ms/token  {tps:5.1f} tok/s")
    print("  caveat: 8 back-to-back *separate* launches of the same kernel gave 641, not 738,")
    print("  so the effect recurs per launch and the per-launch cost is unmeasured. Modelling it")
    print("  as 0.2 ms x 450 launches predicts >100 ms/token, which no shipping engine pays.")


if __name__ == "__main__":
    main()
