#!/usr/bin/env python3
"""GGUF v3 inspector for rasputinie.

The engine loads GGUF natively, so the first thing it must do is read the
container correctly. This script parses the header, the metadata and the tensor
directory with mmap, then reports what the file actually holds:

  - which quantisation types are present, and how many bytes each one owns
  - how many parameters each type encodes
  - the largest tensors, by name
  - checks: the tensor sizes must add up to the data section exactly

Tensor sizes are derived from consecutive `offset` fields, so no block-size
table is needed and no assumption is made about a type's layout. Type ids are
named from the table in Strata's reader (MIT), which is the independent reader
this project is allowed to read.

Usage:
    python bench/02_gguf_probe.py MODEL.gguf [--json OUT.json] [--top 12]
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"

VALUE_TYPES = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
    6: "f32", 7: "bool", 8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64",
}

# ggml tensor type ids, per Strata's include/strata/artifact/gguf_reader.hpp
QUANT_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
    143: "PTQ1_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7",
}


class Cursor:
    """Sequential reader over an mmap with the GGUF value grammar."""

    def __init__(self, mm: mmap.mmap):
        self.mm = mm
        self.o = 0

    def take(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self.mm, self.o)
        self.o += size
        return val

    def u32(self) -> int:
        return self.take("<I")[0]

    def u64(self) -> int:
        return self.take("<Q")[0]

    def string(self) -> str:
        n = self.u64()
        s = bytes(self.mm[self.o:self.o + n]).decode("utf-8", "replace")
        self.o += n
        return s

    def skip_string(self) -> None:
        # Do not write `self.o += self.u64()`: the target is read before the
        # call runs, so the 8 bytes the length prefix consumed are lost.
        n = self.u64()
        self.o += n

    def scalar(self, t: int):
        if t == 0:
            return self.take("<B")[0]
        if t == 1:
            return self.take("<b")[0]
        if t == 2:
            return self.take("<H")[0]
        if t == 3:
            return self.take("<h")[0]
        if t == 4:
            return self.take("<I")[0]
        if t == 5:
            return self.take("<i")[0]
        if t == 6:
            return self.take("<f")[0]
        if t == 7:
            return bool(self.take("<B")[0])
        if t == 10:
            return self.take("<Q")[0]
        if t == 11:
            return self.take("<q")[0]
        if t == 12:
            return self.take("<d")[0]
        raise ValueError(f"not a scalar type: {t}")

    def array(self) -> dict:
        """Return a summary. Large arrays are counted, not materialised."""
        elem = self.u32()
        count = self.u64()
        keep, show = [], 6
        for i in range(count):
            if elem == 8:
                if i < show:
                    keep.append(self.string())
                else:
                    self.skip_string()
            elif elem == 9:
                raise ValueError("nested arrays are not part of the format")
            else:
                v = self.scalar(elem)
                if i < show:
                    keep.append(v)
        return {"elem": VALUE_TYPES.get(elem, elem), "count": count, "first": keep}

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            return self.array()
        return self.scalar(t)


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def probe(path: Path, top: int) -> dict:
    size = path.stat().st_size
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm[:4] != MAGIC:
                raise SystemExit(f"{path.name}: not a GGUF file (magic {mm[:4]!r})")
            c = Cursor(mm)
            c.o = 4
            version = c.u32()
            n_tensors = c.u64()
            n_kv = c.u64()

            meta = {}
            for _ in range(n_kv):
                key = c.string()
                vtype = c.u32()
                meta[key] = c.value(vtype)

            tensors = []
            for _ in range(n_tensors):
                name = c.string()
                n_dims = c.u32()
                dims = tuple(c.u64() for _ in range(n_dims))
                ttype = c.u32()
                offset = c.u64()
                tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

            alignment = meta.get("general.alignment", 32)
            if not isinstance(alignment, int) or alignment <= 0:
                alignment = 32
            metadata_end = c.o
            data_start = align(c.o, alignment)
        finally:
            mm.close()

    # Sizes come from the offsets, not from a block-size table.
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(ordered):
        end = ordered[i + 1]["offset"] if i + 1 < len(ordered) else size - data_start
        t["bytes"] = end - t["offset"]
        t["params"] = 1
        for d in t["dims"]:
            t["params"] *= d

    by_type: dict[int, dict] = {}
    for t in ordered:
        key = t["type"]
        slot = by_type.setdefault(key, {"tensors": 0, "bytes": 0, "params": 0})
        slot["tensors"] += 1
        slot["bytes"] += t["bytes"]
        slot["params"] += t["params"]

    data_bytes = size - data_start
    counted = sum(t["bytes"] for t in ordered)

    report = {
        "file": str(path),
        "bytes": size,
        "gib": round(size / 2**30, 2),
        "version": version,
        "alignment": alignment,
        "metadata_end": metadata_end,
        "metadata_padding": data_start - metadata_end,
        "data_start": data_start,
        "tensor_count": len(ordered),
        "kv_count": n_kv,
        "architecture": meta.get("general.architecture"),
        "name": meta.get("general.name"),
        "checks": {
            "sizes_match_data_section": counted == data_bytes,
            "counted_bytes": counted,
            "data_section_bytes": data_bytes,
        },
        "types": [
            {
                "type": QUANT_TYPES.get(k, f"unknown({k})"),
                "type_id": k,
                **v,
                "gib": round(v["bytes"] / 2**30, 3),
                "share_pct": round(100.0 * v["bytes"] / data_bytes, 2) if data_bytes else 0.0,
                "bits_per_weight": round(8.0 * v["bytes"] / v["params"], 3) if v["params"] else 0.0,
            }
            for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])
        ],
        "largest": [
            {
                "name": t["name"],
                "type": QUANT_TYPES.get(t["type"], f"unknown({t['type']})"),
                "dims": list(t["dims"]),
                "gib": round(t["bytes"] / 2**30, 3),
                "params": t["params"],
            }
            for t in sorted(ordered, key=lambda t: -t["bytes"])[:top]
        ],
        # Full manifest: the engine dispatches per tensor, so it needs this list.
        # Offsets are the raw GGUF `offset` (relative to the data section); add
        # `data_start` for an absolute file offset.
        "tensors": [
            {
                "name": t["name"],
                "type": QUANT_TYPES.get(t["type"], f"unknown({t['type']})"),
                "type_id": t["type"],
                "dims": list(t["dims"]),
                "offset": t["offset"],
                "file_offset": data_start + t["offset"],
                "bytes": t["bytes"],
                "params": t["params"],
            }
            for t in ordered
        ],
        "metadata": {
            k: v for k, v in meta.items()
            if not k.startswith("tokenizer.") and not isinstance(v, dict)
        },
        "metadata_arrays": {k: v for k, v in meta.items() if isinstance(v, dict)},
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    r = probe(args.model, args.top)

    print(f"{Path(r['file']).name}  {r['gib']:.2f} GiB  GGUF v{r['version']}")
    print(f"arch={r['architecture']}  name={r['name']}  tensors={r['tensor_count']}  kv={r['kv_count']}")
    ok = "OK" if r["checks"]["sizes_match_data_section"] else "MISMATCH"
    print(f"size check: {ok}  (tensors {r['checks']['counted_bytes']:,} B, data {r['checks']['data_section_bytes']:,} B)")
    print()
    print(f"{'type':<10} {'tensors':>8} {'GiB':>8} {'share':>7} {'bit/w':>6}")
    for t in r["types"]:
        print(f"{t['type']:<10} {t['tensors']:>8} {t['gib']:>8.2f} {t['share_pct']:>6.1f}% {t['bits_per_weight']:>6.2f}")
    print()
    print("largest tensors:")
    for t in r["largest"]:
        print(f"  {t['gib']:>6.2f} GiB  {t['type']:<8} {'x'.join(map(str, t['dims'])):<22} {t['name']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if r["checks"]["sizes_match_data_section"] else 1


if __name__ == "__main__":
    sys.exit(main())
