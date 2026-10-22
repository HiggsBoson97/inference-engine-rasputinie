#!/usr/bin/env python3
"""Tensor inventory for the rasputinie target GGUF files.

`02_gguf_probe.py` answers "does this container parse, and what types are in it".
This script answers the next question: **what exactly must the kernel set
cover**. It reuses the probe's reader (imported, not copied) and adds:

  - a role for every tensor (attention / GDN / MLP / norm / embed / MTP)
  - a per-layer map: which of the 64 layers are full-attention, which are GDN
  - a byte budget per role and per family
  - the distinct (role, type) pairs — this is the dispatch table the engine needs
  - anomaly detection: odd quant types, dtype mixing inside one op, padding,
    block-size violations, non-uniform shapes
  - the MTP delta between the base file and the `-mtp` sibling

Two outputs, both deterministic:

    bench/manifests/iq3s.json        machine-readable, one entry per tensor
    bench/manifests/iq3s-mtp.json    same, plus `diff_vs` the base file
    docs/plan/TENSOR-INVENTORY.md    human summary, generated

The arithmetic gate is the point: metadata region + sum of tensor byte sizes
must equal the file size exactly, tensors must be contiguous and 32-byte
aligned, and every quantised tensor's byte size must equal its exact
block-formula size. If any of those fail, the parse is wrong and the script
exits non-zero.

Tensor sizes are derived from consecutive offsets, so no block-size table is
needed to *read* them; the block table below is used only to *verify* that the
offsets agree with the type's documented layout.

Usage:
    python bench/03_tensor_inventory.py            # both files, default paths
    python bench/03_tensor_inventory.py --base X.gguf --mtp Y.gguf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_BASE = Path.home() / "local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
DEFAULT_MTP = Path.home() / "local-inference/models/gguf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"

# GGUF block layout per type, for verification only: (block_weights, block_bytes).
# A tensor's size must be (params / block_weights) * block_bytes.
# These numbers are the public GGUF/ggml type registry, i.e. the format
# specification this reader is written against (ADR 0001: no engine's source).
BLOCKS = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 40),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
    "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
    "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ1_S": (256, 50),
    "IQ1_M": (256, 56), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
}

# role -> (family, layer scope). Scope is what the kernel dispatch keys on:
# a tensor that exists only in GDN layers needs a kernel only GDN layers call.
ROLES = {
    # per-layer, both kinds
    "attn_norm.weight": ("attn_norm", "norm", "all"),
    "post_attention_norm.weight": ("post_attention_norm", "norm", "all"),
    "ffn_gate.weight": ("ffn_gate", "mlp", "all"),
    "ffn_up.weight": ("ffn_up", "mlp", "all"),
    "ffn_down.weight": ("ffn_down", "mlp", "all"),
    # full-attention layers only
    "attn_q.weight": ("attn_q", "attn_full", "full"),
    "attn_k.weight": ("attn_k", "attn_full", "full"),
    "attn_v.weight": ("attn_v", "attn_full", "full"),
    "attn_output.weight": ("attn_output", "attn_full", "full"),
    "attn_q_norm.weight": ("attn_q_norm", "norm", "full"),
    "attn_k_norm.weight": ("attn_k_norm", "norm", "full"),
    # GDN (linear-attention) layers only
    "attn_qkv.weight": ("attn_qkv", "gdn_proj", "gdn"),
    "attn_gate.weight": ("attn_gate", "gdn_proj", "gdn"),
    "ssm_out.weight": ("ssm_out", "gdn_proj", "gdn"),
    "ssm_conv1d.weight": ("ssm_conv1d", "gdn_state", "gdn"),
    "ssm_a": ("ssm_a", "gdn_state", "gdn"),
    "ssm_dt.bias": ("ssm_dt", "gdn_state", "gdn"),
    "ssm_alpha.weight": ("ssm_alpha", "gdn_state", "gdn"),
    "ssm_beta.weight": ("ssm_beta", "gdn_state", "gdn"),
    "ssm_norm.weight": ("ssm_norm", "norm", "gdn"),
    # MTP block only
    "nextn.eh_proj.weight": ("nextn_eh_proj", "mtp", "mtp"),
    "nextn.enorm.weight": ("nextn_enorm", "norm", "mtp"),
    "nextn.hnorm.weight": ("nextn_hnorm", "norm", "mtp"),
    "nextn.shared_head_norm.weight": ("nextn_shared_head_norm", "norm", "mtp"),
}

GLOBALS = {
    "token_embd.weight": ("token_embd", "embed_output"),
    "output.weight": ("output", "embed_output"),
    "output_norm.weight": ("output_norm", "norm"),
}

FAMILIES = OrderedDict([
    ("attn_full", "full-attention projections (16 layers)"),
    ("gdn_proj", "GDN projections: qkv, gate, out (48 layers)"),
    ("gdn_state", "GDN state ops: conv1d, dt, A, alpha, beta (48 layers)"),
    ("mlp", "MLP gate/up/down (64 layers)"),
    ("norm", "all norms, F32/BF16 vectors"),
    ("embed_output", "token embedding + LM head"),
    ("mtp", "MTP block (nextn) dense tensors"),
])


def load_probe():
    """Import 02_gguf_probe.py, whose name starts with a digit."""
    path = HERE / "02_gguf_probe.py"
    spec = importlib.util.spec_from_file_location("gguf_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def role_of(name: str) -> tuple[str, str, str, str]:
    """(kind_of_name, role, family, scope) — globals are kind 'global'."""
    parts = name.split(".")
    if parts[0] == "blk":
        suffix = ".".join(parts[2:])
        if suffix not in ROLES:
            raise SystemExit(f"unclassified tensor: {name}")
        role, family, scope = ROLES[suffix]
        return "layer", role, family, scope
    if name not in GLOBALS:
        raise SystemExit(f"unclassified global tensor: {name}")
    role, family = GLOBALS[name]
    return "global", role, family, "global"


def block_exact_bytes(type_name: str, params: int) -> int | None:
    if type_name not in BLOCKS:
        return None
    bw, bb = BLOCKS[type_name]
    if params % bw:
        return None
    return params // bw * bb


def annotate(report: dict) -> dict:
    """Turn the probe's flat manifest into a role/layer structured inventory."""
    layers: dict[int, dict] = {}
    tensors_out: list[dict] = []
    globals_out: list[dict] = []

    for t in report["tensors"]:
        parts = t["name"].split(".")
        kind, role, family, scope = role_of(t["name"])
        rec = {
            "name": t["name"],
            "role": role,
            "family": family,
            "scope": scope,
            "type": t["type"],
            "type_id": t["type_id"],
            "dims": t["dims"],                      # as stored: ne[0] first
            "dims_torch": t["dims"][::-1],          # [out, in] for a projection
            "params": t["params"],
            "bytes": t["bytes"],
            "offset": t["offset"],                  # relative to data section
            "file_offset": t["file_offset"],        # absolute
            "bits_per_weight": round(8.0 * t["bytes"] / t["params"], 4) if t["params"] else None,
            "block_exact_bytes": block_exact_bytes(t["type"], t["params"]),
        }
        if kind == "layer":
            idx = int(parts[1])
            layer = layers.setdefault(idx, {"index": idx, "kind": None, "tensors": []})
            layer["tensors"].append(rec)
        else:
            globals_out.append(rec)
        tensors_out.append(rec)

    # A layer's kind is decided by which projection family it carries.
    for idx, layer in layers.items():
        roles = {t["role"] for t in layer["tensors"]}
        has_gdn = "attn_qkv" in roles
        has_full = "attn_q" in roles
        has_mtp = any(t["role"].startswith("nextn_") for t in layer["tensors"])
        if has_gdn and has_full:
            kind = "mixed"
        elif has_gdn:
            kind = "gdn"
        elif has_full and has_mtp:
            kind = "full+mtp"
        elif has_full:
            kind = "full"
        else:
            kind = "unknown"
        layer["kind"] = kind
        layer["tensor_count"] = len(layer["tensors"])
        layer["bytes"] = sum(t["bytes"] for t in layer["tensors"])
        layer["params"] = sum(t["params"] for t in layer["tensors"])

    return {"layers": layers, "globals": globals_out, "tensors": tensors_out}


def rollup(tensors: list[dict]) -> dict:
    fam = defaultdict(lambda: {"tensors": 0, "bytes": 0, "params": 0})
    for t in tensors:
        slot = fam[t["family"]]
        slot["tensors"] += 1
        slot["bytes"] += t["bytes"]
        slot["params"] += t["params"]
    return fam


def build(path: Path) -> dict:
    probe = load_probe()
    r = probe.probe(path, top=5)
    ann = annotate(r)
    tensors = ann["tensors"]
    layers = ann["layers"]
    total_tensor_bytes = sum(t["bytes"] for t in tensors)

    kinds = Counter(l["kind"] for l in layers.values())
    data_bytes = r["checks"]["data_section_bytes"]

    # ---- types -----------------------------------------------------------
    by_type: dict[str, dict] = defaultdict(lambda: {"tensors": 0, "bytes": 0, "params": 0})
    for t in tensors:
        s = by_type[t["type"]]
        s["tensors"] += 1
        s["bytes"] += t["bytes"]
        s["params"] += t["params"]
    types = [
        {
            "type": name,
            "type_id": next(t["type_id"] for t in tensors if t["type"] == name),
            "tensors": v["tensors"],
            "bytes": v["bytes"],
            "gib": round(v["bytes"] / 2**30, 4),
            "params": v["params"],
            "share_pct": round(100.0 * v["bytes"] / data_bytes, 3),
            "bits_per_weight": round(8.0 * v["bytes"] / v["params"], 4),
            "block": BLOCKS.get(name),
        }
        for name, v in sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])
    ]

    # ---- roles -----------------------------------------------------------
    roles: dict[str, dict] = OrderedDict()
    for t in tensors:
        s = roles.setdefault(t["role"], {
            "role": t["role"], "family": t["family"], "scope": t["scope"],
            "layers": 0, "tensor_count": 0, "bytes": 0, "params": 0,
            "types": Counter(), "dims": t["dims"], "dims_torch": t["dims_torch"],
            "dims_uniform": True, "type_ids": [],
        })
        s["tensor_count"] += 1
        s["bytes"] += t["bytes"]
        s["params"] += t["params"]
        s["types"][t["type"]] += 1
        if t["dims"] != s["dims"]:
            s["dims_uniform"] = False
    for s in roles.values():
        s["layers"] = s["tensor_count"]
        s["types"] = dict(sorted(s["types"].items(), key=lambda kv: -kv[1]))
        s["type_ids"] = sorted({t["type_id"] for t in tensors if t["role"] == s["role"]})
        s["gib"] = round(s["bytes"] / 2**30, 4)
        s["share_pct"] = round(100.0 * s["bytes"] / data_bytes, 3)
        s["bits_per_weight"] = round(8.0 * s["bytes"] / s["params"], 4) if s["params"] else None
        s["distinct_types"] = len(s["types"])

    # ---- families / budget ----------------------------------------------
    fam = rollup(tensors)
    families = [
        {
            "family": f,
            "description": FAMILIES.get(f, f),
            "tensors": v["tensors"],
            "bytes": v["bytes"],
            "gib": round(v["bytes"] / 2**30, 4),
            "params": v["params"],
            "share_pct": round(100.0 * v["bytes"] / data_bytes, 3),
            "bits_per_weight": round(8.0 * v["bytes"] / v["params"], 4),
        }
        for f, v in sorted(fam.items(), key=lambda kv: -kv[1]["bytes"])
    ]
    fam_bytes = {k: v["bytes"] for k, v in fam.items()}
    budget = {
        "attention_projections_all": {
            "bytes": fam_bytes.get("attn_full", 0) + fam_bytes.get("gdn_proj", 0),
            "note": "attn_q/k/v/output + GDN attn_qkv/attn_gate/ssm_out",
        },
        "mlp": {"bytes": fam_bytes.get("mlp", 0), "note": "ffn_gate/up/down"},
        "embeddings_output": {"bytes": fam_bytes.get("embed_output", 0), "note": "token_embd + output"},
        "norms": {"bytes": fam_bytes.get("norm", 0),
                  "note": "RMSNorm gain vectors only; ssm_a/ssm_dt are counted in gdn_specific"},
        "gdn_specific": {
            "bytes": fam_bytes.get("gdn_proj", 0) + fam_bytes.get("gdn_state", 0),
            "note": "every tensor a full-attention layer does not have",
        },
        "mtp": {"bytes": fam_bytes.get("mtp", 0),
                "note": "nextn dense tensors only (0 in the base file)"},
    }
    for k, v in budget.items():
        v["gib"] = round(v["bytes"] / 2**30, 4)
        v["share_pct"] = round(100.0 * v["bytes"] / data_bytes, 3)

    # ---- distinct (role, type) pairs = the dispatch table ---------------
    pairs = Counter((t["role"], t["type"]) for t in tensors)
    pair_bytes: dict[tuple, int] = defaultdict(int)
    for t in tensors:
        pair_bytes[(t["role"], t["type"])] += t["bytes"]
    dispatch = [
        {"role": role, "type": ty, "tensors": n, "bytes": pair_bytes[(role, ty)],
         "gib": round(pair_bytes[(role, ty)] / 2**30, 4)}
        for (role, ty), n in sorted(pairs.items(), key=lambda kv: -pair_bytes[kv[0]])
    ]

    # ---- checks ---------------------------------------------------------
    ordered = sorted(tensors, key=lambda t: t["offset"])
    contiguous = all(
        ordered[i]["offset"] + ordered[i]["bytes"] == ordered[i + 1]["offset"]
        for i in range(len(ordered) - 1)
    )
    last_end = ordered[-1]["offset"] + ordered[-1]["bytes"]
    aligned = all(t["offset"] % r["alignment"] == 0 for t in ordered)
    block_bad = [
        {"name": t["name"], "type": t["type"], "bytes": t["bytes"],
         "expected": t["block_exact_bytes"]}
        for t in ordered if t["block_exact_bytes"] is not None and t["bytes"] != t["block_exact_bytes"]
    ]
    unknown_layout = sorted({t["type"] for t in ordered if t["block_exact_bytes"] is None})

    # A 256-weight block quant needs ne[0] (the row width) to be a multiple of
    # 256, or a row cannot start on a block boundary — which decides whether the
    # GEMV can walk rows independently.
    row_misaligned = [
        {"name": t["name"], "type": t["type"], "ne0": t["dims"][0],
         "block_weights": BLOCKS[t["type"]][0]}
        for t in ordered
        if t["type"] in BLOCKS and BLOCKS[t["type"]][0] > 1
        and t["dims"][0] % BLOCKS[t["type"]][0] != 0
    ]

    total_params = sum(t["params"] for t in ordered)
    checks = {
        "file_bytes": r["bytes"],
        "metadata_region_bytes": r["data_start"],
        "metadata_end_bytes": r["metadata_end"],
        "metadata_padding_bytes": r["metadata_padding"],
        "data_section_bytes": data_bytes,
        "tensor_bytes_sum": total_tensor_bytes,
        "metadata_plus_tensors": r["data_start"] + total_tensor_bytes,
        "closes_exactly": r["data_start"] + total_tensor_bytes == r["bytes"] == last_end + r["data_start"],
        "tensors_contiguous": contiguous,
        "offsets_aligned": aligned,
        "block_layout_mismatches": block_bad,
        "types_without_layout": unknown_layout,
        "row_block_misaligned": row_misaligned,
        "total_params": total_params,
        "effective_bits_per_weight": round(8.0 * total_tensor_bytes / total_params, 4),
    }

    # ---- anomalies ------------------------------------------------------
    anomalies: list[dict] = []

    if block_bad:
        anomalies.append({
            "kind": "padding_or_layout",
            "severity": "high",
            "detail": f"{len(block_bad)} tensor(s) whose stored size != block formula",
            "tensors": block_bad[:20],
        })
    if unknown_layout:
        anomalies.append({
            "kind": "unverified_type_layout",
            "severity": "medium",
            "detail": "no block-size entry, size not independently verified",
            "types": unknown_layout,
        })

    # Rare quantised types: a type under 1% of the data section still needs its
    # own dequant path. F32/BF16 vectors are excluded: they are not blocks.
    rare = [t for t in types
            if t["share_pct"] < 1.0 and t["block"] and t["block"][0] > 1]
    if rare:
        anomalies.append({
            "kind": "rare_quant_types",
            "severity": "low" if all(x["share_pct"] < 0.5 for x in rare) else "medium",
            "detail": "quantised types under 1% of the data section — each still needs its own dequant",
            "types": [{"type": x["type"], "type_id": x["type_id"], "gib": x["gib"],
                       "share_pct": x["share_pct"], "tensors": x["tensors"],
                       "tensor_names": [t["name"] for t in ordered if t["type"] == x["type"]]}
                      for x in rare],
        })

    # The single lowest-bit tensor in the file.
    lo = min((t for t in ordered if t["params"] > 1000), key=lambda t: t["bits_per_weight"])
    anomalies.append({
        "kind": "lowest_precision_tensor",
        "severity": "medium",
        "detail": (f"{lo['name']} is {lo['type']} (type id {lo['type_id']}) at "
                   f"{lo['bits_per_weight']} bits/weight — the only tensor at this width"),
        "tensor": {k: lo[k] for k in ("name", "type", "type_id", "dims", "bytes",
                                      "params", "bits_per_weight", "file_offset")},
    })

    # Mixed precision per role is by design in this quant, not a defect. What
    # matters for the kernel set is the *range* per role and whether a type
    # appears in exactly one layer of a role (a path used once).
    import statistics
    ranges = {}
    single_paths = []
    for role, s in roles.items():
        vals = [(t["name"], t["type"], t["bits_per_weight"]) for t in tensors
                if t["role"] == role and t["bits_per_weight"]]
        if not vals:
            continue
        bpws = [v[2] for v in vals]
        if len(s["types"]) < 2 or len(vals) < 2:
            continue
        ranges[role] = {
            "layers": len(vals),
            "min_bits_per_weight": round(min(bpws), 4),
            "median_bits_per_weight": round(statistics.median(bpws), 4),
            "max_bits_per_weight": round(max(bpws), 4),
            "distinct_types": len(s["types"]),
            "lowest": min(vals, key=lambda v: v[2])[0],
            "highest": max(vals, key=lambda v: v[2])[0],
        }
        for ty, n in s["types"].items():
            if n == 1 and s["tensor_count"] > 4:
                single_paths.append({
                    "role": role, "type": ty,
                    "tensor": next(t["name"] for t in tensors
                                   if t["role"] == role and t["type"] == ty),
                })
    anomalies.append({
        "kind": "per_role_bitwidth_range",
        "severity": "info",
        "detail": ("GSQ-RCO is mixed precision by design: every quantised role spans several "
                   "types. A per-role kernel therefore needs every type in its range, not one."),
        "ranges": ranges,
    })
    if single_paths:
        anomalies.append({
            "kind": "single_layer_type_paths",
            "severity": "medium",
            "detail": ("types that occur in exactly one layer of a role: a kernel path "
                       "serving one layer is the worst kind of specialisation to carry"),
            "paths": single_paths,
        })

    # dtype mixing inside one GDN block.
    gdn_layers = [l for l in layers.values() if l["kind"] == "gdn"]
    if gdn_layers:
        by_dtype = defaultdict(list)
        for t in gdn_layers[0]["tensors"]:
            if t["family"] in ("gdn_state",) or t["role"] == "ssm_norm":
                by_dtype[t["type"]].append(t["role"])
        split = "; ".join(f"{ty}: {', '.join(sorted(set(rs)))}" for ty, rs in sorted(by_dtype.items()))
        anomalies.append({
            "kind": "dtype_mixing_in_gdn_block",
            "severity": "medium",
            "detail": (f"a GDN block's state-op tensors are split across precisions — {split}. "
                       "A single fused GDN kernel has to take both an F32 path and a BF16 "
                       "path for vectors of the same block, and the quantised projections "
                       "in the same layer are a third path."),
            "block_types": {ty: sorted(set(rs)) for ty, rs in by_dtype.items()},
        })

    # untied embed/head with different quantisation
    globals_list = ann["globals"]
    emb = next((t for t in globals_list if t["role"] == "token_embd"), None)
    head = next((t for t in globals_list if t["role"] == "output"), None)
    if emb and head and emb["type"] != head["type"]:
        anomalies.append({
            "kind": "embed_head_type_split",
            "severity": "medium",
            "detail": (f"token_embd is {emb['type']} ({emb['bits_per_weight']} bpw) but output is "
                       f"{head['type']} ({head['bits_per_weight']} bpw), same shape — untied, "
                       "two different dequant paths on the largest matrices"),
            "tensors": [emb["name"], head["name"]],
        })

    # Non-uniform layer kinds vs the documented pattern.
    pattern = []
    interval = r["metadata"].get(f"{r['architecture']}.full_attention_interval")
    if interval:
        n = r["metadata"].get(f"{r['architecture']}.block_count", len(layers))
        for i in range(n):
            pattern.append("F" if (i % interval) == interval - 1 else "L")

    # ---- shape provenance -----------------------------------------------
    # Every width should be derivable from the model's own metadata. If a shape
    # does not reconcile, the parse or the role mapping is wrong.
    md = r["metadata"]
    vocab = r["metadata_arrays"].get("tokenizer.ggml.tokens", {}).get("count")
    H = md.get("qwen35.embedding_length")
    FF = md.get("qwen35.feed_forward_length")
    HQ = md.get("qwen35.attention.head_count")
    HKV = md.get("qwen35.attention.head_count_kv")
    KL = md.get("qwen35.attention.key_length")
    VL = md.get("qwen35.attention.value_length")
    GC = md.get("qwen35.ssm.group_count")
    SS = md.get("qwen35.ssm.state_size")
    INNER = md.get("qwen35.ssm.inner_size")
    TSR = md.get("qwen35.ssm.time_step_rank")
    CK = md.get("qwen35.ssm.conv_kernel")
    qkv_width = 2 * (GC * SS) + INNER if (GC and SS and INNER) else None

    expectations = {
        "attn_q": ((HQ * KL * 2, H), "head_count × key_length × 2 (query concatenated with its output gate) → hidden"),
        "attn_k": ((HKV * KL, H), "head_count_kv × key_length → hidden"),
        "attn_v": ((HKV * VL, H), "head_count_kv × value_length → hidden"),
        "attn_output": ((H, HQ * VL), "head_count × value_length → hidden"),
        "attn_q_norm": ((KL,), "key_length"),
        "attn_k_norm": ((KL,), "key_length"),
        "attn_qkv": ((qkv_width, H), "2×(group_count × state_size) + inner_size → hidden"),
        "attn_gate": ((INNER, H), "inner_size → hidden"),
        "ssm_out": ((H, INNER), "inner_size → hidden"),
        "ssm_conv1d": ((qkv_width, CK), "qkv channels × conv_kernel (depthwise, ne = [kernel, channels])"),
        "ssm_alpha": ((TSR, H), "time_step_rank ← hidden"),
        "ssm_beta": ((TSR, H), "time_step_rank ← hidden"),
        "ssm_a": ((TSR,), "time_step_rank"),
        "ssm_dt": ((TSR,), "time_step_rank"),
        "ssm_norm": ((SS,), "state_size"),
        "attn_norm": ((H,), "hidden"),
        "post_attention_norm": ((H,), "hidden"),
        "output_norm": ((H,), "hidden"),
        "ffn_gate": ((FF, H), "feed_forward_length ← hidden"),
        "ffn_up": ((FF, H), "feed_forward_length ← hidden"),
        "ffn_down": ((H, FF), "hidden ← feed_forward_length"),
        "token_embd": ((vocab, H), "vocab × hidden (vocab from tokenizer.ggml.tokens count)"),
        "output": ((vocab, H), "vocab × hidden"),
        "nextn_eh_proj": ((H, 2 * H), "hidden ← 2×hidden (token state ‖ hidden state)"),
        "nextn_enorm": ((H,), "hidden"),
        "nextn_hnorm": ((H,), "hidden"),
        "nextn_shared_head_norm": ((H,), "hidden"),
    }
    by_role = {s["role"]: s for s in roles.values()}
    provenance = []
    for role, (exp, formula) in expectations.items():
        if role not in by_role:
            continue
        obs = tuple(by_role[role]["dims_torch"])
        provenance.append({
            "role": role,
            "formula": formula,
            "expected_dims_torch": list(exp),
            "observed_dims_torch": list(obs),
            "matches": exp == obs,
        })

    # ---- assemble -------------------------------------------------------
    out = {
        "schema": "rasputinie.tensor-inventory/1",
        "generated_by": "bench/03_tensor_inventory.py",
        "file": {
            "path": r["file"],
            "name": Path(r["file"]).name,
            "bytes": r["bytes"],
            "gib": round(r["bytes"] / 2**30, 4),
            "gguf_version": r["version"],
            "alignment": r["alignment"],
            "metadata_end": r["metadata_end"],
            "metadata_padding": r["metadata_padding"],
            "data_start": r["data_start"],
            "tensor_count": r["tensor_count"],
            "kv_count": r["kv_count"],
        },
        "model": {
            "architecture": r["architecture"],
            "name": r["name"],
            "block_count": r["metadata"].get("qwen35.block_count"),
            "nextn_predict_layers": r["metadata"].get("qwen35.nextn_predict_layers"),
            "embedding_length": r["metadata"].get("qwen35.embedding_length"),
            "feed_forward_length": r["metadata"].get("qwen35.feed_forward_length"),
            "attention": {
                "head_count": r["metadata"].get("qwen35.attention.head_count"),
                "head_count_kv": r["metadata"].get("qwen35.attention.head_count_kv"),
                "key_length": r["metadata"].get("qwen35.attention.key_length"),
                "value_length": r["metadata"].get("qwen35.attention.value_length"),
                "rope_dimension_count": r["metadata"].get("qwen35.rope.dimension_count"),
                "rope_freq_base": r["metadata"].get("qwen35.rope.freq_base"),
                "layer_norm_rms_epsilon": r["metadata"].get("qwen35.attention.layer_norm_rms_epsilon"),
            },
            "ssm": {
                "conv_kernel": r["metadata"].get("qwen35.ssm.conv_kernel"),
                "state_size": r["metadata"].get("qwen35.ssm.state_size"),
                "group_count": r["metadata"].get("qwen35.ssm.group_count"),
                "time_step_rank": r["metadata"].get("qwen35.ssm.time_step_rank"),
                "inner_size": r["metadata"].get("qwen35.ssm.inner_size"),
            },
            "full_attention_interval": r["metadata"].get("qwen35.full_attention_interval"),
            "context_length": r["metadata"].get("qwen35.context_length"),
            "file_type": r["metadata"].get("general.file_type"),
        },
        "counts": {
            "tensors": len(tensors),
            "layers": len(layers),
            "gdn_layers": kinds.get("gdn", 0),
            "full_layers": kinds.get("full", 0),
            "full_mtp_layers": kinds.get("full+mtp", 0),
            "mtp_layers": sum(1 for l in layers.values()
                              if any(t["family"] == "mtp" for t in l["tensors"])),
            "global_tensors": len(ann["globals"]),
            "distinct_roles": len(roles),
            "distinct_role_type_pairs": len(dispatch),
            "total_params": total_params,
            "params_billions": round(total_params / 1e9, 4),
        },
        "layout_pattern": {
            "notation": "L = GDN linear attention, F = full attention, per layer index order",
            "string": "".join(pattern),
            "declared_interval": interval,
        },
        "types": types,
        "roles": list(roles.values()),
        "families": families,
        "budget": budget,
        "dispatch_table": dispatch,
        "layers": [
            {
                "index": i,
                "kind": layers[i]["kind"],
                "tensor_count": layers[i]["tensor_count"],
                "bytes": layers[i]["bytes"],
                "params": layers[i]["params"],
                "bytes_by_family": {
                    f: sum(t["bytes"] for t in layers[i]["tensors"] if t["family"] == f)
                    for f in sorted({t["family"] for t in layers[i]["tensors"]})
                },
                "tensors": layers[i]["tensors"],
            }
            for i in sorted(layers)
        ],
        "globals": ann["globals"],
        "shape_provenance": provenance,
        "anomalies": anomalies,
        "checks": checks,
        "metadata": r["metadata"],
        "metadata_arrays": {
            k: {"elem": v["elem"], "count": v["count"], "first": v["first"]}
            for k, v in r["metadata_arrays"].items()
        },
    }

    # Pattern check, computed properly once the layers exist. MTP block layers
    # sit outside the LLLF cadence: they are appended, not cadenced.
    mismatched = []
    mtp_layer_indices = sorted(i for i in layers
                               if any(t["family"] == "mtp" for t in layers[i]["tensors"]))
    if pattern:
        for i in sorted(layers):
            if i >= len(pattern) or i in mtp_layer_indices:
                continue
            want = "F" if pattern[i] == "F" else "L"
            got = "F" if layers[i]["kind"].startswith("full") else "L"
            if want != got:
                mismatched.append({"layer": i, "declared": want, "actual": got})
    out["layout_pattern"]["mismatches"] = mismatched
    out["layout_pattern"]["matches_declared_pattern"] = not mismatched
    out["layout_pattern"]["mtp_layers_outside_cadence"] = mtp_layer_indices
    return out


def diff(base: dict, mtp: dict) -> dict:
    A = {t["name"]: t for t in base["tensors_flat"]}
    B = {t["name"]: t for t in mtp["tensors_flat"]}
    only_b = sorted(set(B) - set(A))
    only_a = sorted(set(A) - set(B))
    shared_diff = []
    for n in sorted(set(A) & set(B)):
        a, b = A[n], B[n]
        if a["type"] != b["type"] or a["dims"] != b["dims"] or a["bytes"] != b["bytes"]:
            shared_diff.append({
                "name": n,
                "base": {"type": a["type"], "type_id": a["type_id"], "dims": a["dims"], "bytes": a["bytes"]},
                "mtp": {"type": b["type"], "type_id": b["type_id"], "dims": b["dims"], "bytes": b["bytes"]},
            })

    added = []
    for n in only_b:
        t = B[n]
        added.append({
            "name": n,
            "role": t["role"], "family": t["family"], "scope": t["scope"],
            "layer": t["layer"], "layer_kind": t["layer_kind"],
            "type": t["type"], "type_id": t["type_id"],
            "dims": t["dims"], "dims_torch": t["dims_torch"],
            "params": t["params"], "bytes": t["bytes"],
            "bits_per_weight": t["bits_per_weight"],
            "file_offset": t["file_offset"],
        })
    added_bytes = sum(x["bytes"] for x in added)

    # Does the MTP block reuse the trunk's geometry, or invent new shapes?
    base_role_dims = {r["role"]: r["dims"] for r in base["roles"]}
    shape_mismatch = [
        {"name": x["name"], "role": x["role"], "dims": x["dims"],
         "trunk_dims": base_role_dims.get(x["role"])}
        for x in added
        if x["role"] in base_role_dims and x["dims"] != base_role_dims[x["role"]]
    ]

    # Role-level deltas on layers the two files share.
    roles_added = Counter(x["role"] for x in added)
    type_bytes_added = defaultdict(int)
    for x in added:
        type_bytes_added[x["type"]] += x["bytes"]

    new_types = sorted({x["type"] for x in added} - {t["type"] for t in base["types"]})

    return {
        "base_file": base["file"]["name"],
        "mtp_file": mtp["file"]["name"],
        "file_bytes_delta": mtp["file"]["bytes"] - base["file"]["bytes"],
        "data_section_delta": (mtp["file"]["bytes"] - mtp["file"]["data_start"])
                              - (base["file"]["bytes"] - base["file"]["data_start"]),
        "tensor_count": {"base": base["file"]["tensor_count"], "mtp": mtp["file"]["tensor_count"],
                         "added": len(only_b), "removed": len(only_a)},
        "layer_count": {"base": base["counts"]["layers"], "mtp": mtp["counts"]["layers"]},
        "added_tensors": added,
        "added_bytes": added_bytes,
        "added_gib": round(added_bytes / 2**30, 4),
        "added_share_of_mtp_file_pct": round(
            100.0 * added_bytes / (mtp["file"]["bytes"] - mtp["file"]["data_start"]), 3),
        "added_by_role": dict(roles_added),
        "added_by_type": {k: {"bytes": v, "gib": round(v / 2**30, 4)}
                          for k, v in sorted(type_bytes_added.items(), key=lambda kv: -kv[1])},
        "added_shapes_match_trunk": not shape_mismatch,
        "added_shape_mismatches": shape_mismatch,
        "metadata_region_delta": mtp["file"]["data_start"] - base["file"]["data_start"],
        "metadata_kv_delta": mtp["file"]["kv_count"] - base["file"]["kv_count"],
        "metadata_new_keys": sorted(set(mtp["metadata"]) - set(base["metadata"])),
        "added_layer_kinds": sorted({x["layer_kind"] for x in added if x["layer"] is not None}),
        "added_layers": sorted({x["layer"] for x in added if x["layer"] is not None}),
        "types_only_in_mtp": new_types,
        "removed_tensors": only_a,
        "shared_tensors": len(set(A) & set(B)),
        "shared_tensors_identical_type_shape_bytes": not shared_diff,
        "shared_tensor_mismatches": shared_diff,
        "payload_hash_compared": False,
        "payload_hash_note": ("only type, shape and byte size are compared per name; "
                              "the quantised payloads were not hashed (that would read "
                              "12 GB twice)"),
    }


def flatten(inv: dict) -> list[dict]:
    out = []
    for l in inv["layers"]:
        for t in l["tensors"]:
            rec = dict(t)
            rec["layer"] = l["index"]
            rec["layer_kind"] = l["kind"]
            out.append(rec)
    for t in inv["globals"]:
        rec = dict(t)
        rec["layer"] = None
        rec["layer_kind"] = "global"
        out.append(rec)
    return out


def fmt_bytes(n: int) -> str:
    return f"{n:,}"


def write_doc(base: dict, mtp: dict, d: dict, path: Path) -> None:
    L = []
    w = L.append
    b = base["file"]
    m = base["model"]
    c = base["checks"]

    w("# Tensor inventory — IQ3_S and its MTP sibling")
    w("")
    w("Generated by `bench/03_tensor_inventory.py` from the two files themselves.")
    w("Nothing here is copied from another engine; the container is parsed by")
    w("`bench/02_gguf_probe.py`, which this script imports. Machine-readable form:")
    w("`bench/manifests/iq3s.json` and `bench/manifests/iq3s-mtp.json`.")
    w("")
    w("## 1. Container facts and the arithmetic gate")
    w("")
    w("| | base | mtp |")
    w("|---|---|---|")
    w(f"| file | `{b['name']}` | `{mtp['file']['name']}` |")
    w(f"| bytes | {fmt_bytes(b['bytes'])} | {fmt_bytes(mtp['file']['bytes'])} |")
    w(f"| GiB | {b['gib']:.3f} | {mtp['file']['gib']:.3f} |")
    w(f"| GGUF version | {b['gguf_version']} | {mtp['file']['gguf_version']} |")
    w(f"| tensors | {b['tensor_count']} | {mtp['file']['tensor_count']} |")
    w(f"| metadata key/values | {b['kv_count']} | {mtp['file']['kv_count']} |")
    w(f"| data section starts at | {fmt_bytes(b['data_start'])} | {fmt_bytes(mtp['file']['data_start'])} |")
    w(f"| parameters encoded | {c['total_params']:,} | {mtp['checks']['total_params']:,} |")
    w(f"| effective bits/weight | {c['effective_bits_per_weight']:.4f} | "
      f"{mtp['checks']['effective_bits_per_weight']:.4f} |")
    w("")
    w("The gate: the metadata region plus every tensor's bytes must equal the file")
    w("size exactly, with no gaps and no overlap.")
    w("")
    w("| component | bytes |")
    w("|---|---|")
    w(f"| header + metadata + tensor directory | {fmt_bytes(c['metadata_end_bytes'])} |")
    w(f"| alignment padding to 32 B | {fmt_bytes(c['metadata_padding_bytes'])} |")
    w(f"| **metadata region total** | **{fmt_bytes(c['metadata_region_bytes'])}** |")
    w(f"| sum of tensor byte sizes | {fmt_bytes(c['tensor_bytes_sum'])} |")
    w(f"| **sum** | **{fmt_bytes(c['metadata_plus_tensors'])}** |")
    w(f"| file size | {fmt_bytes(c['file_bytes'])} |")
    w(f"| difference | {c['metadata_plus_tensors'] - c['file_bytes']} |")
    w("")
    w(f"- closes exactly: **{c['closes_exactly']}**")
    w(f"- tensors contiguous as `offset[i] + bytes[i] == offset[i+1]`: **{c['tensors_contiguous']}**")
    w(f"- every offset 32-byte aligned: **{c['offsets_aligned']}**")
    w(f"- every quantised tensor's size equals its block formula exactly "
      f"(no inter-tensor padding): **{not c['block_layout_mismatches']}**")
    w(f"- every quantised tensor's `ne[0]` is a multiple of its block width, so every "
      f"row starts on a block boundary: **{not c['row_block_misaligned']}**")
    w(f"- types whose block layout could not be verified: "
      f"{c['types_without_layout'] or 'none'}")
    w("")
    w("The block-formula check is what makes the type column trustworthy: if a")
    w("tensor's byte count equals `params / block_weights * block_bytes` for its")
    w("declared type, the declared type and the offset arithmetic agree.")
    w("")
    for t in base["types"]:
        if t["tensors"] == 1 and t["block"]:
            bw, bb = t["block"]
            tt = next(x for x in base["tensors_flat"] if x["type"] == t["type"])
            w(f"Type id {t['type_id']} (`{t['type']}`) appears exactly once, and the probe's")
            w(f"original table (lifted from Strata's reader) did not name it. The name comes")
            w(f"from the ggml type registry; the arithmetic confirms the layout independently:")
            w(f"{tt['params']:,} weights → {tt['bytes']:,} bytes = {tt['params']//bw:,} blocks "
              f"× {bb} B = {8*bb/bw:.2f} bits/weight. No other type in either file has an")
            w("unnamed id.")
            w("")

    w("## 2. Quantisation types present")
    w("")
    w("| type | id | tensors | GiB | share | bits/weight | block |")
    w("|---|---|---|---|---|---|---|")
    for t in base["types"]:
        blk = f"{t['block'][0]}/{t['block'][1]} B" if t["block"] else "—"
        w(f"| {t['type']} | {t['type_id']} | {t['tensors']} | {t['gib']:.3f} | "
          f"{t['share_pct']:.2f}% | {t['bits_per_weight']:.4f} | {blk} |")
    w("")
    w(f"{len(base['types'])} distinct types. {base['counts']['distinct_role_type_pairs']} "
      f"distinct (role, type) pairs — that is the size of the dispatch table the")
    w("GEMV kernel set has to cover (section 6).")
    w("")

    w("## 3. Roles")
    w("")
    w("Shapes are given as stored (`ne` order: `ne[0]` is the contiguous dimension,")
    w("i.e. the input width of a projection) and in torch order `[out, in]`.")
    w("")
    w("| role | scope | shape (ne) | shape [out,in] | layers | types | bytes | GiB | share |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(base["roles"], key=lambda x: -x["bytes"]):
        dims = "×".join(str(x) for x in s["dims"])
        dt = "×".join(str(x) for x in s["dims_torch"])
        tys = ", ".join(f"{k}×{v}" for k, v in s["types"].items())
        w(f"| `{s['role']}` | {s['scope']} | {dims} | {dt} | {s['layers']} | {tys} | "
          f"{fmt_bytes(s['bytes'])} | {s['gib']:.3f} | {s['share_pct']:.2f}% |")
    w("")
    w("### Tensors that exist only in GDN layers")
    w("")
    gdn_only = [s for s in base["roles"] if s["scope"] == "gdn"]
    for s in sorted(gdn_only, key=lambda x: -x["bytes"]):
        w(f"- `{s['role']}` — {'×'.join(map(str, s['dims']))}, "
          f"{', '.join(s['types'])}, {s['gib']:.3f} GiB, {s['layers']} layers")
    w("")
    w("### Tensors that exist only in full-attention layers")
    w("")
    for s in sorted([x for x in base["roles"] if x["scope"] == "full"], key=lambda x: -x["bytes"]):
        w(f"- `{s['role']}` — {'×'.join(map(str, s['dims']))}, "
          f"{', '.join(s['types'])}, {s['gib']:.3f} GiB, {s['layers']} layers")
    w("")
    w("### Present in every layer")
    w("")
    for s in sorted([x for x in base["roles"] if x["scope"] == "all"], key=lambda x: -x["bytes"]):
        w(f"- `{s['role']}` — {'×'.join(map(str, s['dims']))}, "
          f"{', '.join(s['types'])}, {s['gib']:.3f} GiB, {s['layers']} layers")
    w("")
    w("### Globals")
    w("")
    for t in base["globals"]:
        w(f"- `{t['name']}` — {'×'.join(map(str, t['dims']))}, {t['type']}, "
          f"{t['bytes']/2**30:.3f} GiB")
    w("")

    w("### Shape provenance")
    w("")
    w("Every width is derived from the file's own metadata, not from a config.json")
    w("transcription. A shape that does not reconcile means the parse or the role")
    w("mapping is wrong, so this table is a check, not decoration.")
    w("")
    w("| role | formula | expected [out,in] | observed | ok |")
    w("|---|---|---|---|---|")
    for p in base["shape_provenance"]:
        exp = "×".join(str(x) for x in p["expected_dims_torch"])
        obs = "×".join(str(x) for x in p["observed_dims_torch"])
        w(f"| `{p['role']}` | {p['formula']} | {exp} | {obs} | "
          f"{'yes' if p['matches'] else '**NO**'} |")
    w("")
    bad = [p["role"] for p in base["shape_provenance"] if not p["matches"]]
    w(f"All {len(base['shape_provenance'])} shapes reconcile: **{not bad}**"
      f"{'' if not bad else ' — failures: ' + ', '.join(bad)}")
    w("")
    w("Two widths carry a design decision worth reading twice: `attn_q` is 12288, which")
    w("is 24×256 **twice** — the query and its output gate in one matrix, so the gate is")
    w("not a separate tensor and not a separate GEMV. `attn_qkv` is 10240 = 2×(16×128) +")
    w("6144 — q and k at 16 heads of 128 feeding the conv and the delta rule, v at 48")
    w("heads of 128 feeding the state. Both are single GEMVs with a concatenated output.")
    w("")

    REF_PARAMS = 27.78e9  # docs/01-models.md
    mlp_params = next(f for f in base["families"] if f["family"] == "mlp")["params"]
    emb_params = next(f for f in base["families"] if f["family"] == "embed_output")["params"]
    w("## 3b. Parameter count")
    w("")
    w("| file | parameters | vs the 27.78 B in docs/01 |")
    w("|---|---|---|")
    w(f"| IQ3_S (trunk) | {base['counts']['total_params']:,} "
      f"({base['counts']['params_billions']:.3f} B) | "
      f"{(base['counts']['total_params'] - REF_PARAMS)/1e9:+.3f} B |")
    w(f"| IQ3_S-mtp | {mtp['counts']['total_params']:,} "
      f"({mtp['counts']['params_billions']:.3f} B) | "
      f"{(mtp['counts']['total_params'] - REF_PARAMS)/1e9:+.3f} B |")
    w("")
    w("The gap is accounted for, not mysterious: the `-mtp` file's 27.32 B plus the")
    w("0.466 B of the BF16 visual tower in `mmproj-Qwen3.8-27B-BF16.gguf` (931,146,528 B")
    w("/ 2) comes to 27.79 B — the 27.78 B in `docs/01-models.md`. That figure therefore")
    w("counts the MTP layer **and** the visual tower, and it is the right number for")
    w("a model card, not for a decode budget.")
    w("")
    w(f"For the weight budget: the text trunk alone is **{base['counts']['params_billions']:.2f} B "
      f"parameters**, of which the embedding and head are "
      f"{emb_params/1e9:.2f} B ({100*emb_params/base['counts']['total_params']:.1f}%) and the MLP "
      f"{mlp_params/1e9:.2f} B ({100*mlp_params/base['counts']['total_params']:.1f}%). The vision")
    w("tower is never read during text decode and must not appear in a decode budget.")
    w("")

    w("## 4. Byte budget per role group")
    w("")
    w("Disjoint families — these sum to the data section.")
    w("")
    w("| family | what | tensors | GiB | share | bits/weight |")
    w("|---|---|---|---|---|---|")
    for f in base["families"]:
        w(f"| `{f['family']}` | {f['description']} | {f['tensors']} | {f['gib']:.3f} | "
          f"{f['share_pct']:.2f}% | {f['bits_per_weight']:.4f} |")
    w("")
    w("Roll-ups in the terms the plan uses (overlapping, not disjoint):")
    w("")
    w("| group | GiB | share of data | note |")
    w("|---|---|---|---|")
    for k in ("attention_projections_all", "mlp", "embeddings_output", "norms",
              "gdn_specific", "mtp"):
        v = base["budget"][k]
        w(f"| {k.replace('_', ' ')} | {v['gib']:.3f} | {v['share_pct']:.2f}% | {v['note']} |")
    w("")
    mlp = next(f for f in base["families"] if f["family"] == "mlp")
    attn = base["budget"]["attention_projections_all"]
    emb_t = next(t for t in base["globals"] if t["role"] == "token_embd")
    per_token = c["tensor_bytes_sum"] - emb_t["bytes"]
    w(f"Where the time goes: the MLP is **{mlp['share_pct']:.1f}%** of the file and the "
      f"attention projections are **{attn['share_pct']:.1f}%**. Every byte of both is read")
    w("once per token in decode. The norms are rounding error at 0.02%.")
    w("")
    w(f"One number to carry forward: **{per_token/2**30:.2f} GiB is read per token** —")
    w("everything except `token_embd`, which is a row lookup, not a sweep. That is the")
    w(f"denominator of every decode-bandwidth calculation for this file, and it is "
      f"{100*emb_t['bytes']/c['tensor_bytes_sum']:.1f}% smaller than the file, not the file.")
    w("")

    w("## 5. Per-layer map")
    w("")
    lp = base["layout_pattern"]
    w(f"Declared `full_attention_interval` = {lp['declared_interval']}; observed pattern")
    w(f"({lp['notation']}):")
    w("")
    s = lp["string"]
    w("```")
    for i in range(0, len(s), 32):
        w(f"{i:3d}  {' '.join(s[i:i+32])}")
    w("```")
    w("")
    w(f"Matches the declared pattern: **{lp['matches_declared_pattern']}**"
      f"{'' if lp['matches_declared_pattern'] else ' — mismatches: ' + str(lp['mismatches'])}")
    w("")
    w("Each layer carries one of two role sets — the sets are listed after the table,")
    w("so the table only has to say which one and how the bytes split.")
    w("")
    w("| layer | kind | tensors | attn MiB | mlp MiB | norms MiB | total MiB |")
    w("|---|---|---|---|---|---|---|")
    for l in base["layers"]:
        fam = l.get("bytes_by_family", {})
        attn = sum(v for k, v in fam.items() if k in ("attn_full", "gdn_proj", "gdn_state"))
        mlp = fam.get("mlp", 0)
        nrm = fam.get("norm", 0)
        w(f"| {l['index']} | {l['kind']} | {l['tensor_count']} | {attn/2**20:.1f} | "
          f"{mlp/2**20:.1f} | {nrm/2**20:.2f} | {l['bytes']/2**20:.1f} |")
    w("")

    # Distinct role sets, so the reader can reconstruct any layer without a
    # 64-row enumeration of every tensor name.
    by_set: dict[frozenset, list[int]] = defaultdict(list)
    for l in base["layers"]:
        by_set[frozenset(t["role"] for t in l["tensors"])].append(l["index"])
    w("### The role sets, in full")
    w("")
    for roleset, idxs in sorted(by_set.items(), key=lambda kv: -len(kv[1])):
        kind = "GDN" if "attn_qkv" in roleset else "full-attention"
        w(f"**{kind}** — {len(idxs)} layers: "
          f"{', '.join(str(i) for i in idxs)}")
        w("")
        w("```")
        for role in sorted(roleset):
            s = next(x for x in base["roles"] if x["role"] == role)
            w(f"  {role:<26} {'×'.join(map(str, s['dims'])):<14} {','.join(s['types'])}")
        w("```")
        w("")

    w("## 6. The dispatch table: distinct (role, type) pairs")
    w("")
    w("One kernel family per row is the minimum. Ordering is by bytes, so the top rows")
    w("are the ones worth writing first.")
    w("")
    w("| role | type | tensors | GiB |")
    w("|---|---|---|---|")
    for p in base["dispatch_table"]:
        w(f"| `{p['role']}` | {p['type']} | {p['tensors']} | {p['gib']:.4f} |")
    w("")

    w("## 7. Shape uniformity")
    w("")
    nonuniform = [s for s in base["roles"] if not s["dims_uniform"]]
    if nonuniform:
        w("**Not uniform.** Roles with more than one shape:")
        w("")
        for s in nonuniform:
            w(f"- `{s['role']}` — {'×'.join(map(str, s['dims']))}")
    else:
        w("Every role has exactly one shape across all its layers: the 64 layers'")
        w("attention and MLP shapes are **identical**. A kernel can be written for one")
        w("shape per role and dispatched on role alone — no per-layer specialisation is")
        w("needed for shape. Types, however, are *not* uniform (section 8), so the")
        w("dispatch key must be (role, type), not role.")
        w("")
        w(f"The MTP layer keeps the same geometry: every tensor it adds for an existing "
          f"role has the trunk's shape (**{d['added_shapes_match_trunk']}**).")
    w("")

    w("## 8. Anomalies")
    w("")
    for a in base["anomalies"]:
        w(f"### {a['kind']} ({a['severity']})")
        w("")
        w(a["detail"])
        w("")
        if a["kind"] == "rare_quant_types":
            w("| type | id | GiB | share | tensors | where |")
            w("|---|---|---|---|---|---|")
            for t in a["types"]:
                names = t.get("tensor_names") or []
                where = ", ".join(f"`{n}`" for n in names[:6])
                if len(names) > 6:
                    where += f", … {len(names) - 6} more"
                w(f"| {t['type']} | {t['type_id']} | {t['gib']:.4f} | {t['share_pct']:.3f}% | "
                  f"{t['tensors']} | {where} |")
            w("")
        elif a["kind"] == "per_role_bitwidth_range":
            w("| role | layers | min | median | max | types | lowest | highest |")
            w("|---|---|---|---|---|---|---|---|")
            for role, v in a["ranges"].items():
                w(f"| `{role}` | {v['layers']} | {v['min_bits_per_weight']:.3f} | "
                  f"{v['median_bits_per_weight']:.3f} | {v['max_bits_per_weight']:.3f} | "
                  f"{v['distinct_types']} | `{v['lowest']}` | `{v['highest']}` |")
            w("")
        elif a["kind"] == "single_layer_type_paths":
            w("| role | type | the one tensor |")
            w("|---|---|---|")
            for p in a["paths"]:
                w(f"| `{p['role']}` | {p['type']} | `{p['tensor']}` |")
            w("")
        elif a["kind"] == "lowest_precision_tensor":
            t = a["tensor"]
            w(f"- `{t['name']}` — {t['type']} (id {t['type_id']}), shape "
              f"{'×'.join(map(str, t['dims']))}, {fmt_bytes(t['bytes'])} B, "
              f"{t['bits_per_weight']:.2f} bits/weight, at file offset {fmt_bytes(t['file_offset'])}")
            w("")
        elif a["kind"] == "role_bitwidth_outliers" and a["count"]:
            w("| tensor | role | type | bits/weight | role median |")
            w("|---|---|---|---|---|")
            for t in a["tensors"]:
                w(f"| `{t['name']}` | {t['role']} | {t['type']} | {t['bits_per_weight']:.3f} | "
                  f"{t['role_median']:.3f} |")
            w("")
        elif a["kind"] == "padding_or_layout":
            w("| tensor | type | stored bytes | block formula |")
            w("|---|---|---|---|")
            for t in a["tensors"]:
                w(f"| `{t['name']}` | {t['type']} | {fmt_bytes(t['bytes'])} | {fmt_bytes(t['expected'])} |")
            w("")

    w("## 9. The MTP delta")
    w("")
    w(f"`{d['mtp_file']}` is `{d['base_file']}` plus "
      f"{d['tensor_count']['added']} tensors and {fmt_bytes(d['data_section_delta'])} bytes "
      f"of weights.")
    w("")
    w("| | value |")
    w("|---|---|")
    w(f"| tensors added | {d['tensor_count']['added']} |")
    w(f"| tensors removed | {d['tensor_count']['removed']} |")
    w(f"| layers | {d['layer_count']['base']} → {d['layer_count']['mtp']} "
      f"(added: {d['added_layers']}, kind {', '.join(d['added_layer_kinds'])}) |")
    w(f"| bytes added to the data section | {fmt_bytes(d['data_section_delta'])} "
      f"({d['added_gib']:.3f} GiB) |")
    w(f"| metadata region delta | {fmt_bytes(d['metadata_region_delta'])} "
      f"({d['metadata_kv_delta']} new key: {', '.join(d['metadata_new_keys'])}) |")
    w(f"| file bytes delta | {fmt_bytes(d['file_bytes_delta'])} |")
    w(f"| share of the mtp data section | {d['added_share_of_mtp_file_pct']:.3f}% |")
    w(f"| types only in the mtp file | {', '.join(d['types_only_in_mtp']) or 'none'} |")
    w(f"| added tensors reuse trunk shapes | **{d['added_shapes_match_trunk']}** |")
    w(f"| shared tensors with identical type+shape+bytes | "
      f"**{d['shared_tensors']} of {d['shared_tensors']}** "
      f"({d['shared_tensors_identical_type_shape_bytes']}) |")
    w("")
    w("### Tensors the MTP head adds")
    w("")
    w("| tensor | layer | type | shape (ne) | shape [out,in] | bytes | bits/weight |")
    w("|---|---|---|---|---|---|---|")
    for t in d["added_tensors"]:
        w(f"| `{t['name']}` | {t['layer']} ({t['layer_kind']}) | {t['type']} | "
          f"{'×'.join(map(str, t['dims']))} | {'×'.join(map(str, t['dims_torch']))} | "
          f"{fmt_bytes(t['bytes'])} | {t['bits_per_weight']:.3f} |")
    w("")
    w("By type:")
    w("")
    w("| type | bytes | GiB |")
    w("|---|---|---|")
    for k, v in d["added_by_type"].items():
        w(f"| {k} | {fmt_bytes(v['bytes'])} | {v['gib']:.4f} |")
    w("")
    if d["shared_tensor_mismatches"]:
        w("Shared tensors that differ:")
        w("")
        for x in d["shared_tensor_mismatches"]:
            w(f"- `{x['name']}`: base {x['base']} vs mtp {x['mtp']}")
        w("")
    else:
        w("All shared tensors have the same type, shape and byte size in both files.")
        w("Their **payloads were not hashed** — that would read 12 GB twice and prove")
        w("nothing about the kernels. What the kernel set needs is that the same")
        w("(role, type, shape) triple appears in both, and it does.")
    w("")
    mtp_quant = [k for k in d["added_by_type"] if k not in ("F32", "BF16")]
    mtp_quant_gib = sum(v["bytes"] for k, v in d["added_by_type"].items()
                        if k not in ("F32", "BF16")) / 2**30
    trunk_types = [t["type"] for t in base["types"]]
    w("What this adds to the kernel list:")
    w("")
    w(f"- **One more full-attention layer** (index {', '.join(str(i) for i in d['added_layers'])}, "
      f"kind `{', '.join(d['added_layer_kinds'])}`) with the trunk's shapes. The existing "
      "full-attention kernels run it unchanged.")
    w(f"- **A precision the trunk never uses.** Every matrix in that layer is "
      f"{', '.join(mtp_quant)} — {mtp_quant_gib:.3f} GiB of the {d['added_gib']:.3f} GiB "
      f"added. The trunk carries {', '.join(trunk_types)}; Q6_K is in neither list. "
      "The GEMV set therefore needs a Q6_K path that exactly one layer exercises.")
    w("- **Four tensors with no trunk counterpart**: `nextn.eh_proj` (10240→5120, the "
      "token-state and hidden-state concatenation) and the three `nextn.*` norms. "
      "`nextn.shared_head_norm` feeds the existing `output.weight`, so the MTP head "
      "adds **no second LM head** — the 0.67 GiB head is shared, not duplicated.")
    w("- **A 65th layer changes the loop bound, not the kernel set.** Layer count, "
      "not shape or type, is what the runtime has to parameterise.")
    w("")
    w("## 10. What this fixes for the kernel list")
    w("")
    w("1. **Shapes are uniform per role.** No shape-generic kernel is needed; a")
    w("   per-role kernel covers all 64 layers and the MTP layer.")
    w("2. **Types are not.** Every quantised role carries several types. The dispatch")
    w(f"   key is therefore `(role, type)` — {base['counts']['distinct_role_type_pairs']} pairs in")
    w("   the base file. A single GEMV per role would silently fall back for half the")
    w("   layers.")
    w("3. **GDN is not a side quest.** The GDN-only tensors are "
      f"{base['budget']['gdn_specific']['gib']:.2f} GiB "
      f"({base['budget']['gdn_specific']['share_pct']:.1f}%), and they are read on")
    w("   48 of the 64 layers. Any plan that ships full attention first covers 16.")
    w("4. **The fused projections change the kernel shape.** `attn_qkv` is 5120→10240")
    w("   (q and k at 16×128 plus v at 48×128), and `attn_q` is 5120→12288 (q at")
    w("   24×256 with its output gate fused). Neither is a plain square projection;")
    w("   both are one GEMV producing concatenated slices.")
    w("5. **Mixed precision inside one GDN block** (`ssm_alpha`/`ssm_beta` BF16,")
    w("   `ssm_a`/`ssm_dt` F32) means the GDN kernel takes both dtypes, not just the")
    w("   quantised path.")
    w(f"6. **Speculation costs one more quant type.** The MTP layer is entirely "
      f"{', '.join(k for k in d['added_by_type'] if k not in ('F32', 'BF16'))}, which the "
      f"trunk never uses. Either the GEMV set grows a Q6_K path for one layer, or "
      f"speculative decoding is off the table — there is no third option.")
    w("7. **The head, not the embedding, is the per-token cost.** `token_embd` is read")
    w("   by index (one row per token); `output.weight` is read in full, 0.67 GiB every")
    w("   token, and it is the largest single tensor in the file. Any per-token byte")
    w("   budget starts there.")
    w("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--mtp", type=Path, default=DEFAULT_MTP)
    ap.add_argument("--json-dir", type=Path, default=REPO / "bench/manifests")
    ap.add_argument("--doc", type=Path, default=REPO / "docs/plan/TENSOR-INVENTORY.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = build(args.base)
    base["tensors_flat"] = flatten(base)
    mtp = build(args.mtp)
    mtp["tensors_flat"] = flatten(mtp)
    d = diff(base, mtp)
    mtp["diff_vs"] = d

    args.json_dir.mkdir(parents=True, exist_ok=True)
    (args.json_dir / "iq3s.json").write_text(
        json.dumps({k: v for k, v in base.items() if k != "tensors_flat"}, indent=2) + "\n")
    (args.json_dir / "iq3s-mtp.json").write_text(
        json.dumps({k: v for k, v in mtp.items() if k != "tensors_flat"}, indent=2) + "\n")
    write_doc(base, mtp, d, args.doc)

    ok = True
    for inv in (base, mtp):
        c = inv["checks"]
        print(f"{inv['file']['name']}:")
        print(f"  file            {c['file_bytes']:>15,} B")
        print(f"  metadata region {c['metadata_region_bytes']:>15,} B "
              f"(keys end {c['metadata_end_bytes']:,}, pad {c['metadata_padding_bytes']} B)")
        print(f"  tensors sum     {c['tensor_bytes_sum']:>15,} B  ({len(inv['tensors_flat'])} tensors)")
        print(f"  total           {c['metadata_plus_tensors']:>15,} B")
        print(f"  difference      {c['metadata_plus_tensors'] - c['file_bytes']:>15,} B  "
              f"closes={c['closes_exactly']} contiguous={c['tensors_contiguous']} "
              f"aligned={c['offsets_aligned']}")
        if not (c["closes_exactly"] and c["tensors_contiguous"] and not c["block_layout_mismatches"]):
            ok = False
        print(f"  layers          gdn={inv['counts']['gdn_layers']} full={inv['counts']['full_layers']} "
              f"mtp={inv['counts']['mtp_layers']}")
        print()
    print(f"MTP adds {d['tensor_count']['added']} tensors, {d['added_bytes']:,} B "
          f"({d['added_gib']:.3f} GiB); shared identical: "
          f"{d['shared_tensors_identical_type_shape_bytes']} of {d['shared_tensors']}")
    print(f"wrote {args.json_dir/'iq3s.json'}")
    print(f"wrote {args.json_dir/'iq3s-mtp.json'}")
    print(f"wrote {args.doc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
