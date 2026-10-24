#!/usr/bin/env python3
"""Unit 0.1 — diff the GGUF's per-tensor types against the publisher's RCO allocation.

Both sides are read from primary artifacts: the GGUF tensor table, and the
`*.rco-allocation.txt` the publisher ships next to the weights. No third source.

  proof:            python3 bench/04_recipe_diff.py <gguf> <rco-allocation.txt>
  negative control: python3 bench/04_recipe_diff.py <gguf> <rco> --mutate blk.0.attn_q.weight=Q4_K

Exit 0 when every tensor matches, 1 otherwise. `--mutate` corrupts one expected
type first, so a run that passes with it is a diff that cannot fail.
"""
import struct
import sys

GGML = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}


def read_gguf_tensor_types(path):
    """name -> GGML type name, from the GGUF tensor table."""
    f = open(path, "rb")
    read = lambda fmt, n: struct.unpack("<" + fmt, f.read(n))
    if f.read(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version, = read("I", 4)
    n_tensors, = read("Q", 8)
    n_kv, = read("Q", 8)

    def rstr():
        n, = read("Q", 8)
        return f.read(n).decode("utf-8", "replace")

    def skip_value(t):
        if t == 8:            # string
            rstr()
        elif t == 9:          # array
            et, = read("I", 4)
            n, = read("Q", 8)
            for _ in range(n):
                skip_value(et)
        else:
            f.read(struct.calcsize("<" + SCALAR[t]))

    for _ in range(n_kv):
        rstr()
        t, = read("I", 4)
        skip_value(t)

    out = {}
    for _ in range(n_tensors):
        name = rstr()
        nd, = read("I", 4)
        f.read(8 * nd)        # dims
        tt, = read("I", 4)
        f.read(8)             # offset
        out[name] = GGML.get(tt, f"type{tt}")
    f.close()
    return version, out


def read_rco(path):
    """name -> type, plus the header's declared per-type counts."""
    out, counts = {}, {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            if "Quant-type counts:" in line:
                for part in line.split(":", 1)[1].split(","):
                    k, _, v = part.strip().partition("=")
                    if k and v.isdigit():
                        counts[k] = int(v)
            continue
        line = line.strip()
        if not line:
            continue
        name, _, typ = line.rpartition(":")
        out[name.strip()] = typ.strip()
    return out, counts


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    gguf_path, rco_path = argv[1], argv[2]
    mutate = None
    if "--mutate" in argv:
        spec = argv[argv.index("--mutate") + 1]
        name, _, typ = spec.partition("=")
        mutate = (name, typ)

    version, gguf = read_gguf_tensor_types(gguf_path)
    rco, declared = read_rco(rco_path)
    if mutate:
        if mutate[0] not in rco:
            print(f"NEGATIVE CONTROL BROKEN: {mutate[0]} is not in the allocation")
            return 2
        rco[mutate[0]] = mutate[1]
        print(f"negative control: forced {mutate[0]} = {mutate[1]}")

    print(f"GGUF v{version}: {len(gguf)} tensors   allocation: {len(rco)} tensors")

    # cross-check the allocation against its own header
    tally = {}
    for t in rco.values():
        tally[t] = tally.get(t, 0) + 1
    if declared and tally != declared:
        print(f"  allocation header disagrees with its own body: {declared} vs {tally}")

    only_gguf = sorted(set(gguf) - set(rco))
    only_rco = sorted(set(rco) - set(gguf))
    mismatch = [(n, gguf[n], rco[n]) for n in sorted(set(gguf) & set(rco)) if gguf[n] != rco[n]]

    for n in only_gguf[:5]:
        print(f"  only in GGUF: {n}")
    for n in only_rco[:5]:
        print(f"  only in allocation: {n}")
    for n, a, b in mismatch[:10]:
        print(f"  MISMATCH {n}: GGUF {a} vs allocation {b}")

    ok = not only_gguf and not only_rco and not mismatch
    print(f"matched {len(set(gguf) & set(rco))}/{len(rco)}  "
          f"only-GGUF {len(only_gguf)}  only-alloc {len(only_rco)}  mismatched {len(mismatch)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
