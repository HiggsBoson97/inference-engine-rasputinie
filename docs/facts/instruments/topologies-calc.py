#!/usr/bin/env python3
"""s4-topologies: every two-GPU strategy on rasputin, in bytes and milliseconds.

Constants from seed/docs/00-facts.md and research/03-bandwidth.md; W_TOK is the
parent's figure for the IQ3_S file.
"""
GiB = 1024**3
GB = 1000**3

W_TOK   = 10.57 * GiB      # weights read per token, IQ3_S  (11.35 GB)
B_HBM   = 776e9            # GB/s measured read wall
B_ENG   = 650e9            # GB/s engine-realistic
LINK    = 14.2 * GiB       # B/s cross-GPU == host bounce
HOST    = 46.42 * GiB      # B/s DDR5 copy
NVME    = 5.59 * GB        # B/s, 4 streams
HID     = 5120 * 2         # bf16 activation bytes
AR_MEAS = 20.65e-6         # s, measured 10 KiB all-reduce
AR_IDEA = 6e-6             # s, hand-written IPC reduce target
CARD    = 19.98 * GiB
LAYERS, ATTN_L, GDN_L = 64, 16, 48

t1 = W_TOK / B_HBM
budget = LINK * t1         # bytes the link can carry inside one on-card step
def m(x): return x*1e3

print("== constants ==")
print(f"W_TOK {W_TOK/1e9:.2f} GB = {W_TOK/GiB:.2f} GiB | 1 card {m(t1):.2f} ms = {1/t1:.1f} tok/s")
print(f"1 card @650 GB/s {m(W_TOK/B_ENG):.2f} ms = {1/(W_TOK/B_ENG):.1f} tok/s")
print(f"PCIe:HBM = 1:{B_HBM/LINK:.0f}; link is {100*LINK/B_HBM:.1f}% of one card")
print(f"hidden bf16 {HID} B; one handoff costs {HID/LINK*1e6:.2f} us of link time")
print(f"link budget inside one step = {budget/GiB:.3f} GiB\n")

# ---- per-layer param budget (derived from docs/01-models.md shapes)
mlp  = 3*5120*17408
att  = 5120*(24*256) + 5120*(4*256) + 5120*(4*256) + (24*256)*5120 + 5120*(24*256)
gdn  = 5120*(16*128) + 5120*(48*128) + (48*128)*5120 + 5120*(48*128)
print(f"params/layer: mlp {mlp/1e6:.1f}M  attn {att/1e6:.1f}M  gdn {gdn/1e6:.1f}M")
print(f"attn layer / gdn layer size ratio = {att/gdn:.3f}  (LLLF x16 -> split at 32 is exact)\n")

rows = []
def row(name, byt, step, note, lat=None, agg=None):
    rows.append((name, byt, step, agg if agg is not None else 1/step,
                 lat if lat is not None else step, note))

# ---- TP=2
tpl = W_TOK/LAYERS/2/B_HBM
for lbl, ar in (("TP=2, RCCL AR 20.65us", AR_MEAS), ("TP=2, hand AR 6us", AR_IDEA)):
    step = LAYERS*(tpl + 2*ar)
    row(lbl, 128*HID, step, f"128 collectives, {m(128*ar):.2f} ms of AR = "
        f"{100*128*ar/step:.0f}% of step; per card {W_TOK/2/GiB:.2f} GiB")
t_tp = LAYERS*(tpl + 2*AR_MEAS)
t_tp_fast = LAYERS*(tpl + 2*AR_IDEA)

# ---- DP=2
row("DP=2 two replicas", 0, t1, "zero link; per card 10.57 GiB of weights",
    lat=t1, agg=2*1/t1)

# ---- PP=2
t_s = (W_TOK/2)/B_HBM
HO  = 12e-6
row("PP=2, 1 stream (bubble)", HID, 2*t_s + HO, "one stage idle; == one card",
    lat=2*t_s+HO, agg=1/(2*t_s+HO))
row("PP=2, >=2 streams", HID, t_s, "both stages busy; 1 handoff/token; "
    f"per card {W_TOK/2/GiB:.2f} GiB", lat=2*t_s, agg=1/t_s)

# ---- hybrid: TP only the layers whose bytes repay the ARs
f_mlp = LAYERS*mlp/(LAYERS*mlp+ATTN_L*att+GDN_L*gdn)
f_att = ATTN_L*att/(LAYERS*mlp+ATTN_L*att+GDN_L*gdn)
b_mlp = 0.5*f_mlp*W_TOK + (1-f_mlp)*W_TOK
b_att = 0.5*f_att*W_TOK + (1-f_att)*W_TOK
row("Hybrid: TP the MLP only", LAYERS*2*HID, b_mlp/B_HBM + LAYERS*2*AR_MEAS,
    f"per card {b_mlp/GiB:.2f} GiB, all 128 ARs still paid")
row("Hybrid: TP attention only", ATTN_L*2*HID, b_att/B_HBM + ATTN_L*2*AR_MEAS,
    f"per card {b_att/GiB:.2f} GiB, only 32 ARs, saves 7% of bytes")

# ---- streaming
# every weight stream crosses PCIe at 14.2 GiB/s, whichever end it starts from
for name, b, bw in (("Stream weights from host DRAM (PCIe)", W_TOK, LINK),
                    ("Stream weights from NVMe", W_TOK, NVME),
                    ("PP + stream each stage from host", W_TOK/2, LINK)):
    dt = b/bw
    row(name, b, dt, f"{b/GiB:.1f} GiB over {bw/GiB:.1f} GiB/s")
# GPU+CPU split: the CPU half is read on the CPU, so DDR5 applies
cpu_b = (51.77 - 19.98) * GiB
row("GPU+CPU split (llama.cpp -ngl), BF16", cpu_b, cpu_b/HOST,
    f"{cpu_b/GiB:.1f} GiB on the CPU at {HOST/GiB:.1f} GiB/s, no PCIe")

# ---- fill table
hdr = f"{'strategy':34s} {'B/token':>9s} {'%of14.6ms':>10s} {'ival ms':>8s} {'agg t/s':>8s} {'lat ms':>7s}"
print(hdr); print("-"*len(hdr))
for n,b,s,a,l,note in rows:
    bs = "0" if b==0 else (f"{b/1024:.0f}KiB" if b<2**20 else f"{b/2**20:.1f}MiB")
    print(f"{n:34s} {bs:>9s} {100*(b/LINK)/t1:10.2f} {m(s):8.2f} {a:8.1f} {m(l):7.2f}")
print()
for n,b,s,a,l,note in rows:
    print(f"  {n:34s} {note}")

print("\n== KV: read once per token, wherever it lives ==")
KT = 64*1024
for ctx in (4096, 32768, 131072, 262144):
    kv = ctx*KT
    print(f" {ctx:>7} ctx  {kv/GiB:5.2f} GiB/sess  on-card {m(kv/B_HBM):6.2f} ms"
          f"  over link {m(kv/LINK):7.1f} ms  x{kv/LINK/(kv/B_HBM):.0f}"
          f"  = {100*(kv/B_HBM)/t1:4.0f}% of a 1-card step")

print("\n== stores: which memory is worth crossing the link for ==")
print(f" host DRAM {HOST/GiB:.1f} GiB/s / 60 GiB  vs  card1 {LINK/GiB:.1f} GiB/s / 20 GiB"
      f" -> host {HOST/LINK:.1f}x faster, 3x bigger")
print(f" NVMe {NVME/1e9:.2f} GB/s -> {LINK/(NVME):.1f}x slower than the link")
for lbl, bw in (("host", HOST), ("card1", LINK), ("nvme", NVME)):
    print(f"  159 MB recurrent state restore from {lbl:5s}: {m(0.159*GB/bw):6.2f} ms"
          f"  (re-prefill 2K tok @4600 tok/s = {m(2048/4600):.0f} ms)")

print("\n== fit envelope (19.98 GiB/card) ==")
for q, g in (("IQ3_S",10.57),("Q4_K_XL",16.35),("Q6_K",20.47),("Q8_0",27.05),("BF16",51.77)):
    one = g <= 19.98; half = g/2 <= 19.98
    dp  = g <= 19.98
    print(f" {q:8s} {g:6.2f} GiB | 1 card {'yes' if one else 'NO ':3s} | "
          f"TP/PP per card {g/2:6.2f} {'yes' if half else 'NO ':3s} | DP {'yes' if dp else 'NO'}"
          f" | KV left/card: PP {19.98-g/2:5.2f}  DP {19.98-g if dp else 0:5.2f}")

print("\n== PP stage balance with the LM head (vocab 248320, Q6_K 0.97 GiB) ==")
heads, emb = 0.97, 0.50
lay = (11.35 - heads*GiB/GiB - emb)  # GiB of stack
for k in (28, 32):
    s0 = emb + lay*k/64; s1 = lay*(64-k)/64 + heads
    print(f" split at layer {k}: stage0 {s0:.2f} GiB, stage1 {s1:.2f} GiB,"
          f" imbalance {100*(max(s0,s1)/min(s0,s1)-1):.1f}%")
print(f" split head in half across cards: 5.4 / 5.4 GiB, +1 handoff (0.7 us)")

print("\n== prefill link cost, 4096-token prompt ==")
print(f" TP=2: 4096 x 1.25 MiB = {4096*128*HID/1e9:.2f} GB -> {m(4096*128*HID/LINK):.0f} ms")
print(f" PP=2: 4096 x 10 KiB  = {4096*HID/1e6:.1f} MB  -> {m(4096*HID/LINK):.2f} ms"
      f"   ({4096*128*HID/(4096*HID):.0f}x cheaper)")
