# invariants/

Project rules the coder loop must not violate, one file per invariant, each
stating the rule, the evidence standard that proves it holds, and what a
violation looks like. These are the checks a slice cannot negotiate away, as
opposed to acceptance criteria, which vary per slice: the hardware target
(gfx1100 only, one card first), the correctness-before-speed ordering (every
kernel checked against a torch reference before any timing is recorded), the
measurement rule (no number without its baseline in the same artifact), the data
placement rule (nothing designed around peer-to-peer DMA — cross-GPU traffic is
host traffic at 14.2 GiB/s), and the forbidden-dependency check in `docs/decisions/0001` (the exclusion
decision). PLAN seeds this directory from the
rig's gate set; a coder reads it before touching a kernel, and a reviewer cites
the invariant it is grading against.
