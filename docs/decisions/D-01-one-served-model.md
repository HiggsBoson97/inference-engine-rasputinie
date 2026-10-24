# D-01 — One served model: the published IQ3_S file, with its MTP head

- **Decided:** the engine serves exactly `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`,
  `…-IQ3_S.gguf`, with its shipped MTP head. No second served model, no second quant file of the
  target, no mmproj/vision path. **A drafter is not a second served model** — the MTP head ships
  inside the file, and the DFlash2 drafter is admitted as a *draft* model under `D-08`.
- **Rejected:** a second quantisation of the same weights — it would duplicate 11.8 GB of VRAM and
  buy nothing this file does not already contain; and a bigger file (17.56 GB `UD-Q4_K_XL`) — it
  costs KV capacity for a rate that `facts/quant.md` shows is not uniformly better.
- **Why it binds:** the model choice fixes the byte budget, the 72 `(role, type)` pairs, and
  therefore every kernel.
- **Evidence:** publisher card (`huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`), which states
  the files are **standard GGUF**; `facts/quant.md` for the type inventory.
- **Cited by:** `D-05`, `D-06`, every kernel unit.
