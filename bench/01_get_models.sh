#!/usr/bin/env bash
# Fetch the model files baselines are measured on.
# Every file is size-checked and hashed. Hashes land next to the file so a
# later run can prove it measured the same bytes.
#
# Run on rasputin:  bash bench/01_get_models.sh 27b-q4
# Models are large. Download only what the current task needs.
set -euo pipefail

MODELS="${MODELS:-$HOME/local-inference/models}"
DEST="$MODELS/gguf"
mkdir -p "$DEST"

fetch() { # url dest
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "exists: $dest ($(du -h "$dest" | cut -f1))"
  else
    echo "fetch: $url"
    curl -fSL -C - --retry 5 --retry-delay 3 -o "$dest" "$url"
  fi
  sha256sum "$dest" | tee "$dest.sha256"
}

case "${1:-}" in
  27b-q4)
    fetch "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
          "$DEST/Qwen3.8-27B-UD-Q4_K_XL.gguf"
    fetch "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Q4_0.gguf" \
          "$DEST/mtp-Qwen3.8-27B-Q4_0.gguf"
    ;;
  27b-iq3)
    fetch "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_S.gguf" \
          "$DEST/Qwen3.8-27B-UD-IQ3_S.gguf"
    ;;
  flash-iq3)
    base="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ3_S"
    for i in 1 2; do
      fetch "$base/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000$i-of-00002.gguf" \
            "$DEST/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000$i-of-00002.gguf"
    done
    ;;
  *)
    echo "usage: $0 {27b-q4|27b-iq3|flash-iq3}"; exit 2
    ;;
esac

echo "--- inventory ---"
ls -lh "$DEST"/*.gguf 2>/dev/null | awk '{print $5, $9}'
