#!/usr/bin/env bash
# fetch-tiny.sh — download a SMALL public TinyLlama GGUF into models/tiny.gguf
#
# Weight: TinyLlama 1.1B Chat v1.0, Q2_K GGUF (~482 MiB)
# Host:   Hugging Face  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Page:   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# File:   tinyllama-1.1b-chat-v1.0.Q2_K.gguf
# License of the weights: Apache 2.0 (TinyLlama). This script is MIT.
#
# Q4_0 (~638 MiB) if you want a slightly less quantized stand-in:
#   FILE="tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
#
# Compose mounts ./models as /models (read-only). The server flag is:
#   -m /models/tiny.gguf --alias tinyllama
# Do not commit the .gguf. This repo does not vendor weights.
#
# Linux / WSL / Git Bash:
#   chmod +x scripts/fetch-tiny.sh
#   ./scripts/fetch-tiny.sh
#
# Copy-paste without the script (same URL):
#   mkdir -p models
#   wget -O models/tiny.gguf \
#     https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
#   # or:
#   curl -L --fail -o models/tiny.gguf \
#     https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${ROOT}/models"
DEST="${DEST_DIR}/tiny.gguf"
REPO="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILE="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
  echo "already present: ${DEST}"
  ls -lh "${DEST}"
  exit 0
fi

echo "GET ${URL}"
echo " -> ${DEST}"

if command -v wget >/dev/null 2>&1; then
  wget -O "${DEST}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L --fail --output "${DEST}" "${URL}"
else
  echo "need wget or curl" >&2
  exit 1
fi

ls -lh "${DEST}"
echo "compose mount: ${DEST} -> /models/tiny.gguf"
