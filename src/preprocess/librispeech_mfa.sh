#!/usr/bin/env bash
BASE="/data/LibriSpeech"
OUT_BASE="/data/LibriSpeech/TextGrid"

# Download models once
mfa model download language_model english_mfa_lm
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa model download g2p english_us_mfa

# List of subsets to process
SUBSETS=("test-clean" "test-other")

for name in "${SUBSETS[@]}"; do
  ROOT_PATH="$BASE/$name"
  RESULT_PATH="$OUT_BASE/$name"
  mkdir -p "$RESULT_PATH"

  # Skip if folder is empty or missing
  if [ ! -d "$ROOT_PATH" ] || [ -z "$(find "$ROOT_PATH" -type f -name '*.wav' 2>/dev/null)" ]; then
    echo "[SKIP] $name – no audio files found."
    continue
  fi

  echo "[ALIGN] $name"
  mfa align --clean "$ROOT_PATH" english_mfa english_mfa "$RESULT_PATH"
done