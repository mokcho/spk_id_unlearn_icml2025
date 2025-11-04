BASE="/data/libriheavy"
OUT_BASE="$BASE/TextGrid"

# Download models once (safe to repeat)
mfa model download language_model english_mfa_lm
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa model download g2p english_us_mfa

# Find all directories under BASE that contain WAVs
find "$BASE" -type f -name "*.wav" | while read -r wav; do
  DIR="$(dirname "$wav")"
  REL_PATH="${DIR#$BASE/}"               # strip the /data/libriheavy/ prefix
  OUT_DIR="$OUT_BASE/$REL_PATH"

  # Create matching output directory
  mkdir -p "$OUT_DIR"

  # Align once per directory (skip if already aligned)
  if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null || true)" ]; then
    echo "[SKIP] Already aligned: $REL_PATH"
    continue
  fi

  echo "[ALIGN] $REL_PATH"
  mfa align --clean "$DIR" english_mfa english_mfa "$OUT_DIR"
done