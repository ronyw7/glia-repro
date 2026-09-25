#!/usr/bin/env bash
# Reproducible setup: pinned Vidur + Glia-setup patch + synthesized A10 profile.
set -euo pipefail
cd "$(dirname "$0")/.."
VIDUR_COMMIT=abae7f6   # microsoft/vidur main, 2026-08-24

if [ ! -d third_party/vidur ]; then
  git clone https://github.com/microsoft/vidur.git third_party/vidur
  git -C third_party/vidur checkout -q "$VIDUR_COMMIT"
  git -C third_party/vidur apply ../../patches/vidur-glia.patch
fi
pip install -q -r third_party/vidur/requirements.txt

# A10 / Llama-3-8B compute profile (derived from Vidur's A100 + A40 measurements)
python3 scripts/make_a10_profile.py --vidur third_party/vidur \
  --out_dir third_party/vidur/data/profiling/compute/a10/meta-llama/Meta-Llama-3-8B

# Train/cache Vidur's runtime predictors once (multi-core)
python3 -m glia_repro.warm_cache
