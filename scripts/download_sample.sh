#!/usr/bin/env bash
# The default sample PDF is an original CC BY 4.0 teaching note committed
# under samples/brain_mri_overview.pdf. This script regenerates it if you
# have the build-time extras installed. It does not download a third-party
# paper (licenses for many well-known preprints are not always CC-BY).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f samples/brain_mri_overview.pdf ]]; then
  echo "Sample PDF already present: samples/brain_mri_overview.pdf"
  echo "Re-run generation anyway? Installing nothing; invoking generator if reportlab+pillow are available."
fi

python - << 'PY'
import importlib.util
import sys
missing = [m for m in ("reportlab", "PIL") if importlib.util.find_spec(m) is None]
if missing:
    print(
        "To regenerate the sample PDF, install build-time extras:\n"
        "  pip install reportlab pillow\n"
        "The committed file samples/brain_mri_overview.pdf is already enough to run the README examples.",
        file=sys.stderr,
    )
    sys.exit(0)
PY

python scripts/generate_sample.py
