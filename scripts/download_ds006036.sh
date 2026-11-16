#!/usr/bin/env bash
# Downloads the preprocessed eyes-open/photic EEG of OpenNeuro ds006036 (~1 GB, CC0, no account needed).
# Same 88 people as ds004504.
set -euo pipefail
DEST=data/ds006036
mkdir -p "$DEST"
for f in participants.tsv participants.json README dataset_description.json; do
  aws s3 cp --no-sign-request --only-show-errors "s3://openneuro.org/ds006036/$f" "$DEST/" || true
done
aws s3 sync --no-sign-request --only-show-errors s3://openneuro.org/ds006036/derivatives/ "$DEST/derivatives/"
echo "done: $DEST"
