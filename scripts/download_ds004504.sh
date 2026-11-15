#!/usr/bin/env bash
# Downloads the preprocessed EEG of OpenNeuro ds004504 (~2.9 GB, CC0, no account needed).
set -euo pipefail
DEST=data/ds004504
mkdir -p "$DEST"
for f in participants.tsv participants.json README dataset_description.json; do
  aws s3 cp --no-sign-request --only-show-errors "s3://openneuro.org/ds004504/$f" "$DEST/"
done
aws s3 sync --no-sign-request --only-show-errors s3://openneuro.org/ds004504/derivatives/ "$DEST/derivatives/"
echo "done: $DEST"
