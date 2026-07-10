#!/usr/bin/env bash
# Stage one demo dataset onto an EC2 instance's data volume.
#
#   ./fetch_data.sh v1           /data      # from figshare (public)
#   ./fetch_data.sh hippocampus  /data
#   ./fetch_data.sh lbm          /data
#
# With S3_BUCKET set, pulls from S3 instead — much faster inside AWS, and the
# right choice if you are running the demos more than once:
#
#   S3_BUCKET=my-suite3d-data ./fetch_data.sh v1 /data
#
# To populate that bucket once, from a machine that already has the archive:
#   aws s3 sync ./figshare s3://my-suite3d-data/ --exclude '*' --include 'v1/*'

set -euo pipefail

DATASET="${1:?usage: fetch_data.sh <v1|hippocampus|lbm> <dest-dir>}"
DEST="${2:?usage: fetch_data.sh <v1|hippocampus|lbm> <dest-dir>}"

case "$DATASET" in
    v1)          SIZE_GB=21  ;;
    hippocampus) SIZE_GB=21  ;;
    lbm)         SIZE_GB=57  ;;
    *) echo "unknown dataset '$DATASET' (v1 | hippocampus | lbm)" >&2; exit 1 ;;
esac

mkdir -p "$DEST/$DATASET/raw"

avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt $((SIZE_GB * 2)) ]; then
    echo "ERROR: only ${avail_gb} GB free at $DEST." >&2
    echo "       '$DATASET' needs ~${SIZE_GB} GB raw plus roughly as much again" >&2
    echo "       for the registered movie. Attach a bigger EBS volume." >&2
    exit 1
fi

if [ -n "${S3_BUCKET:-}" ]; then
    echo "==> syncing s3://$S3_BUCKET/$DATASET/ -> $DEST/$DATASET/"
    aws s3 sync "s3://$S3_BUCKET/$DATASET/" "$DEST/$DATASET/"
else
    cat >&2 <<EOF
==> No S3_BUCKET set.

The figshare archive does not expose stable per-file URLs, so this script
cannot download it unattended. Two options:

  1. Download the archive once (browser or figshare CLI), then copy the
     dataset folder up:
         aws s3 sync ./figshare/$DATASET s3://<your-bucket>/$DATASET/
         S3_BUCKET=<your-bucket> $0 $DATASET $DEST

  2. scp it straight from a machine that has it:
         scp -r ./figshare/$DATASET ubuntu@<instance>:$DEST/

Expected layout when you are done:
    $DEST/$DATASET/raw/*.tif
EOF
    exit 1
fi

n_tifs=$(find "$DEST/$DATASET/raw" -name '*.tif' | wc -l)
echo "==> $n_tifs tifs in $DEST/$DATASET/raw"
[ "$n_tifs" -gt 0 ] || { echo "ERROR: no tifs landed — check the bucket layout." >&2; exit 1; }
echo "==> ready:  --data-root $DEST"
