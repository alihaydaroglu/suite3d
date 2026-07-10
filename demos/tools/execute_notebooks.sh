#!/usr/bin/env bash
# Execute the demo walkthroughs and make them safe to publish.
#
# The executed notebooks are rendered to static HTML for the public website, so
# this does two things: runs them against real data, then scrubs absolute paths
# out of the output cells and fails loudly if any survive.
#
#   export SUITE3D_DEMO_DATA=/path/to/figshare     # holds v1/ hippocampus/ lbm/
#   export SUITE3D_DEMO_OUT=/path/to/scratch       # job dirs land here (large!)
#   ./tools/execute_notebooks.sh 01-v1-tc030
#   ./tools/execute_notebooks.sh                   # all four, in dependency order
#
# 04-sweep re-uses 03-hippocampus's job directory, so run 03 first. The default
# ordering below does that for you.
#
# Notes
#   * Needs a CUDA GPU. Demo 02 (LBM) is large and slow; see demos/README.md.
#   * Each notebook gets its own timeout; a hung kernel should not wedge the run.
#   * Notebooks are executed IN PLACE, so the outputs are committed with them.

set -euo pipefail

cd "$(dirname "$0")/.."          # demos/
PY="${PYTHON:-python}"
TIMEOUT="${CELL_TIMEOUT:-14400}" # seconds per cell (LBM registration is hours)

: "${SUITE3D_DEMO_DATA:?set SUITE3D_DEMO_DATA to your figshare download}"
: "${SUITE3D_DEMO_OUT:?set SUITE3D_DEMO_OUT to a scratch dir with room to spare}"

DEMOS=("$@")
if [ ${#DEMOS[@]} -eq 0 ]; then
    DEMOS=(01-v1-tc030 03-hippocampus 04-sweep 02-lbm-ss004)
fi

for demo in "${DEMOS[@]}"; do
    nb="$demo/walkthrough.ipynb"
    [ -f "$nb" ] || { echo "no such notebook: $nb" >&2; exit 1; }

    echo "=== executing $nb"
    start=$(date +%s)
    "$PY" -m nbconvert --to notebook --execute --inplace \
          --ExecutePreprocessor.timeout="$TIMEOUT" \
          --ExecutePreprocessor.kernel_name=python3 \
          "$nb"
    elapsed=$(( $(date +%s) - start ))
    printf '=== %s finished in %dm %ds\n' "$demo" $((elapsed / 60)) $((elapsed % 60))
done

echo
echo "=== scrubbing absolute paths out of output cells"
"$PY" tools/scrub_outputs.py "${DEMOS[@]/%//walkthrough.ipynb}"

echo
echo "=== verifying (this is the gate, not the scrub above)"
"$PY" tools/scrub_outputs.py --check-only "${DEMOS[@]/%//walkthrough.ipynb}"
echo "Safe to commit."
