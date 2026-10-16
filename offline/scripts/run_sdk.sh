#!/bin/bash
# Run official SDK tools with the license-relative working directory.
set -euo pipefail
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$LAB/tools/bin:$LAB/envs/anygrasp/bin:/usr/bin:/bin"
export PYTHONPATH="$LAB/vendor/anygrasp_sdk/grasp_detection:$LAB/src"
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="$LAB/tools/lib:${LD_LIBRARY_PATH:-}"
cd "$LAB/vendor/anygrasp_sdk/grasp_detection"
exec "$LAB/envs/anygrasp/bin/python" "$@"
