#!/bin/bash
# Fresh capture -> candidate review -> one operator-gated full grasp cycle; gripper stays closed at Ready.
set -eo pipefail
EXECUTE=false
REVIEWED=false
for arg in "$@"; do
 case "$arg" in
  --execute) EXECUTE=true ;;
  --operator-reviewed) REVIEWED=true ;;
  *) echo "Usage: $0 [--execute --operator-reviewed]" >&2; exit 2 ;;
 esac
done
if "$EXECUTE" && ! "$REVIEWED"; then echo 'Execution requires --operator-reviewed and later GRASP confirmation.' >&2; exit 2; fi
if [ ! -t 0 ]; then echo 'Run from an interactive terminal.' >&2; exit 2; fi
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source /opt/robot_ws/install/setup.bash
source "$LAB/install_ros/setup.bash"
TRIAL="pringles_cycle_$(date +%Y%m%d_%H%M%S)"
CAPTURE="$LAB/data/$TRIAL"
RESULT="$LAB/outputs/${TRIAL}_inference"
CAL=/opt/robot_ws/src/estun_vision_grasp/config/handeye_result.yaml
DIRECTION=$(env PYTHONPATH= "$LAB/envs/anygrasp/bin/python" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); assert c["calibrated"] and c["parent_frame"]=="base_link" and c["child_frame"]=="gemini336_color_optical_frame"; print(*[-c["matrix_T_base_camera"][2][i] for i in range(3)])' "$CAL")
read -r -a APPROACH <<< "$DIRECTION"
echo 'Grasp reference: default flange 150 mm; fallback 160 mm only if every candidate fails preflight. Physical fingertip modeled at flange 180 mm; extra table clearance margin 0 mm.'
printf 'Fresh trial: %s\nROS domain: %s\n' "$TRIAL" "${ROS_DOMAIN_ID:-0}"
/opt/conda/envs/vision_env/bin/python "$LAB/scripts/capture_once.py" --name "$TRIAL"
echo 'Outline the target object or graspable rim with left clicks; Enter finishes. Confirm yellow target only with Enter, or R to reselect. Inspect candidates, then close the 3D window promptly.'
bash "$LAB/scripts/run_sdk.sh" "$LAB/scripts/run_offline.py" "$CAPTURE" --name "${TRIAL}_inference" --select polygon --checkpoint "$LAB/models/checkpoint_detection.tar" --max-width-m 0.10 --finger-height-m 0.03 --geometry-assumed --voxel-m 0.005 --approach-camera "${APPROACH[@]}" --approach-thresh-deg 15 --show
env PYTHONPATH= "$LAB/envs/anygrasp/bin/python" "$LAB/scripts/prepare_moveit.py" "$RESULT/results.json" --calibration "$CAL" --output "$RESULT/cycle_candidates.yaml" --purpose cycle
ARGS=(--candidate-file "$RESULT/cycle_candidates.yaml" --report-file "$RESULT/cycle_report.yaml")
if "$EXECUTE"; then ARGS+=(--execute --operator-reviewed); fi
printf 'Results: %s\n' "$RESULT"
/usr/bin/python3 "$LAB/scripts/run_cycle.py" "${ARGS[@]}" 2>&1 | tee "$RESULT/cycle_console.log"
