#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/bench
MODE="${BENCH_MODE:-smoke}"
RUN_ID="${BENCH_RUN_ID:-simplexlora-${MODE}}"
[[ "$MODE" == smoke || "$MODE" == benchmark ]] || { echo "Invalid BENCH_MODE"; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid BENCH_RUN_ID"; exit 2; }
SECONDS_LIMIT="${BENCH_TIMEOUT_SECONDS:-14400}"
[[ "$SECONDS_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid timeout"; exit 2; }
mkdir -p "/workspace/results/$RUN_ID"
# --resume validates source/config/checksums and restarts incomplete runs from zero.
# The timeout stops training, not billing. Stop/terminate the Pod in RunPod afterwards.
set +e
timeout --signal=TERM --kill-after=30s "$SECONDS_LIMIT" \
  python -m simplexbench.cli matrix --config "configs/$MODE.json" \
    --output "/workspace/results/$RUN_ID" --resume \
  2>&1 | tee -a "/workspace/results/$RUN_ID/console.log"
RESULT=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RESULT" > "/workspace/results/$RUN_ID/exit-code.txt"
# Keep results inspectable; exiting a container does not guarantee Pod billing stops.
# RunPod may restart a completed container; the matrix safely skips validated runs.
exit "$RESULT"
