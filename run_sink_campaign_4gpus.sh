#!/usr/bin/env bash
set -euo pipefail

# Run a sink campaign with 4 GPU queues (one queue per GPU).
# - Generates (model × dataset) commands
# - Assigns them round-robin to GPU 0..3
# - Runs the 4 queues concurrently
#
# Usage:
#   bash run_sink_campaign_4gpus.sh
#   CAMPAIGN_CONFIG=configs/sink_campaign.json bash run_sink_campaign_4gpus.sh
#
# Notes:
# - Some models (e.g., meta-llama/*, google/*) may require accepting licenses / HF auth.
# - Each command is run via `uv run python ...` so it uses the repo environment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CAMPAIGN_CONFIG:-"$ROOT/configs/sink_campaign.json"}"
WORKDIR="$ROOT"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

JOBS_DIR="$ROOT/artifacts/campaign_jobs"
mkdir -p "$JOBS_DIR"

JOB0="$JOBS_DIR/gpu0.sh"
JOB1="$JOBS_DIR/gpu1.sh"
JOB2="$JOBS_DIR/gpu2.sh"
JOB3="$JOBS_DIR/gpu3.sh"

cat > "$JOB0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
cat > "$JOB1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
cat > "$JOB2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
cat > "$JOB3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF

chmod +x "$JOB0" "$JOB1" "$JOB2" "$JOB3"

# Generate job scripts from JSON config.
uv run python scripts/build_campaign_jobs.py \
  --config "$CONFIG" \
  --job_paths "$JOB0" "$JOB1" "$JOB2" "$JOB3"

echo "Launching 4 GPU queues..."
(
  cd "$WORKDIR"
  bash "$JOB0"
) > "$JOBS_DIR/gpu0.log" 2>&1 &
P0=$!

(
  cd "$WORKDIR"
  bash "$JOB1"
) > "$JOBS_DIR/gpu1.log" 2>&1 &
P1=$!

(
  cd "$WORKDIR"
  bash "$JOB2"
) > "$JOBS_DIR/gpu2.log" 2>&1 &
P2=$!

(
  cd "$WORKDIR"
  bash "$JOB3"
) > "$JOBS_DIR/gpu3.log" 2>&1 &
P3=$!

echo "PIDs: gpu0=$P0 gpu1=$P1 gpu2=$P2 gpu3=$P3"
echo "Logs: $JOBS_DIR/gpu{0,1,2,3}.log"
echo "You can tail them, e.g.: tail -f $JOBS_DIR/gpu0.log"

wait "$P0" "$P1" "$P2" "$P3"
echo "All queues finished."

