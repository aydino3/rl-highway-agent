
set -euo pipefail



TOTAL=${1:-300000}
MID=${2:-150000}

python3 -m src.agents.train_ppo --total-timesteps "$TOTAL" --mid-timesteps "$MID"
