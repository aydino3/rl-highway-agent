
set -euo pipefail



ASSETS_DIR="assets"
OUT_LOG_DIR="outputs/logs"
OUT_VIDEO_DIR="outputs/videos"
OUT_PLOT_DIR="outputs/plots"
OUT_MODEL_DIR="outputs/models"

mkdir -p "$ASSETS_DIR" "$OUT_LOG_DIR" "$OUT_VIDEO_DIR" "$OUT_PLOT_DIR" "$OUT_MODEL_DIR"


MID_MODEL="$OUT_MODEL_DIR/ppo_mid.zip"
FINAL_MODEL="$OUT_MODEL_DIR/ppo_final.zip"

if [[ ! -f "$MID_MODEL" || ! -f "$FINAL_MODEL" ]]; then
  echo "[make_assets] Models not found. Training a quick run (increase timesteps for best results)..."
  python3 -m src.agents.train_ppo --n-envs 4 --total-timesteps 200000 --mid-timesteps 100000
else
  echo "[make_assets] Found existing models. Skipping training."
fi


echo "[make_assets] Generating training curve..."
python3 -m src.plots.plot_reward_curve --log-dir "$OUT_LOG_DIR" --out "$ASSETS_DIR/training_curve.png"


echo "[make_assets] Generating evolution video (3 stages)..."
python3 -m src.video.make_evolution_video --steps 1500 --make-gif --out-dir "$OUT_VIDEO_DIR"


if [[ -f "$OUT_VIDEO_DIR/evolution.mp4" ]]; then
  cp "$OUT_VIDEO_DIR/evolution.mp4" "$ASSETS_DIR/evolution.mp4"
fi
if [[ -f "$OUT_VIDEO_DIR/evolution.gif" ]]; then
  cp "$OUT_VIDEO_DIR/evolution.gif" "$ASSETS_DIR/evolution.gif"
fi

echo "[make_assets] Done."
echo "  - $ASSETS_DIR/training_curve.png"
echo "  - $ASSETS_DIR/evolution.gif"
echo "  - $ASSETS_DIR/evolution.mp4 (optional)"
