# RL Highway Agent (PPO) — Speed vs Safety in Dense Traffic

![Evolution (Untrained → Half → Final)](assets/evolution.gif)

> MP4 version: [assets/evolution.mp4](assets/evolution.mp4)

---

## Project Members

- **Aydın Özkan** — 2103508
- **Rüya Ödül Sakuçoğlu** — 2200445
- **Emre Topal** — 2103501

---

## Work Split (Who did what)

This section makes it explicit which teammate implemented which part of the repository.

| Member | Responsibilities | Key files / folders |
|---|---|---|
| **Aydın Özkan** | PPO training pipeline (fast run), custom reward shaping for “speed vs safety”, saving half/final checkpoints, main training entrypoint integration | `src/agents/train_ppo_fast.py`, `src/train.py`, parts of `src/config.py` |
| **Rüya Ödül Sakuçoğlu** | Baseline environment wrapper (speed/alive/crash), vectorized env factory, configurable PPO trainer with mid/final checkpoints, smoke tests for env + wrapper | `src/envs/reward_wrapper.py`, `src/envs/make_env.py`, `src/agents/train_ppo.py`, `src/agents/smoke_test_env.py`, `src/agents/smoke_test_wrapped_env.py` |
| **Emre Topal** | Training analysis utilities (reward curve plotting), evolution video (untrained → half → final), assets/scripts packaging + README organization | `src/plots/plot_reward_curve.py`, `src/video/make_evolution_video.py`, `scripts/`, `assets/`, `README.md` |

> Note: the breakdown above is based on how the codebase is structured (each responsibility maps to the listed modules).

---

## Objective

Train an RL agent to drive **as fast as possible** in **dense traffic** while staying **safe** (avoid collisions) and **stable** (avoid unnecessary weaving) in `highway-env / highway-v0`.

---

## Environment

- **Library:** `highway-env` (Gymnasium)
- **Env ID:** `highway-v0`
- **Observation:** numeric kinematics/features (not pixels)
- **Actions:** discrete meta-actions (lane changes + speed control)

---

## Methodology

### Algorithm

- **RL algorithm:** PPO (Stable-Baselines3)
- **Policy:** `MlpPolicy` (MLP)

We keep two training variants:

1. **Baseline PPO trainer** (`train_ppo.py`): uses the baseline reward wrapper (`reward_wrapper.py`).
2. **Fast PPO trainer** (`train_ppo_fast.py`): uses a more shaped reward to reduce zig-zagging and crashes.

### Reward Function

#### A) Baseline reward wrapper (`src/envs/reward_wrapper.py`)

Per-step reward:

\[
R_t = w_{speed}\cdot \frac{v_t}{30} + w_{crash}\cdot \mathbf{1}[\text{crash}_t]\cdot(-1) + w_{alive}\cdot 1
\]

Default weights (from code):
- \(w_{speed}=1.0\)
- \(w_{crash}=5.0\)
- \(w_{alive}=0.05\)

#### B) Shaped reward (used in `src/agents/train_ppo_fast.py`)

We also experimented with a shaped reward to balance speed with smoother behavior:

- normalized speed in \([0,1]\) using \(v_{min}=20\), \(v_{max}=30\)
- encourage staying to the right lane (less aggressive weaving)
- penalize lane changes
- strong collision penalty

Default shaping weights (from code):
- `w_speed = 0.6`
- `w_right_lane = 0.3`
- `w_lane_change = 0.1`
- `w_collision = 1.0`

The shaped reward is clipped into \([0,1]\) to keep training stable.

### Key Hyperparameters

Main training hyperparameters are in `src/config.py` (learning rate, gamma, etc.).  
For best reproducibility, run training with the commands below (they create the same output paths every time).

---

## Training Analysis

Reward curve (episode reward + moving average):

![Reward vs Episodes](assets/reward_vs_episodes.png)

---

## How to Run

### 1) Install (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install rich
```

### 2) Smoke tests (quick sanity)

```bash
bash scripts/smoke.sh
```

This runs:
- `src/agents/smoke_test_env.py`
- `src/agents/smoke_test_wrapped_env.py`

### 3) Train (recommended default — fast trainer)

`src/train.py` runs `src/agents/train_ppo_fast.py` and saves **half** and **final** checkpoints:

```bash
PYTHONPATH=. python -m src.train
```

Outputs:
- `outputs/models/ppo_half.zip`
- `outputs/models/ppo_final.zip`
- `outputs/logs/monitor.csv`

### 4) Plot reward curve

```bash
python -m src.plots.plot_reward_curve \
  --monitor outputs/logs/monitor.csv \
  --out outputs/plots/reward_vs_episodes.png

cp -f outputs/plots/reward_vs_episodes.png assets/reward_vs_episodes.png
```

### 5) Make evolution video (3 stages)

This creates a single video that contains:
1) **Untrained** (random)  
2) **Half-trained** checkpoint  
3) **Final** checkpoint  

```bash
python -m src.video.make_evolution_video \
  --env-id highway-v0 \
  --mid-model outputs/models/ppo_half.zip \
  --final-model outputs/models/ppo_final.zip \
  --steps 1200 \
  --make-gif \
  --out-dir outputs/videos

mkdir -p assets
cp -f outputs/videos/evolution.mp4 assets/evolution.mp4
cp -f outputs/videos/evolution.gif assets/evolution.gif
```

---

## Alternative: Longer training (baseline trainer + mid/final)

If you want a longer run with a “mid” checkpoint and a “final” checkpoint, use `train_ppo.py`.

> Important: in this mode, **set `total_timesteps >= mid_timesteps`**, otherwise only the mid checkpoint will be created.

```bash
python -m src.agents.train_ppo \
  --env-id highway-v0 \
  --n-envs 8 \
  --total-timesteps 300000 \
  --mid-timesteps 150000
```

Outputs:
- `outputs/models/ppo_mid.zip`
- `outputs/models/ppo_final.zip` (only if total > mid)

---

## Project Structure

```
assets/                       # visuals used in README (gif/mp4/plots)
outputs/                      # generated models/logs/videos/plots (created after running)
scripts/                      # helper scripts (smoke/train)
src/
  agents/
    train_ppo_fast.py         # main training (shaped reward + half/final checkpoints)
    train_ppo.py              # configurable trainer (mid/final checkpoints)
    smoke_test_env.py
    smoke_test_wrapped_env.py
  envs/
    reward_wrapper.py         # baseline reward wrapper (speed + alive − crash)
    make_env.py               # vectorized env creation helper
  plots/
    plot_reward_curve.py      # plot reward vs episodes from monitor.csv
  video/
    make_evolution_video.py   # build evolution video (untrained/half/final)
  config.py                   # main hyperparameters & defaults
  train.py                    # entrypoint -> train_ppo_fast
```

---

## Evidence / Screenshots (fill in)

Add your own screenshots here (requested for submission):

- **Training completed (terminal output):**  
<img width="2158" height="1096" alt="image" src="https://github.com/user-attachments/assets/58222197-635e-46a4-bac4-94a813b0b615" />

- **Reward curve produced:**  
  _[PLACEHOLDER — add screenshot]_

- **Evolution video frames (half vs final):**  
  _[PLACEHOLDER — add screenshot]_

---

## Notes / Troubleshooting

- If `imageio` complains about video codecs, try reinstalling it or using a different Python environment.
- If your system has issues with `SubprocVecEnv` spawning, reduce `n_envs` in `src/config.py`.

---

## Acknowledgements

- `highway-env` for the traffic simulation environment
- Gymnasium for the RL interface
- Stable-Baselines3 for PPO implementation

