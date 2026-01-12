# RL Highway Agent (PPO) — Speed vs Safety in Dense Traffic

![Evolution (Untrained → Half → Final)](assets/evolution.gif)

> MP4 version: [assets/evolution.mp4](assets/evolution.mp4)

---
## Project Member
Aydın Özkan - 2103508

Rüya Ödül Sakuçoğlu - 2200445

Emre Topal - 2103501

---

## Objective
Train an RL agent to drive as fast as possible in dense traffic 
 (highway-env / highway-v0).

## Environment
- **Library:** highway-env (Gymnasium)
- **Env ID:** `highway-v0`
- **Observation:** numeric kinematics/features (not pixels)
- **Actions:** discrete actions (lane changes + speed control)

## Methodology

### Reward Function (Math)
$$
R_t = w_{speed}\cdot \frac{v_t}{30} - w_{crash}\cdot \mathbf{1}[\text{crash}_t] + w_{alive}
$$

Defaults (see `src/envs/reward_wrapper.py`):
- $w_{speed}=1.0$
- $w_{crash}=1.0$
- $w_{alive}=0.01$

### Model
- **Algorithm:** PPO (Stable-Baselines3)
- **Policy:** `MlpPolicy` (default SB3 MLP)

### Key Hyperparameters
See `src/config.py`.

## Training Analysis
![Reward vs Episodes](assets/reward_vs_episodes.png)


## How to Run

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install rich
```

### Train
```bash
PYTHONPATH=. python -m src.train
```

### Make Evolution Video (3 stages)
```bash
python -m src.video.make_evolution_video \
  --env-id highway-v0 \
  --mid-model outputs/models/ppo_half.zip \
  --final-model outputs/models/ppo_final.zip \
  --steps 1200 \
  --make-gif \
  --out-dir outputs/videos
```

### Plot Reward Curve
```bash
python -m src.plots.plot_reward_curve --log-dir outputs/logs --out outputs/plots/reward_vs_episodes.png
cp -f outputs/plots/reward_vs_episodes.png assets/reward_vs_episodes.png
```


