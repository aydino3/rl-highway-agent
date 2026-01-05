<div align="center">

# Applied Reinforcement Learning — Highway-Env Agent (PPO)
**Author:** Aydın Özkan

Train an autonomous driving agent in **highway-env** to drive fast in dense traffic **without crashing**.

</div>

---

## 🎬 Evolution (3 Stages)

**Untrained → Half-Trained → Fully Trained** (exactly 3 stages, in a single file)

<div align="center">

![Evolution](assets/evolution.gif)

</div>

> If the GIF does not render (first run), generate it with: `./scripts/make_assets.sh`  
> Optional: the MP4 version is saved as `assets/evolution.mp4`.

---

## 📦 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ✅ Smoke Tests

```bash
./scripts/smoke.sh
```

---

## 🧠 Environment & Agent

### Environment
- **Env:** `highway-v0` (Gymnasium + highway-env)
- **Observation:** kinematics-style vector (ego + nearby vehicles)
- **Action space:** discrete lane-change / speed-control actions

### Custom Reward (Wrapper)

Training uses a reward wrapper (`src/envs/reward_wrapper.py`) to make learning more stable.

Reward definition:

$$
r_t =
w_{speed}\cdot \frac{v_t}{30}
\;-\;
w_{crash}\cdot \mathbb{1}[crashed_t]
\;+\;
w_{alive}\cdot 1
$$

Default weights (as implemented in code):
- $w_{speed}=1.0$
- $w_{crash}=5.0$
- $w_{alive}=0.05$

Intuition:
- Encourage maintaining higher speed.
- Penalize collisions strongly.
- Add a small survival bonus to reduce “freeze” behavior.

---

## 🤖 Training Method: PPO

We use **Proximal Policy Optimization (PPO)** from Stable-Baselines3 with an MLP policy.

**Key hyperparameters** (see `src/config.py` + `src/agents/train_ppo.py`):
- `learning_rate = 3e-4`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `n_steps = 2048` (auto-adjusts for very small runs)
- `batch_size = 64` (auto-adjusts to divide rollout size)

### Train (recommended)

```bash
./scripts/train.sh 300000 150000
```

This writes:
- `outputs/models/ppo_mid.zip`
- `outputs/models/ppo_final.zip`
- logs under `outputs/logs/` (CSV/TensorBoard)

---

## 📈 Training Curve

<div align="center">

![Training Curve](assets/training_curve.png)

</div>

### Commentary (how to read it)
- **Early episodes:** reward is low/unstable while PPO explores and crashes frequently.
- **Mid training:** reward rises as the agent learns safer lane changes and speed control.
- **Late training:** the curve tends to plateau once the policy finds a stable balance between speed and safety.

---

## 🧩 Challenges & Fixes

**Hurdle:** With very small `total_timesteps`, PPO can look “stuck” because rollout size (`n_envs * n_steps`) is large, so the policy updates rarely.

**Fix:** The training script auto-adjusts:
- `n_steps` down to fit small runs
- `batch_size` so that it divides rollout size

See: `src/agents/train_ppo.py` (`_auto_steps`, `_auto_batch`).

---

## 🛠️ Build Report Assets (GIF + Plot)

One command to generate everything required for the report:

```bash
./scripts/make_assets.sh
```

Outputs:
- `assets/evolution.gif` (required)
- `assets/training_curve.png` (required)
- `assets/evolution.mp4` (optional)

---

## 🧼 Repo Hygiene

- `outputs/` contains logs, models, and raw videos and is **gitignored**.
- Commit only the final report media under `assets/`.
