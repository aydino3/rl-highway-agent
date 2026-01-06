# mypy: ignore-errors
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

try:
    
    from src.envs.reward_wrapper import CustomRewardWrapper 
except Exception:
    CustomRewardWrapper = None  


def _title_frame(text: str, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    img = Image.new("RGB", (w, h), (15, 15, 18))
    draw = ImageDraw.Draw(img)

    
    try:
        font = ImageFont.truetype("Arial.ttf", size=max(18, h // 14))
    except Exception:
        font = ImageFont.load_default()

    
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, font=font, fill=(240, 240, 245))
    return np.array(img)


def _make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    if CustomRewardWrapper is not None:
        env = CustomRewardWrapper(env)
    env.reset(seed=seed)
    return env


def _rollout_and_write(
    writer: imageio.Writer,
    env_id: str,
    stage_name: str,
    steps: int,
    fps: int,
    seed: int,
    policy: Optional[Callable[[np.ndarray], int]] = None,
    title_seconds: float = 1.0,
) -> None:
    env = _make_env(env_id, seed=seed)

    first = env.render()
    if first is None:
        raise RuntimeError("env.render() returned None. Try a different env_id or check render_mode.")
    h, w = first.shape[0], first.shape[1]

    
    title = _title_frame(stage_name, (w, h))
    for _ in range(int(fps * title_seconds)):
        writer.append_data(title)

    obs, _info = env.reset(seed=seed)
    frame = env.render()
    if frame is not None:
        writer.append_data(frame)

    for _t in range(steps):
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(obs)

        obs, _r, terminated, truncated, _info = env.step(action)
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        if terminated or truncated:
            obs, _info = env.reset()
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="highway-v0")
    p.add_argument("--mid-model", type=str, required=True)
    p.add_argument("--final-model", type=str, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/videos")
    p.add_argument("--make-gif", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / "evolution.mp4"
    gif_path = out_dir / "evolution.gif"

    
    mid_model = PPO.load(args.mid_model, device="cpu")
    final_model = PPO.load(args.final_model, device="cpu")

    def mid_policy(obs: np.ndarray) -> int:
        a, _ = mid_model.predict(obs, deterministic=True)
        return int(a)

    def final_policy(obs: np.ndarray) -> int:
        a, _ = final_model.predict(obs, deterministic=True)
        return int(a)

    
    with imageio.get_writer(str(mp4_path), fps=args.fps) as writer:
        _rollout_and_write(
            writer,
            env_id=args.env_id,
            stage_name="Stage 1 — Untrained (Random)",
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
            policy=None,
        )
        _rollout_and_write(
            writer,
            env_id=args.env_id,
            stage_name="Stage 2 — Half-Trained",
            steps=args.steps,
            fps=args.fps,
            seed=args.seed + 1,
            policy=mid_policy,
        )
        _rollout_and_write(
            writer,
            env_id=args.env_id,
            stage_name="Stage 3 — Fully Trained",
            steps=args.steps,
            fps=args.fps,
            seed=args.seed + 2,
            policy=final_policy,
        )

    print(f"Wrote: {mp4_path}")

    if args.make_gif:
        
        reader = imageio.get_reader(str(mp4_path))
        frames = []
        step = 2  
        for i, frame in enumerate(reader):
            if i % step == 0:
                frames.append(frame)
        reader.close()
        imageio.mimsave(str(gif_path), frames, duration=(step / args.fps))
        print(f"Wrote: {gif_path}")


if __name__ == "__main__":
    main()
