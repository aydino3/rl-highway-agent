from __future__ import annotations

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401  (register envs)
from moviepy.editor import VideoFileClip, concatenate_videoclips
from stable_baselines3 import PPO

from src.config import TrainConfig
from src.envs.reward_wrapper import CustomRewardWrapper


def _make_env(env_id: str, seed: int, record_dir: str, name_prefix: str) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    env = CustomRewardWrapper(env)

    os.makedirs(record_dir, exist_ok=True)

    # Record only the first episode (one clip per stage)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=record_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda ep: ep == 0,
        disable_logger=True,
    )
    env.reset(seed=seed)
    # highway-env recommends registering the RecordVideo wrapper to capture intermediate frames
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except Exception:
        pass
    return env


def _rollout_random(env: gym.Env, n_steps: int) -> None:
    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def _rollout_model(env: gym.Env, model: PPO, n_steps: int) -> None:
    obs, info = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def _pick_latest_mp4(folder: str, prefix: str) -> str:
    mp4s = [f for f in os.listdir(folder) if f.endswith(".mp4") and f.startswith(prefix)]
    if not mp4s:
        # gym may create nested dirs; search recursively
        found = []
        for r, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".mp4") and f.startswith(prefix):
                    found.append(os.path.join(r, f))
        if not found:
            raise FileNotFoundError(f"No mp4 found under {folder} with prefix {prefix}")
        return sorted(found)[-1]
    return os.path.join(folder, sorted(mp4s)[-1])


def _concat_three(untrained: str, mid: str, final: str, out_path: str) -> None:
    clips = [VideoFileClip(untrained), VideoFileClip(mid), VideoFileClip(final)]
    final_clip = concatenate_videoclips(clips, method="compose")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_clip.write_videofile(out_path, audio=False, verbose=False, logger=None)
    for c in clips:
        c.close()
    final_clip.close()


def maybe_make_gif(mp4_path: str, gif_path: str, fps: int = 12) -> None:
    clip = VideoFileClip(mp4_path)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    clip.write_gif(gif_path, fps=fps, program="ffmpeg")
    clip.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500, help="Steps per stage")
    p.add_argument("--mid-model", type=str, default="outputs/models/ppo_mid.zip")
    p.add_argument("--final-model", type=str, default="outputs/models/ppo_final.zip")
    p.add_argument("--out-dir", type=str, default="outputs/videos")
    p.add_argument("--make-gif", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    env_id = args.env_id or cfg.env_id

    stage_dir = os.path.join(args.out_dir, "stages")
    os.makedirs(stage_dir, exist_ok=True)

    # 1) Untrained (random actions)
    env = _make_env(env_id, args.seed, stage_dir, "untrained")
    _rollout_random(env, args.steps)
    env.close()
    untrained_mp4 = _pick_latest_mp4(stage_dir, "untrained")

    # 2) Mid model
    if not os.path.exists(args.mid_model):
        raise FileNotFoundError(f"Mid model not found: {args.mid_model}")
    mid_model = PPO.load(args.mid_model)
    env = _make_env(env_id, args.seed, stage_dir, "half_trained")
    _rollout_model(env, mid_model, args.steps)
    env.close()
    mid_mp4 = _pick_latest_mp4(stage_dir, "half_trained")

    # 3) Final model
    if not os.path.exists(args.final_model):
        raise FileNotFoundError(f"Final model not found: {args.final_model}")
    final_model = PPO.load(args.final_model)
    env = _make_env(env_id, args.seed, stage_dir, "fully_trained")
    _rollout_model(env, final_model, args.steps)
    env.close()
    final_mp4 = _pick_latest_mp4(stage_dir, "fully_trained")

    # Combine
    combined_mp4 = os.path.join(args.out_dir, "evolution.mp4")
    _concat_three(untrained_mp4, mid_mp4, final_mp4, combined_mp4)
    print(f"Saved combined video to: {combined_mp4}")

    if args.make_gif:
        gif_path = os.path.join(args.out_dir, "evolution.gif")
        maybe_make_gif(combined_mp4, gif_path)
        print(f"Saved GIF to: {gif_path}")


if __name__ == "__main__":
    main()
