from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def _find_progress_csv(log_dir: str) -> str:
    candidate = os.path.join(log_dir, "progress.csv")
    if os.path.exists(candidate):
        return candidate

    # SB3 sometimes nests, so search one level
    for root, _, files in os.walk(log_dir):
        if "progress.csv" in files:
            return os.path.join(root, "progress.csv")

    raise FileNotFoundError(f"Could not find progress.csv under: {log_dir}")


def plot_reward_curve(
    csv_path: str,
    out_path: str,
    title: str = "Reward vs Episodes",
    x_col: str = "episodes",
    y_col: str = "rollout/ep_rew_mean",
) -> None:
    df = pd.read_csv(csv_path)

    # Some SB3 versions log different columns; try fallbacks.
    if y_col not in df.columns:
        for alt in ["rollout/ep_rew_mean", "rollout/ep_rew_mean", "rollout/ep_rew_mean"]:
            if alt in df.columns:
                y_col = alt
                break

    if x_col not in df.columns:
        # Approximate episodes with index if missing
        df[x_col] = range(1, len(df) + 1)

    if y_col not in df.columns:
        # last fallback: try any column that contains 'rew' and 'mean'
        candidates = [c for c in df.columns if "rew" in c and "mean" in c]
        if not candidates:
            raise KeyError(
                f"Reward column not found. Available columns: {list(df.columns)}"
            )
        y_col = candidates[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episode Reward")
    plt.title(title)
    plt.legend([y_col])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=str, default="outputs/logs", help="SB3 log directory")
    p.add_argument("--csv", type=str, default=None, help="Path to progress.csv (optional)")
    p.add_argument("--out", type=str, default="outputs/plots/reward_vs_episode.png")
    p.add_argument("--title", type=str, default="Reward vs Episodes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or _find_progress_csv(args.log_dir)
    plot_reward_curve(csv_path, args.out, title=args.title)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
