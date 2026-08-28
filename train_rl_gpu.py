#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the canonical RL trainer.

The production training implementation lives in ``ml.rl_train``.  This entry
point is kept only so older operator commands keep working without maintaining
a second training contract.
"""
from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical RL training via ml.rl_train")
    parser.add_argument("--model", "-m", default="ppo", choices=["ppo", "all"], help="Compatibility option; maps to PPO")
    parser.add_argument("--timesteps", "-t", type=int, default=250_000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--profile", default="balanced", choices=["fast", "balanced", "max"])
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--optimize", "-o", action="store_true", help="Removed; use the dedicated Optuna workflow")
    parser.add_argument("--trials", type=int, default=50, help="Compatibility option for removed --optimize")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.optimize:
        raise SystemExit("--optimize was removed from train_rl_gpu.py; use ml.run_optuna_optimization explicitly")

    from ml.rl_train import RLTrainingConfig, train

    cfg = RLTrainingConfig(
        timesteps=args.timesteps,
        device=args.device,
        profile=args.profile,
        progress_bar=bool(args.progress_bar),
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
