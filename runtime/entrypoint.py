from __future__ import annotations

import argparse
import asyncio
import json

from .runtime_main_service import main_service

OFFICIAL_RUNTIME_ENTRYPOINT = "runtime.entrypoint:main"
OFFICIAL_RUNTIME_BOOT_COMMAND = "python -m runtime"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proculus official Pipeline V2 runtime")
    parser.add_argument("--mode", default="paper", choices=["paper", "sim", "demo", "dry-run", "live"])
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--once", action="store_true", help="Run one analysis/decision cycle and exit")
    return parser


def main() -> None:
    args = _parser().parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    result = asyncio.run(main_service(symbols=symbols, runtime_mode=args.mode, once=args.once))
    if args.once:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
