from __future__ import annotations

from typing import Any


class DecisionPipeline:
    """Canonical Proculus decision entrypoint.

    Keeps callers insulated from controller internals while preserving the
    existing dictionary decision contract.
    """

    def __init__(self, exchange: Any = None) -> None:
        self.exchange = exchange

    async def decide_batch(self, symbol_inputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        from controller_async import decide_batch

        return await decide_batch(symbol_inputs, exchange=self.exchange)
