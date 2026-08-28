"""
background_worker_runner.py compatibility module
-----------------------------------------------

This module provides a simple wrapper for running the bot in a background worker
architecture.  While the main codebase is organised as a monolithic
application, the underlying logic can be decomposed into in-process workers
(data collection, signal generation, order execution).  This runner
implements a cooperative, in‑memory queue based approach for
worker-mode compatibility purposes.

The classes ``DataCollector``, ``SignalGenerator`` and ``OrderExecutor``
are imported from ``background_workers.py``.  They each expose ``step()``
methods that process tasks from asynchronous queues.  The
``BackgroundWorkerManager`` instantiates these workers and runs them in
threads.  Each worker communicates via Python lists used as queues.

To enable background worker mode set the environment variable
``USE_BACKGROUND_WORKERS=1`` before starting ``main_bot_async.py``.  When
enabled, the main loop will start the background worker manager and
periodically feed it with the required inputs.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import os
import threading
import time
from typing import Any, Dict, List

try:
    from background_workers import DataCollector, SignalGenerator, OrderExecutor  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    # Define fallback workers if missing.
    class DataCollector:
        def __init__(self, in_queue: List[Any], out_queue: List[Any]):
            self.in_queue = in_queue
            self.out_queue = out_queue
        def step(self):
            # Consume price fetch tasks and produce market data
            if self.in_queue:
                self.in_queue.pop(0)
                # Fail closed: do not publish synthetic price data.
                return
    class SignalGenerator:
        def __init__(self, in_queue: List[Any], out_queue: List[Any]):
            self.in_queue = in_queue
            self.out_queue = out_queue
        def step(self):
            if self.in_queue:
                data = self.in_queue.pop(0)
                self.out_queue.append({"symbol": data["symbol"], "signal": 0})
    class OrderExecutor:
        def __init__(self, in_queue: List[Any]):
            self.in_queue = in_queue
        def step(self):
            if self.in_queue:
                order = self.in_queue.pop(0)
                # In real implementation, send order via ccxt
                print(f"[ORDER] Executed {order}")

logger = logging.getLogger(__name__)


class BackgroundWorkerManager:
    """Manage and run background workers using shared in-memory queues.

    This class attempts to adapt between the skeleton workers defined
    in ``background_workers.py`` (which expect callables for subscribe/publish)
    and the simple list‑based fallback used for demonstration.  If the
    imported ``DataCollector`` class defines a ``publish`` parameter in its
    ``__init__`` signature, callables are wired up to internal lists.  If
    not, the legacy list‑based interface is used.  This prevents errors
    like ``'list' object is not callable`` when running with the skeleton
    workers.
    """

    def __init__(self) -> None:
        # Shared queues for passing messages between services
        self.q_price_req: List[Any] = []  # price requests (used only by fallback)
        self.q_price_data: List[Any] = []  # price data from collector → generator
        self.q_orders: List[Any] = []  # signals/orders from generator → executor
        # Attempt to adapt to worker classes that accept callables
        try:
            import inspect
            sig = inspect.signature(DataCollector.__init__)
            param_names = [p.name for p in list(sig.parameters.values())[1:]]  # skip 'self'
            # Skeleton DataCollector declares a 'publish' parameter
            if 'publish' in param_names and 'in_queue' not in param_names:
                # Wire callables to internal queues
                def publish_price(data: Any) -> None:
                    self.q_price_data.append(data)

                def subscribe_price() -> Any:
                    if self.q_price_data:
                        return self.q_price_data.pop(0)
                    return None

                def publish_signal(sig: Any) -> None:
                    self.q_orders.append(sig)

                def subscribe_order() -> Any:
                    if self.q_orders:
                        return self.q_orders.pop(0)
                    return None

                # Instantiate skeleton services with callables
                self.collector = DataCollector(publish_price)
                # SignalGenerator in skeleton expects (subscribe, publish)
                self.generator = SignalGenerator(subscribe_price, publish_signal)
                # OrderExecutor in skeleton expects subscribe only
                self.executor = OrderExecutor(subscribe_order)
            else:
                # Fallback to list‑based interface (imported classes expect queues)
                self.collector = DataCollector(self.q_price_req, self.q_price_data)  # type: ignore
                # In fallback mode, generator writes signals to q_orders directly
                self.generator = SignalGenerator(self.q_price_data, self.q_orders)  # type: ignore
                self.executor = OrderExecutor(self.q_orders)  # type: ignore
        except BEST_EFFORT_EXCEPTIONS:
            # In case of inspection failure, assume fallback interface
            self.collector = DataCollector(self.q_price_req, self.q_price_data)  # type: ignore
            self.generator = SignalGenerator(self.q_price_data, self.q_orders)  # type: ignore
            self.executor = OrderExecutor(self.q_orders)  # type: ignore
        self._stop_event = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run_service, args=(self.collector,), daemon=True).start()
        threading.Thread(target=self._run_service, args=(self.generator,), daemon=True).start()
        threading.Thread(target=self._run_service, args=(self.executor,), daemon=True).start()
        logger.info("Background workers started.")

    def stop(self) -> None:
        self._stop_event.set()

    def _run_service(self, service: Any) -> None:
        while not self._stop_event.is_set():
            try:
                service.step()
            except BEST_EFFORT_EXCEPTIONS as exc:
                logger.error("Background worker error: %s", exc)
            time.sleep(0.1)

    def add_price_request(self, symbol: str) -> None:
        # Only meaningful for list‑based fallback; worker classes
        # continuously produce data and signals.
        self.q_price_req.append(symbol)

    def add_order(self, order: Dict[str, Any]) -> None:
        self.q_orders.append(order)


def run_background_workers() -> BackgroundWorkerManager:
    """Instantiate and start background workers, returning the manager."""
    mgr = BackgroundWorkerManager()
    mgr.start()
    return mgr


MicroserviceManager = BackgroundWorkerManager


def run_microservices() -> BackgroundWorkerManager:
    """Backward-compatible alias for older runtime imports."""
    return run_background_workers()


if __name__ == "__main__":
    mgr = run_background_workers()
    # Example demonstration: periodically request prices and process signals
    symbols = ["BTC/USDT", "ETH/USDT"]
    try:
        while True:
            for s in symbols:
                mgr.add_price_request(s)
            time.sleep(5)
    except KeyboardInterrupt:
        mgr.stop()
