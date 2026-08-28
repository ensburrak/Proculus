# -*- coding: utf-8 -*-
"""
monte_carlo.py - Monte Carlo Validation

[P3.1] Monte Carlo simulation for strategy validation.
Provides confidence intervals and robustness testing.
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from atomic_io import atomic_write_json

import logging

try:
    from logger import get_logger
    log = get_logger("monte_carlo")
except ImportError:
    log = logging.getLogger("monte_carlo")


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    mean_return: float
    std_return: float
    mean_sharpe: float
    std_sharpe: float
    mean_max_dd: float
    std_max_dd: float
    win_rate_mean: float
    confidence_95_return: Tuple[float, float]
    confidence_95_sharpe: Tuple[float, float]
    probability_of_profit: float
    probability_of_ruin: float  # P(DD > 50%)
    var_95: float
    cvar_95: float
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Monte Carlo Simulation Results ({self.n_simulations} runs)
{'='*50}
Returns:
  Mean: {self.mean_return*100:.2f}% (std: {self.std_return*100:.2f}%)
  95% CI: [{self.confidence_95_return[0]*100:.2f}%, {self.confidence_95_return[1]*100:.2f}%]

Sharpe Ratio:
  Mean: {self.mean_sharpe:.2f} (std: {self.std_sharpe:.2f})
  95% CI: [{self.confidence_95_sharpe[0]:.2f}, {self.confidence_95_sharpe[1]:.2f}]

Risk Metrics:
  Max Drawdown: {self.mean_max_dd*100:.1f}% (std: {self.std_max_dd*100:.1f}%)
  VaR (95%): {self.var_95*100:.2f}%
  CVaR (95%): {self.cvar_95*100:.2f}%

Probabilities:
  Profit: {self.probability_of_profit*100:.1f}%
  Ruin (DD>50%): {self.probability_of_ruin*100:.1f}%
  Win Rate: {self.win_rate_mean*100:.1f}%
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "n_simulations": self.n_simulations,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "mean_max_dd": self.mean_max_dd,
            "probability_of_profit": self.probability_of_profit,
            "probability_of_ruin": self.probability_of_ruin,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategy validation.
    
    Methods:
    - Bootstrap resampling of trade returns
    - Random path generation
    - Confidence interval calculation
    - Probability distributions
    """
    
    def __init__(self, n_simulations: int = 1000, seed: Optional[int] = None):
        self._n_simulations = n_simulations
        self._rng = random.Random(seed)
    
    def simulate_bootstrap(
        self,
        trade_returns: List[float],
        n_trades_per_sim: Optional[int] = None,
        initial_capital: float = 10000,
        block_size: int = 5,
    ) -> MonteCarloResult:
        """
        Run bootstrap Monte Carlo simulation.

        [FIX] Block bootstrap: ardışık trade'leri blok halinde örnekler
        (korelasyonu korur). Bileşik büyüme ile equity hesaplanır.

        Args:
            trade_returns: List of trade returns (as decimals)
            n_trades_per_sim: Trades per simulation (default: same as input)
            initial_capital: Starting capital
            block_size: Block bootstrap blok boyutu (varsayılan=5)

        Returns:
            MonteCarloResult with statistics
        """
        if not trade_returns or len(trade_returns) < 10:
            log.warning("[MC] Insufficient trade history for simulation")
            return self._empty_result()

        n_trades = n_trades_per_sim or len(trade_returns)

        sim_returns = []
        sim_sharpes = []
        sim_max_dds = []
        sim_win_rates = []

        for _ in range(self._n_simulations):
            # [FIX] Block bootstrap — korelasyonu korur
            sampled = []
            while len(sampled) < n_trades:
                start = self._rng.randint(0, max(0, len(trade_returns) - block_size))
                block = trade_returns[start:start + block_size]
                sampled.extend(block)
            sampled = sampled[:n_trades]

            # [FIX] Bileşik büyüme ile equity hesaplama
            equity = [initial_capital]
            for ret in sampled:
                equity.append(equity[-1] * (1 + ret))

            total_return = (equity[-1] / equity[0]) - 1
            max_dd = self._calculate_max_drawdown(equity)
            sharpe = self._calculate_sharpe(sampled)
            win_rate = sum(1 for r in sampled if r > 0) / len(sampled)

            sim_returns.append(total_return)
            sim_sharpes.append(sharpe)
            sim_max_dds.append(max_dd)
            sim_win_rates.append(win_rate)

        return self._compile_results(
            sim_returns, sim_sharpes, sim_max_dds, sim_win_rates
        )
    
    def simulate_random_walks(
        self,
        mean_return: float,
        std_return: float,
        n_trades: int,
        initial_capital: float = 10000,
    ) -> MonteCarloResult:
        """
        Run random walk Monte Carlo simulation.
        
        Generates random returns from normal distribution.
        
        Args:
            mean_return: Expected return per trade
            std_return: Standard deviation of returns
            n_trades: Number of trades per simulation
            initial_capital: Starting capital
        
        Returns:
            MonteCarloResult with statistics
        """
        sim_returns = []
        sim_sharpes = []
        sim_max_dds = []
        sim_win_rates = []
        
        for _ in range(self._n_simulations):
            # Generate random returns
            returns = [self._rng.gauss(mean_return, std_return) for _ in range(n_trades)]
            
            # Calculate equity curve
            equity = [initial_capital]
            for ret in returns:
                equity.append(equity[-1] * (1 + ret))
            
            # Metrics
            total_return = (equity[-1] / equity[0]) - 1
            max_dd = self._calculate_max_drawdown(equity)
            sharpe = self._calculate_sharpe(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            
            sim_returns.append(total_return)
            sim_sharpes.append(sharpe)
            sim_max_dds.append(max_dd)
            sim_win_rates.append(win_rate)
        
        return self._compile_results(
            sim_returns, sim_sharpes, sim_max_dds, sim_win_rates
        )
    
    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity[0]
        max_dd = 0
        
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe(self, returns: List[float], rf: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.001
        
        # Annualize assuming daily trades
        annual_factor = math.sqrt(252)
        sharpe = ((mean_ret - rf) / std_ret) * annual_factor
        
        return sharpe
    
    def _compile_results(
        self,
        returns: List[float],
        sharpes: List[float],
        max_dds: List[float],
        win_rates: List[float],
    ) -> MonteCarloResult:
        """Compile simulation results into MonteCarloResult."""
        n = len(returns)
        
        # Returns statistics
        mean_return = sum(returns) / n
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / n)
        
        # Sharpe statistics
        mean_sharpe = sum(sharpes) / n
        std_sharpe = math.sqrt(sum((s - mean_sharpe) ** 2 for s in sharpes) / n)
        
        # Drawdown statistics
        mean_max_dd = sum(max_dds) / n
        std_max_dd = math.sqrt(sum((d - mean_max_dd) ** 2 for d in max_dds) / n)
        
        # Win rate
        win_rate_mean = sum(win_rates) / n
        
        # Confidence intervals (95%)
        sorted_returns = sorted(returns)
        sorted_sharpes = sorted(sharpes)
        ci_low_idx = int(0.025 * n)
        ci_high_idx = int(0.975 * n)
        
        ci_return = (sorted_returns[ci_low_idx], sorted_returns[ci_high_idx])
        ci_sharpe = (sorted_sharpes[ci_low_idx], sorted_sharpes[ci_high_idx])
        
        # Probabilities
        prob_profit = sum(1 for r in returns if r > 0) / n
        prob_ruin = sum(1 for d in max_dds if d > 0.5) / n
        
        # VaR/CVaR
        var_idx = int(0.05 * n)
        var_95 = abs(sorted_returns[var_idx])
        cvar_95 = abs(sum(sorted_returns[:var_idx + 1]) / (var_idx + 1)) if var_idx > 0 else var_95
        
        return MonteCarloResult(
            n_simulations=n,
            mean_return=mean_return,
            std_return=std_return,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_max_dd=mean_max_dd,
            std_max_dd=std_max_dd,
            win_rate_mean=win_rate_mean,
            confidence_95_return=ci_return,
            confidence_95_sharpe=ci_sharpe,
            probability_of_profit=prob_profit,
            probability_of_ruin=prob_ruin,
            var_95=var_95,
            cvar_95=cvar_95,
        )
    
    def _empty_result(self) -> MonteCarloResult:
        """Return empty result for insufficient data."""
        return MonteCarloResult(
            n_simulations=0,
            mean_return=0,
            std_return=0,
            mean_sharpe=0,
            std_sharpe=0,
            mean_max_dd=0,
            std_max_dd=0,
            win_rate_mean=0,
            confidence_95_return=(0, 0),
            confidence_95_sharpe=(0, 0),
            probability_of_profit=0,
            probability_of_ruin=1,
            var_95=0,
            cvar_95=0,
        )
    
    def save_results(self, result: MonteCarloResult, path: str = "metrics/monte_carlo.json") -> None:
        """Save simulation results to file."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(p, result.to_dict())
            log.info(f"[MC] Results saved to {path}")
        except BEST_EFFORT_EXCEPTIONS as e:
            log.error(f"[MC] Save error: {e}")


def run_monte_carlo(
    trade_returns: List[float],
    n_simulations: int = 1000,
    seed: Optional[int] = 42,
    block_size: int = 5,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo simulation.

    Args:
        trade_returns: List of trade returns
        n_simulations: Number of simulations
        seed: Random seed for reproducibility
        block_size: Block bootstrap boyutu

    Returns:
        MonteCarloResult
    """
    simulator = MonteCarloSimulator(n_simulations=n_simulations, seed=seed)
    return simulator.simulate_bootstrap(trade_returns, block_size=block_size)


def simulate_trades(trade_returns: List[float], n_sims: int = 1000) -> MonteCarloResult:
    """Backward-compatible alias used by package backtest validation helpers."""
    return run_monte_carlo(trade_returns, n_sims)


__all__ = [
    "MonteCarloSimulator",
    "MonteCarloResult",
    "run_monte_carlo",
    "simulate_trades",
]
