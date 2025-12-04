"""
Demand Process for the Perishable Inventory MDP

Implements stochastic demand models:
D_t ~ F(· | z_t)

Supports Poisson, Negative Binomial, and state-space models.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from scipy import stats

from ..exceptions import InvalidParameterError, InvalidDemandError


class DemandProcess(ABC):
    """
    Abstract base class for demand processes.
    
    D_t ~ F(· | z_t)
    """
    
    @abstractmethod
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """
        Sample demand given current exogenous state.
        
        Args:
            exogenous_state: External state z_t
        
        Returns:
            Sampled demand D_t
        """
        pass
    
    @abstractmethod
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Expected demand E[D_t | z_t]"""
        pass
    
    @abstractmethod
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Variance of demand Var[D_t | z_t]"""
        pass
    
    def std(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Standard deviation of demand"""
        return np.sqrt(self.variance(exogenous_state))
    
    @abstractmethod
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Probability mass function P(D_t = d | z_t)"""
        pass
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Cumulative distribution function P(D_t ≤ d | z_t)"""
        return sum(self.pmf(k, exogenous_state) for k in range(d + 1))
    
    def update_exogenous_state(
        self,
        current_state: Optional[np.ndarray],
        noise: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Update exogenous state: z_{t+1} = g(z_t, ε_t)
        
        Default implementation: state remains unchanged.
        Override in subclasses for dynamic state evolution.
        """
        return current_state


class PoissonDemand(DemandProcess):
    """
    Poisson demand process.
    
    D_t ~ Poisson(λ(z_t))
    
    Attributes:
        base_rate: Base demand rate λ_0
        seasonality_fn: Optional function mapping exogenous state to rate multiplier
    """
    
    def __init__(
        self,
        base_rate: float,
        seasonality_fn: Optional[Callable[[np.ndarray], float]] = None
    ):
        if base_rate < 0:
            raise InvalidDemandError(f"Base rate must be non-negative, got {base_rate}")
        self.base_rate = base_rate
        self.seasonality_fn = seasonality_fn
    
    def _get_rate(self, exogenous_state: Optional[np.ndarray]) -> float:
        """Get demand rate λ(z_t)"""
        if self.seasonality_fn is not None and exogenous_state is not None:
            return self.base_rate * self.seasonality_fn(exogenous_state)
        return self.base_rate
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return float(np.random.poisson(rate))
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return self._get_rate(exogenous_state)
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return self._get_rate(exogenous_state)  # Var = λ for Poisson
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return stats.poisson.pmf(d, rate)
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Efficient CDF using scipy"""
        rate = self._get_rate(exogenous_state)
        return stats.poisson.cdf(d, rate)


class NegativeBinomialDemand(DemandProcess):
    """
    Negative Binomial demand process.
    
    D_t ~ NegBin(r, p(z_t))
    
    Good for over-dispersed demand (variance > mean).
    
    Attributes:
        n_successes: Number of successes parameter r
        prob_success: Success probability p (can be state-dependent)
    """
    
    def __init__(
        self,
        n_successes: float,
        prob_success: float = 0.5,
        prob_fn: Optional[Callable[[np.ndarray], float]] = None
    ):
        if n_successes <= 0:
            raise InvalidDemandError(f"Number of successes must be positive, got {n_successes}")
        if not (0 < prob_success < 1):
            raise InvalidDemandError(f"Probability must be in (0, 1), got {prob_success}")
        self.n_successes = n_successes
        self.base_prob = prob_success
        self.prob_fn = prob_fn
    
    def _get_prob(self, exogenous_state: Optional[np.ndarray]) -> float:
        """Get success probability p(z_t)"""
        if self.prob_fn is not None and exogenous_state is not None:
            return self.prob_fn(exogenous_state)
        return self.base_prob
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        p = self._get_prob(exogenous_state)
        return float(np.random.negative_binomial(self.n_successes, p))
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        p = self._get_prob(exogenous_state)
        return self.n_successes * (1 - p) / p
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        p = self._get_prob(exogenous_state)
        return self.n_successes * (1 - p) / (p ** 2)
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        p = self._get_prob(exogenous_state)
        return stats.nbinom.pmf(d, self.n_successes, p)
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Efficient CDF using scipy"""
        p = self._get_prob(exogenous_state)
        return stats.nbinom.cdf(d, self.n_successes, p)


class SeasonalDemand(DemandProcess):
    """
    Demand process with deterministic seasonal pattern.
    
    λ_t = λ_0 * (1 + A * sin(2π * t / T + φ))
    
    where T is the period length.
    """
    
    def __init__(
        self,
        base_rate: float,
        amplitude: float = 0.3,
        period: int = 12,
        phase: float = 0.0
    ):
        if base_rate < 0:
            raise InvalidDemandError(f"Base rate must be non-negative, got {base_rate}")
        if amplitude < 0 or amplitude >= 1:
            raise InvalidDemandError(f"Amplitude must be in [0, 1), got {amplitude}")
        if period <= 0:
            raise InvalidDemandError(f"Period must be positive, got {period}")
        self.base_rate = base_rate
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
    
    def _get_rate(self, exogenous_state: Optional[np.ndarray]) -> float:
        if exogenous_state is None or len(exogenous_state) == 0:
            return self.base_rate
        
        t = exogenous_state[0]  # First element is time
        seasonal_factor = 1 + self.amplitude * np.sin(
            2 * np.pi * t / self.period + self.phase
        )
        return self.base_rate * seasonal_factor
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return float(np.random.poisson(max(0, rate)))
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return max(0, self._get_rate(exogenous_state))
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return self.mean(exogenous_state)
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return stats.poisson.pmf(d, max(0, rate))
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Efficient CDF using scipy"""
        rate = self._get_rate(exogenous_state)
        return stats.poisson.cdf(d, max(0, rate))
    
    def update_exogenous_state(
        self,
        current_state: Optional[np.ndarray],
        noise: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Advance time by one period"""
        if current_state is None:
            return np.array([1.0])
        new_state = current_state.copy()
        new_state[0] += 1
        return new_state


@dataclass
class StochasticLeadTime:
    """
    Stochastic lead time model using Bernoulli advancement.
    
    ξ_t^(s) ~ Bernoulli(p_s)
    
    If ξ = 1, pipeline advances; if ξ = 0, pipeline stays in place.
    
    Attributes:
        supplier_id: Supplier identifier
        advancement_prob: Probability of pipeline advancement p_s
        transition_matrix: Optional state-dependent transition matrix Π^(s)
    """
    supplier_id: int
    advancement_prob: float = 1.0  # Deterministic by default
    transition_matrix: Optional[np.ndarray] = None
    
    def sample_advancement(self, current_position: int = 0) -> bool:
        """
        Sample whether the pipeline advances this period.
        
        Args:
            current_position: Current position in lead-time pipeline
        
        Returns:
            True if pipeline advances, False otherwise
        """
        if self.transition_matrix is not None:
            # State-dependent transition
            prob = self.transition_matrix[current_position, max(0, current_position - 1)]
            return np.random.random() < prob
        else:
            # Simple Bernoulli
            return np.random.random() < self.advancement_prob


def create_demand_scenario(
    base_demand: float,
    scenario_type: str = "stationary",
    **kwargs
) -> DemandProcess:
    """
    Factory function to create demand processes for different scenarios.
    
    Args:
        base_demand: Base demand rate
        scenario_type: One of "stationary", "seasonal", "overdispersed"
        **kwargs: Additional parameters for specific scenario types
    
    Returns:
        Configured DemandProcess
    """
    if scenario_type == "stationary":
        return PoissonDemand(base_demand)
    
    elif scenario_type == "seasonal":
        return SeasonalDemand(
            base_demand,
            amplitude=kwargs.get("amplitude", 0.3),
            period=kwargs.get("period", 12),
            phase=kwargs.get("phase", 0.0)
        )
    
    elif scenario_type == "overdispersed":
        # Parameterize NegBin to match mean = base_demand
        # Mean = r(1-p)/p, so p = r/(r + mean)
        r = kwargs.get("dispersion", 5.0)
        p = r / (r + base_demand)
        return NegativeBinomialDemand(r, p)
    
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

