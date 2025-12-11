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

from .exceptions import InvalidParameterError, InvalidDemandError


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


class DemandSpikeProcess(DemandProcess):
    """
    Demand process with random spike events.
    
    D_t = D_base_t + spike_t * (spike_multiplier - 1) * D_base_t
    
    where spike_t ~ Bernoulli(spike_prob).
    
    When a spike occurs, demand is multiplied by spike_multiplier.
    
    Attributes:
        base_process: Underlying demand process
        spike_prob: Probability of a spike occurring in any given period
        spike_multiplier: Multiplier applied during spikes (e.g., 3.0 = 3x demand)
    """
    
    def __init__(
        self,
        base_process: DemandProcess,
        spike_prob: float = 0.05,
        spike_multiplier: float = 3.0
    ):
        if not (0 <= spike_prob <= 1):
            raise InvalidDemandError(f"Spike probability must be in [0, 1], got {spike_prob}")
        if spike_multiplier < 1:
            raise InvalidDemandError(f"Spike multiplier must be >= 1, got {spike_multiplier}")
        
        self.base_process = base_process
        self.spike_prob = spike_prob
        self.spike_multiplier = spike_multiplier
        self._current_spike = False  # Track if currently in a spike
    
    def _is_spike(self) -> bool:
        """Sample whether a spike occurs."""
        self._current_spike = np.random.random() < self.spike_prob
        return self._current_spike
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        base_demand = self.base_process.sample(exogenous_state)
        if self._is_spike():
            return base_demand * self.spike_multiplier
        return base_demand
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Expected demand accounting for spike probability."""
        base_mean = self.base_process.mean(exogenous_state)
        # E[D] = (1 - p) * base + p * base * multiplier
        return base_mean * (1 - self.spike_prob + self.spike_prob * self.spike_multiplier)
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Variance of demand accounting for spike probability."""
        base_mean = self.base_process.mean(exogenous_state)
        base_var = self.base_process.variance(exogenous_state)
        
        # Using law of total variance
        # Var[D] = E[Var[D|spike]] + Var[E[D|spike]]
        var_given_no_spike = base_var
        var_given_spike = base_var * (self.spike_multiplier ** 2)
        
        mean_given_no_spike = base_mean
        mean_given_spike = base_mean * self.spike_multiplier
        
        expected_var = (1 - self.spike_prob) * var_given_no_spike + self.spike_prob * var_given_spike
        var_of_means = (
            (1 - self.spike_prob) * mean_given_no_spike ** 2 + 
            self.spike_prob * mean_given_spike ** 2 -
            self.mean(exogenous_state) ** 2
        )
        
        return expected_var + var_of_means
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        """PMF is complex for spike process - use base approximation."""
        # This is an approximation; exact PMF requires convolution
        return self.base_process.pmf(d, exogenous_state)
    
    def update_exogenous_state(
        self,
        current_state: Optional[np.ndarray],
        noise: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Delegate to base process."""
        return self.base_process.update_exogenous_state(current_state, noise)


class TrendDemand(DemandProcess):
    """
    Demand process with trend component.
    
    λ_t = base_rate * (1 + trend_rate * t)^trend_power
    
    Supports linear (power=1) or exponential-like growth/decline.
    
    Attributes:
        base_rate: Base demand rate at t=0
        trend_rate: Rate of change per period (positive = growth, negative = decline)
        trend_power: Power of trend (1 = linear, >1 = accelerating)
        min_rate: Minimum demand rate floor
        max_rate: Maximum demand rate ceiling
    """
    
    def __init__(
        self,
        base_rate: float,
        trend_rate: float = 0.01,
        trend_power: float = 1.0,
        min_rate: float = 1.0,
        max_rate: float = float('inf')
    ):
        if base_rate < 0:
            raise InvalidDemandError(f"Base rate must be non-negative, got {base_rate}")
        if min_rate < 0:
            raise InvalidDemandError(f"Min rate must be non-negative, got {min_rate}")
        
        self.base_rate = base_rate
        self.trend_rate = trend_rate
        self.trend_power = trend_power
        self.min_rate = min_rate
        self.max_rate = max_rate
    
    def _get_rate(self, exogenous_state: Optional[np.ndarray]) -> float:
        if exogenous_state is None or len(exogenous_state) == 0:
            return self.base_rate
        
        t = exogenous_state[0]  # First element is time
        trend_factor = (1 + self.trend_rate * t) ** self.trend_power
        rate = self.base_rate * max(0, trend_factor)  # Ensure non-negative
        
        # Apply floor and ceiling
        return np.clip(rate, self.min_rate, self.max_rate)
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return float(np.random.poisson(max(0, rate)))
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return self._get_rate(exogenous_state)
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        return self._get_rate(exogenous_state)  # Poisson variance = mean
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return stats.poisson.pmf(d, max(0.01, rate))  # Avoid zero rate
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state)
        return stats.poisson.cdf(d, max(0.01, rate))
    
    def update_exogenous_state(
        self,
        current_state: Optional[np.ndarray],
        noise: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Advance time by one period."""
        if current_state is None:
            return np.array([1.0])
        new_state = current_state.copy()
        new_state[0] += 1
        return new_state


class CompositeDemand(DemandProcess):
    """
    Composite demand combining multiple patterns.
    
    D_t = base_rate * seasonal_factor * trend_factor * crisis_factor + spike_effect
    
    Supports:
    - Seasonality (sinusoidal pattern)
    - Trend (linear/exponential growth/decline)
    - Spikes (random surge events)
    - Crisis modulation (from exogenous state)
    
    Attributes:
        base_rate: Base demand rate
        seasonality_amplitude: Seasonal variation (0 to 1)
        seasonality_period: Period length for seasonality
        seasonality_phase: Phase offset for seasonality
        trend_rate: Trend rate per period
        spike_prob: Probability of demand spike
        spike_multiplier: Multiplier during spikes
        crisis_multipliers: List of multipliers for crisis states [normal, elevated, crisis]
    """
    
    def __init__(
        self,
        base_rate: float,
        seasonality_amplitude: float = 0.0,
        seasonality_period: int = 12,
        seasonality_phase: float = 0.0,
        trend_rate: float = 0.0,
        spike_prob: float = 0.0,
        spike_multiplier: float = 2.0,
        crisis_multipliers: Optional[Tuple[float, float, float]] = None
    ):
        if base_rate < 0:
            raise InvalidDemandError(f"Base rate must be non-negative, got {base_rate}")
        if not (0 <= seasonality_amplitude < 1):
            raise InvalidDemandError(f"Seasonality amplitude must be in [0, 1), got {seasonality_amplitude}")
        if seasonality_period <= 0:
            raise InvalidDemandError(f"Seasonality period must be positive, got {seasonality_period}")
        if not (0 <= spike_prob <= 1):
            raise InvalidDemandError(f"Spike probability must be in [0, 1], got {spike_prob}")
        
        self.base_rate = base_rate
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.seasonality_phase = seasonality_phase
        self.trend_rate = trend_rate
        self.spike_prob = spike_prob
        self.spike_multiplier = spike_multiplier
        self.crisis_multipliers = crisis_multipliers or (1.0, 1.5, 3.0)
        
        self._current_spike = False
    
    def _get_time(self, exogenous_state: Optional[np.ndarray]) -> float:
        """Extract time from exogenous state."""
        if exogenous_state is None or len(exogenous_state) == 0:
            return 0.0
        return exogenous_state[0]
    
    def _get_crisis_state(self, exogenous_state: Optional[np.ndarray]) -> int:
        """Extract crisis state from exogenous state (index 1 if present)."""
        if exogenous_state is None or len(exogenous_state) < 2:
            return 0  # Normal state
        return int(exogenous_state[1])
    
    def _get_rate(self, exogenous_state: Optional[np.ndarray], include_spike: bool = False) -> float:
        """Calculate demand rate with all modifiers."""
        t = self._get_time(exogenous_state)
        crisis_state = self._get_crisis_state(exogenous_state)
        
        # Start with base rate
        rate = self.base_rate
        
        # Apply seasonality
        if self.seasonality_amplitude > 0:
            seasonal_factor = 1 + self.seasonality_amplitude * np.sin(
                2 * np.pi * t / self.seasonality_period + self.seasonality_phase
            )
            rate *= seasonal_factor
        
        # Apply trend
        if self.trend_rate != 0:
            trend_factor = 1 + self.trend_rate * t
            rate *= max(0.1, trend_factor)  # Floor to prevent negative
        
        # Apply crisis modifier
        crisis_multiplier = self.crisis_multipliers[min(crisis_state, len(self.crisis_multipliers) - 1)]
        rate *= crisis_multiplier
        
        # Apply spike (only for sampling, not for mean calculation)
        if include_spike and self.spike_prob > 0:
            self._current_spike = np.random.random() < self.spike_prob
            if self._current_spike:
                rate *= self.spike_multiplier
        
        return max(0, rate)
    
    def sample(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state, include_spike=True)
        return float(np.random.poisson(max(0, rate)))
    
    def mean(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Expected demand (accounts for spike probability)."""
        base_rate = self._get_rate(exogenous_state, include_spike=False)
        if self.spike_prob > 0:
            return base_rate * (1 - self.spike_prob + self.spike_prob * self.spike_multiplier)
        return base_rate
    
    def variance(self, exogenous_state: Optional[np.ndarray] = None) -> float:
        """Variance of demand."""
        # For Poisson mixture, this is an approximation
        return self.mean(exogenous_state)
    
    def pmf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state, include_spike=False)
        return stats.poisson.pmf(d, max(0.01, rate))
    
    def cdf(self, d: int, exogenous_state: Optional[np.ndarray] = None) -> float:
        rate = self._get_rate(exogenous_state, include_spike=False)
        return stats.poisson.cdf(d, max(0.01, rate))
    
    def update_exogenous_state(
        self,
        current_state: Optional[np.ndarray],
        noise: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Advance time by one period (crisis state is managed externally)."""
        if current_state is None:
            return np.array([1.0, 0.0])  # [time, crisis_state]
        new_state = current_state.copy()
        new_state[0] += 1  # Increment time
        # Crisis state (index 1) is managed by CrisisProcess externally
        return new_state


def create_demand_scenario(
    base_demand: float,
    scenario_type: str = "stationary",
    **kwargs
) -> DemandProcess:
    """
    Factory function to create demand processes for different scenarios.
    
    Args:
        base_demand: Base demand rate
        scenario_type: One of "stationary", "seasonal", "overdispersed", 
                       "spiky", "trending", "composite"
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
    
    elif scenario_type == "spiky":
        # Demand with random spike events
        base_process = PoissonDemand(base_demand)
        return DemandSpikeProcess(
            base_process,
            spike_prob=kwargs.get("spike_prob", 0.05),
            spike_multiplier=kwargs.get("spike_multiplier", 3.0)
        )
    
    elif scenario_type == "trending":
        return TrendDemand(
            base_demand,
            trend_rate=kwargs.get("trend_rate", 0.01),
            trend_power=kwargs.get("trend_power", 1.0),
            min_rate=kwargs.get("min_rate", 1.0),
            max_rate=kwargs.get("max_rate", float('inf'))
        )
    
    elif scenario_type == "composite":
        return CompositeDemand(
            base_demand,
            seasonality_amplitude=kwargs.get("seasonality_amplitude", 0.0),
            seasonality_period=kwargs.get("seasonality_period", 12),
            seasonality_phase=kwargs.get("seasonality_phase", 0.0),
            trend_rate=kwargs.get("trend_rate", 0.0),
            spike_prob=kwargs.get("spike_prob", 0.0),
            spike_multiplier=kwargs.get("spike_multiplier", 2.0),
            crisis_multipliers=kwargs.get("crisis_multipliers", None)
        )
    
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

