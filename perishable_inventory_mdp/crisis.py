"""
Crisis Process for the Perishable Inventory MDP

Models global crisis events affecting both demand and supply.

Crisis states:
- 0: Normal - no disruption
- 1: Elevated - moderate impact on demand/supply
- 2: Crisis - severe impact on demand/supply
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum

from .exceptions import InvalidParameterError


class CrisisLevel(IntEnum):
    """Enumeration of crisis severity levels."""
    NORMAL = 0
    ELEVATED = 1
    CRISIS = 2


@dataclass
class CrisisEvent:
    """
    Definition of a crisis state and its effects.
    
    Attributes:
        level: Crisis severity level (0=normal, 1=elevated, 2=crisis)
        name: Human-readable name for the crisis state
        demand_multiplier: Factor to multiply demand by (e.g., 2.0 = doubles demand)
        supply_disruption_prob: Probability that supplier orders are rejected
        duration_mean: Average duration in periods before transitioning out
        duration_std: Standard deviation of duration
    """
    level: int
    name: str = ""
    demand_multiplier: float = 1.0
    supply_disruption_prob: float = 0.0
    duration_mean: int = 1
    duration_std: float = 0.0
    
    def __post_init__(self):
        if self.demand_multiplier < 0:
            raise InvalidParameterError(
                f"Demand multiplier must be non-negative, got {self.demand_multiplier}"
            )
        if not (0 <= self.supply_disruption_prob <= 1):
            raise InvalidParameterError(
                f"Supply disruption probability must be in [0, 1], got {self.supply_disruption_prob}"
            )
        if self.duration_mean < 1:
            raise InvalidParameterError(
                f"Duration mean must be at least 1, got {self.duration_mean}"
            )
        if self.duration_std < 0:
            raise InvalidParameterError(
                f"Duration std must be non-negative, got {self.duration_std}"
            )


# Predefined crisis events
NORMAL_STATE = CrisisEvent(
    level=CrisisLevel.NORMAL,
    name="Normal",
    demand_multiplier=1.0,
    supply_disruption_prob=0.0,
    duration_mean=20
)

ELEVATED_STATE = CrisisEvent(
    level=CrisisLevel.ELEVATED,
    name="Elevated",
    demand_multiplier=1.5,
    supply_disruption_prob=0.15,
    duration_mean=10
)

CRISIS_STATE = CrisisEvent(
    level=CrisisLevel.CRISIS,
    name="Crisis",
    demand_multiplier=3.0,
    supply_disruption_prob=0.40,
    duration_mean=5
)


@dataclass
class CrisisProcess:
    """
    Markov chain for crisis state transitions.
    
    Models transitions between crisis states using a transition matrix.
    
    Attributes:
        states: List of CrisisEvent definitions [normal, elevated, crisis]
        transition_matrix: 3x3 matrix of transition probabilities
                          transition_matrix[i, j] = P(next_state = j | current_state = i)
        current_state: Current crisis level
        periods_in_state: Number of periods in current state
    """
    states: List[CrisisEvent] = field(default_factory=lambda: [
        NORMAL_STATE, ELEVATED_STATE, CRISIS_STATE
    ])
    transition_matrix: Optional[np.ndarray] = None
    current_state: int = 0
    periods_in_state: int = 0
    
    def __post_init__(self):
        if self.transition_matrix is None:
            # Default transition matrix:
            # From Normal: high prob stay normal, low prob go elevated, very low prob crisis
            # From Elevated: can go up or down, moderate stay prob
            # From Crisis: mostly stay or improve, rarely get worse
            self.transition_matrix = np.array([
                [0.95, 0.04, 0.01],  # From Normal
                [0.20, 0.65, 0.15],  # From Elevated
                [0.10, 0.30, 0.60],  # From Crisis
            ])
        else:
            self.transition_matrix = np.array(self.transition_matrix)
        
        self._validate_transition_matrix()
    
    def _validate_transition_matrix(self) -> None:
        """Validate the transition matrix."""
        if self.transition_matrix.shape != (3, 3):
            raise InvalidParameterError(
                f"Transition matrix must be 3x3, got {self.transition_matrix.shape}"
            )
        
        # Check rows sum to 1 (with tolerance)
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise InvalidParameterError(
                f"Transition matrix rows must sum to 1, got row sums: {row_sums}"
            )
        
        # Check all probabilities are non-negative
        if np.any(self.transition_matrix < 0):
            raise InvalidParameterError("Transition matrix contains negative probabilities")
    
    def sample_transition(self) -> int:
        """
        Sample the next crisis state based on current state.
        
        Returns:
            New crisis level (0, 1, or 2)
        """
        probs = self.transition_matrix[self.current_state]
        next_state = np.random.choice(3, p=probs)
        
        if next_state != self.current_state:
            self.periods_in_state = 0
        else:
            self.periods_in_state += 1
        
        self.current_state = next_state
        return next_state
    
    def get_current_event(self) -> CrisisEvent:
        """Get the CrisisEvent for the current state."""
        return self.states[self.current_state]
    
    def get_demand_multiplier(self) -> float:
        """Get the demand multiplier for current crisis state."""
        return self.states[self.current_state].demand_multiplier
    
    def get_supply_disruption_prob(self) -> float:
        """Get the supply disruption probability for current crisis state."""
        return self.states[self.current_state].supply_disruption_prob
    
    def should_reject_order(self, base_rejection_prob: float = 0.0) -> bool:
        """
        Determine if a supplier order should be rejected based on crisis.
        
        Args:
            base_rejection_prob: Base probability of rejection (supplier-specific)
        
        Returns:
            True if order should be rejected, False otherwise
        """
        crisis_prob = self.get_supply_disruption_prob()
        
        # Combine base and crisis probabilities
        # P(reject) = 1 - (1 - base)(1 - crisis) = base + crisis - base*crisis
        combined_prob = base_rejection_prob + crisis_prob - base_rejection_prob * crisis_prob
        combined_prob = min(0.95, combined_prob)  # Cap at 95%
        
        return np.random.random() < combined_prob
    
    def reset(self, initial_state: int = 0) -> None:
        """Reset the crisis process to initial state."""
        if not (0 <= initial_state <= 2):
            raise InvalidParameterError(
                f"Initial state must be 0, 1, or 2, got {initial_state}"
            )
        self.current_state = initial_state
        self.periods_in_state = 0
    
    def copy(self) -> 'CrisisProcess':
        """Create a copy of this crisis process."""
        return CrisisProcess(
            states=self.states.copy(),
            transition_matrix=self.transition_matrix.copy(),
            current_state=self.current_state,
            periods_in_state=self.periods_in_state
        )
    
    def to_exogenous_state(self, time: float = 0.0) -> np.ndarray:
        """
        Convert crisis state to exogenous state array.
        
        Args:
            time: Current time period
        
        Returns:
            Array [time, crisis_level]
        """
        return np.array([time, float(self.current_state)])
    
    def update_from_exogenous_state(self, exogenous_state: np.ndarray) -> None:
        """
        Update crisis state from exogenous state array.
        
        Args:
            exogenous_state: Array containing [time, crisis_level]
        """
        if exogenous_state is not None and len(exogenous_state) >= 2:
            self.current_state = int(exogenous_state[1])


def create_crisis_process(
    crisis_probability: float = 0.05,
    recovery_rate: float = 0.2,
    severity: str = "moderate"
) -> CrisisProcess:
    """
    Factory function to create a CrisisProcess with common configurations.
    
    Args:
        crisis_probability: Probability of transitioning to crisis from normal
        recovery_rate: Probability of improving one level
        severity: One of "mild", "moderate", "severe" - affects crisis impact
    
    Returns:
        Configured CrisisProcess
    """
    if not (0 <= crisis_probability <= 1):
        raise InvalidParameterError(
            f"Crisis probability must be in [0, 1], got {crisis_probability}"
        )
    if not (0 <= recovery_rate <= 1):
        raise InvalidParameterError(
            f"Recovery rate must be in [0, 1], got {recovery_rate}"
        )
    
    # Build transition matrix
    # From Normal: stay normal = 1 - 2*crisis_prob, go elevated = crisis_prob, go crisis = crisis_prob
    p_normal_to_elevated = min(crisis_probability * 0.8, 0.15)
    p_normal_to_crisis = min(crisis_probability * 0.2, 0.05)
    p_normal_to_normal = 1 - p_normal_to_elevated - p_normal_to_crisis
    
    # From Elevated: recover, stay, or worsen
    p_elevated_to_normal = recovery_rate * 0.8
    p_elevated_to_crisis = crisis_probability
    p_elevated_to_elevated = 1 - p_elevated_to_normal - p_elevated_to_crisis
    
    # From Crisis: mostly recover or stay
    p_crisis_to_normal = recovery_rate * 0.3
    p_crisis_to_elevated = recovery_rate * 0.5
    p_crisis_to_crisis = 1 - p_crisis_to_normal - p_crisis_to_elevated
    
    transition_matrix = np.array([
        [p_normal_to_normal, p_normal_to_elevated, p_normal_to_crisis],
        [p_elevated_to_normal, p_elevated_to_elevated, p_elevated_to_crisis],
        [p_crisis_to_normal, p_crisis_to_elevated, p_crisis_to_crisis],
    ])
    
    # Define crisis states based on severity
    if severity == "mild":
        states = [
            CrisisEvent(0, "Normal", 1.0, 0.0),
            CrisisEvent(1, "Elevated", 1.2, 0.05),
            CrisisEvent(2, "Crisis", 1.5, 0.15),
        ]
    elif severity == "moderate":
        states = [
            CrisisEvent(0, "Normal", 1.0, 0.0),
            CrisisEvent(1, "Elevated", 1.5, 0.15),
            CrisisEvent(2, "Crisis", 2.5, 0.35),
        ]
    elif severity == "severe":
        states = [
            CrisisEvent(0, "Normal", 1.0, 0.0),
            CrisisEvent(1, "Elevated", 2.0, 0.25),
            CrisisEvent(2, "Crisis", 4.0, 0.60),
        ]
    else:
        raise ValueError(f"Unknown severity: {severity}")
    
    return CrisisProcess(states=states, transition_matrix=transition_matrix)
