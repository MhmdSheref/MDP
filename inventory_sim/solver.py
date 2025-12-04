"""
MDP Solvers for the Perishable Inventory Problem

Implements:
- Value Iteration
- Policy Iteration
- Approximate Dynamic Programming (ADP)

Based on the Bellman equation:
V(X) = max_{a∈A(X)} { -c(X,a) + γ * E[V(X') | X, a] }
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from .core.state import InventoryState
from .env import PerishableInventoryMDP
from .interfaces import InventoryAgent


@dataclass
class SolverResult:
    """
    Result from running an MDP solver.
    
    Attributes:
        value_function: Dictionary mapping states to values V(X)
        policy: Dictionary mapping states to actions π(X)
        iterations: Number of iterations until convergence
        converged: Whether the solver converged
        history: Optional list of value function norms per iteration
    """
    value_function: Dict[tuple, float] = field(default_factory=dict)
    policy: Dict[tuple, Dict[int, float]] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    history: List[float] = field(default_factory=list)


class ValueIteration:
    """
    Value Iteration algorithm for solving the MDP.
    
    Iteratively applies the Bellman operator:
    (TV)(X) = max_a { -c(X,a) + γ * E[V(X') | X, a] }
    
    until convergence.
    
    For large/continuous state spaces, uses discretization
    and Monte Carlo estimation.
    """
    
    def __init__(
        self,
        mdp: PerishableInventoryMDP,
        num_demand_samples: int = 50,
        max_iterations: int = 1000,
        tolerance: float = 1e-4
    ):
        self.mdp = mdp
        self.num_demand_samples = num_demand_samples
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def bellman_operator(
        self,
        state: InventoryState,
        action: Dict[int, float],
        value_function: Dict[tuple, float]
    ) -> float:
        """
        Apply Bellman operator for a single state-action pair.
        
        Q(X, a) = -c(X,a) + γ * E[V(X') | X, a]
        
        Uses Monte Carlo to estimate expectation over demand.
        """
        gamma = self.mdp.cost_params.discount_factor
        total_value = 0.0
        
        for _ in range(self.num_demand_samples):
            result = self.mdp.step(state.copy(), action)
            next_state_key = result.next_state.to_tuple()
            
            # Get value of next state (default to 0 if not seen)
            next_value = value_function.get(next_state_key, 0.0)
            
            # Q = -cost + γ * V(X')
            q_value = result.reward + gamma * next_value
            total_value += q_value
        
        return total_value / self.num_demand_samples
    
    def solve(
        self,
        initial_states: List[InventoryState],
        action_space: Optional[List[Dict[int, float]]] = None
    ) -> SolverResult:
        """
        Run value iteration starting from given states.
        
        Args:
            initial_states: List of states to evaluate
            action_space: Optional discrete action space to consider
        
        Returns:
            SolverResult with value function and policy
        """
        # Initialize value function
        value_function: Dict[tuple, float] = defaultdict(float)
        policy: Dict[tuple, Dict[int, float]] = {}
        
        history = []
        
        for iteration in range(self.max_iterations):
            max_change = 0.0
            new_values = {}
            
            for state in initial_states:
                state_key = state.to_tuple()
                
                # Get feasible actions
                if action_space is not None:
                    actions = [a for a in action_space if self.mdp.is_action_feasible(state, a)]
                else:
                    actions = self.mdp.get_feasible_actions(state)
                
                # Find best action
                best_value = float('-inf')
                best_action = actions[0] if actions else {}
                
                for action in actions:
                    q_value = self.bellman_operator(state, action, value_function)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                new_values[state_key] = best_value
                policy[state_key] = best_action
                
                # Track convergence
                old_value = value_function.get(state_key, 0.0)
                max_change = max(max_change, abs(best_value - old_value))
            
            # Update value function
            for key, val in new_values.items():
                value_function[key] = val
            
            history.append(max_change)
            
            # Check convergence
            if max_change < self.tolerance:
                return SolverResult(
                    value_function=dict(value_function),
                    policy=policy,
                    iterations=iteration + 1,
                    converged=True,
                    history=history
                )
        
        return SolverResult(
            value_function=dict(value_function),
            policy=policy,
            iterations=self.max_iterations,
            converged=False,
            history=history
        )


class PolicyIteration:
    """
    Policy Iteration algorithm for solving the MDP.
    
    Alternates between:
    1. Policy Evaluation: Compute V^π for current policy
    2. Policy Improvement: Update policy greedily w.r.t. V^π
    
    Typically converges faster than value iteration.
    """
    
    def __init__(
        self,
        mdp: PerishableInventoryMDP,
        num_demand_samples: int = 50,
        max_iterations: int = 100,
        eval_iterations: int = 50,
        tolerance: float = 1e-4
    ):
        self.mdp = mdp
        self.num_demand_samples = num_demand_samples
        self.max_iterations = max_iterations
        self.eval_iterations = eval_iterations
        self.tolerance = tolerance
    
    def policy_evaluation(
        self,
        policy: Dict[tuple, Dict[int, float]],
        states: List[InventoryState],
        value_function: Dict[tuple, float]
    ) -> Dict[tuple, float]:
        """
        Evaluate a policy by computing V^π.
        
        V^π(X) = -c(X, π(X)) + γ * E[V^π(X') | X, π(X)]
        """
        gamma = self.mdp.cost_params.discount_factor
        new_values = value_function.copy()
        
        for _ in range(self.eval_iterations):
            for state in states:
                state_key = state.to_tuple()
                action = policy.get(state_key, {})
                
                # Estimate expected value
                total_value = 0.0
                for _ in range(self.num_demand_samples):
                    result = self.mdp.step(state.copy(), action)
                    next_key = result.next_state.to_tuple()
                    next_value = new_values.get(next_key, 0.0)
                    total_value += result.reward + gamma * next_value
                
                new_values[state_key] = total_value / self.num_demand_samples
        
        return new_values
    
    def policy_improvement(
        self,
        states: List[InventoryState],
        value_function: Dict[tuple, float],
        action_space: Optional[List[Dict[int, float]]] = None
    ) -> Tuple[Dict[tuple, Dict[int, float]], bool]:
        """
        Improve policy greedily with respect to value function.
        
        π'(X) = argmax_a { -c(X,a) + γ * E[V(X') | X, a] }
        
        Returns:
            Tuple of (new policy, whether policy changed)
        """
        gamma = self.mdp.cost_params.discount_factor
        new_policy = {}
        policy_changed = False
        
        for state in states:
            state_key = state.to_tuple()
            
            # Get feasible actions
            if action_space is not None:
                actions = [a for a in action_space if self.mdp.is_action_feasible(state, a)]
            else:
                actions = self.mdp.get_feasible_actions(state)
            
            # Find best action
            best_value = float('-inf')
            best_action = actions[0] if actions else {}
            
            for action in actions:
                # Estimate Q(X, a)
                total_value = 0.0
                for _ in range(self.num_demand_samples):
                    result = self.mdp.step(state.copy(), action)
                    next_key = result.next_state.to_tuple()
                    next_value = value_function.get(next_key, 0.0)
                    total_value += result.reward + gamma * next_value
                
                q_value = total_value / self.num_demand_samples
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            new_policy[state_key] = best_action
        
        return new_policy, policy_changed
    
    def solve(
        self,
        initial_states: List[InventoryState],
        initial_policy: Optional[InventoryAgent] = None,
        action_space: Optional[List[Dict[int, float]]] = None
    ) -> SolverResult:
        """
        Run policy iteration.
        
        Args:
            initial_states: States to evaluate
            initial_policy: Optional starting policy
            action_space: Optional discrete action space
        
        Returns:
            SolverResult with value function and policy
        """
        # Initialize policy
        policy: Dict[tuple, Dict[int, float]] = {}
        for state in initial_states:
            state_key = state.to_tuple()
            if initial_policy is not None:
                policy[state_key] = initial_policy.act(state, self.mdp)
            else:
                policy[state_key] = {s: 0.0 for s in state.pipelines.keys()}
        
        value_function: Dict[tuple, float] = defaultdict(float)
        history = []
        
        for iteration in range(self.max_iterations):
            # Policy Evaluation
            value_function = self.policy_evaluation(
                policy, initial_states, value_function
            )
            
            # Policy Improvement
            new_policy, _ = self.policy_improvement(
                initial_states, value_function, action_space
            )
            
            # Check convergence (policy unchanged)
            policy_diff = sum(
                1 for k in policy.keys()
                if policy.get(k) != new_policy.get(k)
            )
            
            history.append(policy_diff)
            
            if policy_diff == 0:
                return SolverResult(
                    value_function=dict(value_function),
                    policy=new_policy,
                    iterations=iteration + 1,
                    converged=True,
                    history=history
                )
            
            policy = new_policy
        
        return SolverResult(
            value_function=dict(value_function),
            policy=policy,
            iterations=self.max_iterations,
            converged=False,
            history=history
        )


class SolvedPolicy(InventoryAgent):
    """
    Policy wrapper that uses a solved policy mapping.
    
    Looks up the action from the policy dictionary,
    using nearest neighbor for unseen states.
    """
    
    def __init__(
        self,
        policy_dict: Dict[tuple, Dict[int, float]],
        default_action: Optional[Dict[int, float]] = None
    ):
        self.policy_dict = policy_dict
        self.default_action = default_action or {}
    
    def act(
        self,
        state: InventoryState,
        env: PerishableInventoryMDP
    ) -> Dict[int, float]:
        state_key = state.to_tuple()
        
        if state_key in self.policy_dict:
            return self.policy_dict[state_key].copy()
        
        # Default action for unseen states
        if self.default_action:
            return self.default_action.copy()
        
        return {s: 0.0 for s in state.pipelines.keys()}


class ApproximateDynamicProgramming:
    """
    Approximate Dynamic Programming for large state spaces.
    
    Uses function approximation and simulation-based learning.
    Implements a simple linear value function approximation.
    """
    
    def __init__(
        self,
        mdp: PerishableInventoryMDP,
        feature_fn: Callable[[InventoryState], np.ndarray],
        learning_rate: float = 0.01,
        num_iterations: int = 1000
    ):
        self.mdp = mdp
        self.feature_fn = feature_fn
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
    
    def fit(
        self,
        initial_states: List[InventoryState],
        policy: InventoryAgent
    ) -> np.ndarray:
        """
        Learn value function weights via TD(0) learning.
        
        V(X) ≈ θᵀφ(X)
        
        Update: θ ← θ + α * (r + γV(X') - V(X)) * φ(X)
        """
        # Initialize weights
        sample_features = self.feature_fn(initial_states[0])
        self.weights = np.zeros(len(sample_features))
        
        gamma = self.mdp.cost_params.discount_factor
        
        for _ in range(self.num_iterations):
            # Sample a state
            state = np.random.choice(initial_states).copy()
            
            for _ in range(100):  # Episode length
                features = self.feature_fn(state)
                current_value = np.dot(self.weights, features)
                
                action = policy.act(state, self.mdp)
                result = self.mdp.step(state, action)
                
                next_features = self.feature_fn(result.next_state)
                next_value = np.dot(self.weights, next_features)
                
                # TD error
                td_error = result.reward + gamma * next_value - current_value
                
                # Update weights
                self.weights += self.learning_rate * td_error * features
                
                state = result.next_state
        
        return self.weights
    
    def get_value(self, state: InventoryState) -> float:
        """Get approximate value of a state"""
        if self.weights is None:
            return 0.0
        features = self.feature_fn(state)
        return np.dot(self.weights, features)


def default_feature_fn(state: InventoryState) -> np.ndarray:
    """
    Default feature function for linear value approximation.
    
    Features:
    - Total inventory
    - Inventory by bucket (normalized)
    - Pipeline totals
    - Backorders
    - Inventory position
    """
    features = [
        1.0,  # Bias
        state.total_inventory / 100,
        state.inventory_position / 100,
        state.backorders / 10,
        state.total_pipeline / 100,
    ]
    
    # Inventory by bucket
    for i, inv in enumerate(state.inventory):
        features.append(inv / 50)
    
    return np.array(features)

