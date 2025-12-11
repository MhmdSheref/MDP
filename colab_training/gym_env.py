"""
Enhanced Gymnasium Wrapper for Perishable Inventory MDP.

Addresses RL training issues identified in analysis:
1. Adds cost information to observation space
2. Uses asymmetric action space favoring slow/cheap supplier
3. Shapes rewards with decomposed components

Based on recommendations from rl_performance_analysis.md.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from perishable_inventory_mdp.environment import PerishableInventoryMDP, EnhancedPerishableInventoryMDP
from perishable_inventory_mdp.state import InventoryState
from perishable_inventory_mdp.contracts import ContractManager


@dataclass
class RewardConfig:
    """Configuration for reward shaping components.
    
    shaped_reward = (
        -alpha * procurement_cost
        -beta * (holding + spoilage)
        -gamma * shortage_penalty
        +delta * service_bonus
    )
    """
    alpha: float = 0.5       # Procurement cost weight
    beta: float = 0.3        # Holding + spoilage weight
    gamma: float = 0.2       # Shortage penalty weight
    delta: float = 0.1       # Service bonus per unit sold at target
    target_fill_rate: float = 0.95  # Target fill rate for service bonus
    normalize: bool = True   # Whether to normalize reward
    normalization_scale: float = 10.0  # Scale factor for normalization
    
    def __post_init__(self):
        # Validate weights sum to approximately 1.0
        total = abs(self.alpha) + abs(self.beta) + abs(self.gamma)
        if total < 0.01:
            raise ValueError("At least one cost weight must be non-zero")


class PerishableInventoryGymWrapper(gym.Env):
    """
    Enhanced Gymnasium wrapper for PerishableInventoryMDP.
    
    Key improvements:
    1. Cost-aware observation space with supplier costs and lead times
    2. Asymmetric action space favoring slow/cheap supplier
    3. Decomposed reward shaping with configurable weights
    
    Observation Space Components:
        - Inventory buckets (N): normalized by max_inventory
        - Pipeline quantities (sum of lead times): normalized by total capacity
        - Backorders (1): normalized by mean_demand
        - Supplier costs (S): normalized relative to max cost
        - Lead times (S): normalized relative to max lead time
        - Crisis state (3): one-hot encoded [normal, elevated, crisis]
        - Contract discounts (S): discount rate if active, else 0
        - Time features (2): sin/cos encoded for seasonality
        - Demand history (H): last H periods of demand, normalized
        
    Action Space:
        MultiDiscrete with asymmetric bins per supplier
        - Slow supplier (ID=1): 7 bins -> [0, 10, 20, 30, 40, 50, 60]
        - Fast supplier (ID=0): 5 bins -> [0, 5, 10, 15, 20]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        mdp: PerishableInventoryMDP,
        reward_config: Optional[RewardConfig] = None,
        max_inventory: float = 200.0,
        demand_history_length: int = 5,
        seasonality_period: int = 50,
        contract_manager: Optional[ContractManager] = None,
        action_bins_slow: Optional[List[float]] = None,
        action_bins_fast: Optional[List[float]] = None
    ):
        """
        Initialize enhanced gym wrapper.
        
        Args:
            mdp: The underlying MDP environment
            reward_config: Configuration for reward shaping
            max_inventory: Maximum expected inventory for normalization
            demand_history_length: Number of past demands to include in observation
            seasonality_period: Period for time feature sin/cos encoding
            contract_manager: Optional contract manager for contract features
            action_bins_slow: Custom action bins for slow supplier
            action_bins_fast: Custom action bins for fast supplier
        """
        super().__init__()
        self.mdp = mdp
        self.reward_config = reward_config or RewardConfig()
        self.max_inventory = max_inventory
        self.demand_history_length = demand_history_length
        self.seasonality_period = seasonality_period
        self.contract_manager = contract_manager
        
        # --- Supplier analysis ---
        self.num_suppliers = len(mdp.suppliers)
        self.supplier_info = self._analyze_suppliers()
        
        # --- Action space (asymmetric) ---
        # Default bins if not provided
        self.action_bins_slow = action_bins_slow or [0, 10, 20, 30, 40, 50, 60]
        self.action_bins_fast = action_bins_fast or [0, 5, 10, 15, 20]
        
        # Map supplier IDs to action bins based on cost
        self._setup_action_space()
        
        # --- Observation space ---
        self._setup_observation_space()
        
        # --- Internal state tracking ---
        self.current_state: Optional[InventoryState] = None
        self.current_time: int = 0
        self.demand_history: List[float] = []
        
    def _analyze_suppliers(self) -> Dict[int, Dict[str, Any]]:
        """Analyze suppliers to determine costs and lead times."""
        info = {}
        max_cost = 0.0
        max_lead_time = 0
        
        for supplier in self.mdp.suppliers:
            sid = supplier['id']
            unit_cost = supplier.get('unit_cost', 1.0)
            lead_time = supplier.get('lead_time', 1)
            capacity = supplier.get('capacity', 100)
            
            info[sid] = {
                'unit_cost': unit_cost,
                'lead_time': lead_time,
                'capacity': capacity
            }
            
            max_cost = max(max_cost, unit_cost)
            max_lead_time = max(max_lead_time, lead_time)
        
        # Store normalization factors
        self.max_supplier_cost = max_cost if max_cost > 0 else 1.0
        self.max_lead_time = max_lead_time if max_lead_time > 0 else 1
        
        return info
    
    def _setup_action_space(self):
        """Setup asymmetric MultiDiscrete action space."""
        # Identify slow (cheap) and fast (expensive) suppliers by cost
        supplier_costs = [(sid, info['unit_cost']) for sid, info in self.supplier_info.items()]
        supplier_costs.sort(key=lambda x: x[1])  # Sort by cost ascending
        
        self.supplier_action_bins = {}
        action_dims = []
        self.supplier_order = []  # Order of suppliers in action array
        
        for i, (sid, cost) in enumerate(supplier_costs):
            if i == 0 and len(supplier_costs) > 1:
                # Cheapest supplier gets more action options (slow supplier)
                bins = self.action_bins_slow
            else:
                # More expensive suppliers get fewer options
                bins = self.action_bins_fast
            
            # Clip bins to capacity
            capacity = self.supplier_info[sid]['capacity']
            bins = [b for b in bins if b <= capacity]
            if 0 not in bins:
                bins = [0] + bins
            
            self.supplier_action_bins[sid] = bins
            action_dims.append(len(bins))
            self.supplier_order.append(sid)
        
        self.action_space = spaces.MultiDiscrete(action_dims)
        
    def _setup_observation_space(self):
        """Setup enhanced observation space."""
        self.shelf_life = self.mdp.shelf_life
        
        # Calculate observation components sizes
        self.obs_components = {}
        
        # 1. Inventory buckets
        self.obs_components['inventory'] = self.shelf_life
        
        # 2. Pipeline quantities (sum of lead times across suppliers)
        self.pipeline_size = sum(s['lead_time'] for s in self.mdp.suppliers)
        self.obs_components['pipeline'] = self.pipeline_size
        
        # 3. Backorders
        self.obs_components['backorders'] = 1
        
        # 4. Supplier costs (normalized)
        self.obs_components['supplier_costs'] = self.num_suppliers
        
        # 5. Lead times (normalized)
        self.obs_components['lead_times'] = self.num_suppliers
        
        # 6. Crisis state (one-hot: normal, elevated, crisis)
        self.obs_components['crisis'] = 3
        
        # 7. Contract discounts (per supplier)
        self.obs_components['contracts'] = self.num_suppliers
        
        # 8. Time features (sin, cos)
        self.obs_components['time'] = 2
        
        # 9. Demand history
        self.obs_components['demand_history'] = self.demand_history_length
        
        # Total observation size
        total_obs_size = sum(self.obs_components.values())
        
        self.observation_space = spaces.Box(
            low=-1.0,  # Allow negative for some normalized values
            high=2.0,  # Allow slightly above 1 for edge cases
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_state = self.mdp.reset(seed=seed, options=options)
        self.current_time = 0
        self.demand_history = [0.0] * self.demand_history_length
        
        # Reset crisis process if available
        if hasattr(self.mdp, 'crisis_process') and self.mdp.crisis_process is not None:
            self.mdp.crisis_process.reset()
        
        return self._get_observation(self.current_state), {"time_step": 0}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with enhanced reward shaping."""
        if self.current_state is None:
            raise RuntimeError("Call reset() before step()")
        
        # Convert discrete action to order quantities
        mdp_action = self._decode_action(action)
        
        # Execute MDP step
        result = self.mdp.step(self.current_state, mdp_action)
        
        # Update internal state
        self.current_state = result.next_state
        self.current_time += 1
        
        # Update demand history
        self.demand_history.pop(0)
        self.demand_history.append(result.demand_realized)
        
        # Get shaped reward
        reward = self._compute_shaped_reward(result, mdp_action)
        
        # Get observation
        obs = self._get_observation(self.current_state)
        
        # Episode never terminates (infinite horizon, use TimeLimit wrapper)
        terminated = False
        truncated = False
        
        # Build info dict with decomposed metrics
        info = {
            "demand": result.demand_realized,
            "sales": result.sales,
            "spoilage": result.spoiled,
            "total_cost": result.costs.total_cost,
            "raw_reward": result.reward,
            "procurement_cost": result.costs.purchase_cost,
            "holding_cost": result.costs.holding_cost,
            "shortage_cost": result.costs.shortage_cost,
            "spoilage_cost": result.costs.spoilage_cost,
            "fill_rate": result.sales / max(result.demand_realized, 1e-6),
            "orders": mdp_action,
            "time_step": self.current_time
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, state: InventoryState) -> np.ndarray:
        """Build enhanced observation vector."""
        obs_parts = []
        
        # 1. Inventory buckets (normalized)
        inv_normalized = state.inventory / max(self.max_inventory, 1.0)
        obs_parts.append(inv_normalized)
        
        # 2. Pipeline quantities (normalized by capacity)
        sorted_suppliers = sorted(state.pipelines.items())
        for sid, pipeline in sorted_suppliers:
            capacity = self.supplier_info.get(sid, {}).get('capacity', 100)
            pipeline_normalized = pipeline.pipeline / max(capacity, 1.0)
            obs_parts.append(pipeline_normalized)
        
        # 3. Backorders (normalized by expected demand)
        mean_demand = self._get_mean_demand()
        backorders_normalized = np.array([state.backorders / max(mean_demand * 2, 1.0)])
        obs_parts.append(backorders_normalized)
        
        # 4. Supplier costs (normalized)
        costs_normalized = []
        for sid in sorted(self.supplier_info.keys()):
            cost = self.supplier_info[sid]['unit_cost']
            costs_normalized.append(cost / self.max_supplier_cost)
        obs_parts.append(np.array(costs_normalized))
        
        # 5. Lead times (normalized)
        lead_times_normalized = []
        for sid in sorted(self.supplier_info.keys()):
            lt = self.supplier_info[sid]['lead_time']
            lead_times_normalized.append(lt / self.max_lead_time)
        obs_parts.append(np.array(lead_times_normalized))
        
        # 6. Crisis state (one-hot)
        crisis_one_hot = self._get_crisis_one_hot(state)
        obs_parts.append(crisis_one_hot)
        
        # 7. Contract discounts
        contract_discounts = self._get_contract_discounts()
        obs_parts.append(contract_discounts)
        
        # 8. Time features (sin/cos for seasonality)
        time_features = self._get_time_features()
        obs_parts.append(time_features)
        
        # 9. Demand history (normalized)
        demand_hist_normalized = np.array(self.demand_history) / max(mean_demand * 2, 1.0)
        obs_parts.append(demand_hist_normalized)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _get_crisis_one_hot(self, state: InventoryState) -> np.ndarray:
        """Extract crisis state as one-hot encoding."""
        crisis_level = 0  # Default: normal
        
        # Try to get crisis level from exogenous state
        if state.exogenous_state is not None and len(state.exogenous_state) >= 2:
            crisis_level = int(state.exogenous_state[1])
        
        # Also check if MDP has crisis_process
        if hasattr(self.mdp, 'crisis_process') and self.mdp.crisis_process is not None:
            crisis_level = self.mdp.crisis_process.current_state
        
        # One-hot encode (3 levels: 0=normal, 1=elevated, 2=crisis)
        one_hot = np.zeros(3)
        one_hot[min(crisis_level, 2)] = 1.0
        return one_hot
    
    def _get_contract_discounts(self) -> np.ndarray:
        """Get current contract discount rates per supplier."""
        discounts = np.zeros(self.num_suppliers)
        
        if self.contract_manager is not None:
            for i, sid in enumerate(sorted(self.supplier_info.keys())):
                contract = self.contract_manager.get_contract_for_order(sid)
                if contract is not None and contract.is_active():
                    discounts[i] = contract.discount_rate
        
        return discounts
    
    def _get_time_features(self) -> np.ndarray:
        """Get sin/cos time features for seasonality."""
        phase = 2 * np.pi * self.current_time / self.seasonality_period
        return np.array([np.sin(phase), np.cos(phase)])
    
    def _get_mean_demand(self) -> float:
        """Get mean demand from the demand process.
        
        Handles different demand process types:
        - PoissonDemand: uses base_rate attribute
        - CompositeDemand: uses mean_demand attribute
        - Others: tries mean() method, falls back to 10.0
        """
        dp = self.mdp.demand_process
        
        # PoissonDemand has base_rate
        if hasattr(dp, 'base_rate'):
            return float(dp.base_rate)
        
        # CompositeDemand and others may have mean_demand
        if hasattr(dp, 'mean_demand'):
            return float(dp.mean_demand)
        
        # Try calling mean() method with no args
        if hasattr(dp, 'mean') and callable(dp.mean):
            try:
                return float(dp.mean(None))
            except (TypeError, ValueError):
                pass
        
        # Fallback
        return 10.0
    
    def _decode_action(self, action: np.ndarray) -> Dict[int, float]:
        """Convert MultiDiscrete action to order quantities."""
        mdp_action = {}
        
        for i, sid in enumerate(self.supplier_order):
            action_idx = int(action[i])
            bins = self.supplier_action_bins[sid]
            
            # Clamp to valid index
            action_idx = min(action_idx, len(bins) - 1)
            action_idx = max(action_idx, 0)
            
            qty = bins[action_idx]
            
            # Round to MOQ if needed
            pipeline = self.current_state.pipelines.get(sid)
            if pipeline is not None:
                moq = pipeline.moq
                if moq > 0 and qty > 0:
                    qty = round(qty / moq) * moq
            
            mdp_action[sid] = float(qty)
        
        return mdp_action
    
    def _compute_shaped_reward(self, result, action: Dict[int, float]) -> float:
        """Compute shaped reward with decomposed components."""
        cfg = self.reward_config
        costs = result.costs
        
        # Get mean demand for normalization
        mean_demand = self._get_mean_demand()
        
        # Component 1: Procurement cost
        procurement = costs.purchase_cost + costs.fixed_order_cost
        
        # Component 2: Holding + spoilage
        inventory_cost = costs.holding_cost + costs.spoilage_cost
        
        # Component 3: Shortage penalty
        shortage = costs.shortage_cost
        
        # Component 4: Service bonus (reward for meeting demand efficiently)
        fill_rate = result.sales / max(result.demand_realized, 1e-6)
        if fill_rate >= cfg.target_fill_rate:
            service_bonus = cfg.delta * result.sales
        else:
            service_bonus = 0.0
        
        # Combine with weights
        shaped_reward = (
            -cfg.alpha * procurement
            - cfg.beta * inventory_cost
            - cfg.gamma * shortage
            + service_bonus
        )
        
        # Normalize if configured
        if cfg.normalize:
            # Normalize by expected cost scale
            expected_scale = mean_demand * self.max_supplier_cost * cfg.normalization_scale
            if expected_scale > 0:
                shaped_reward = shaped_reward / expected_scale
        
        return float(shaped_reward)
    
    def get_supplier_action_space_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about the action space for each supplier."""
        info = {}
        for sid in self.supplier_order:
            bins = self.supplier_action_bins[sid]
            info[sid] = {
                'bins': bins,
                'num_actions': len(bins),
                'unit_cost': self.supplier_info[sid]['unit_cost'],
                'lead_time': self.supplier_info[sid]['lead_time']
            }
        return info
    
    def get_observation_space_info(self) -> Dict[str, Tuple[int, int]]:
        """Get start/end indices for each observation component."""
        info = {}
        idx = 0
        for name, size in self.obs_components.items():
            info[name] = (idx, idx + size)
            idx += size
        return info


def create_gym_env(
    shelf_life: int = 5,
    mean_demand: float = 10.0,
    fast_lead_time: int = 1,
    slow_lead_time: int = 3,
    fast_cost: float = 2.0,
    slow_cost: float = 1.0,
    enable_crisis: bool = False,
    reward_config: Optional[RewardConfig] = None,
    **mdp_kwargs
) -> PerishableInventoryGymWrapper:
    """
    Factory function to create enhanced gym environment.
    
    Args:
        shelf_life: Number of expiry buckets
        mean_demand: Mean demand per period
        fast_lead_time: Lead time for fast (expensive) supplier
        slow_lead_time: Lead time for slow (cheap) supplier
        fast_cost: Unit cost for fast supplier
        slow_cost: Unit cost for slow supplier
        enable_crisis: Whether to enable crisis dynamics
        reward_config: Custom reward configuration
        **mdp_kwargs: Additional arguments for MDP creation
    
    Returns:
        Configured PerishableInventoryGymWrapperV2
    """
    from perishable_inventory_mdp.environment import create_simple_mdp, create_enhanced_mdp
    
    if enable_crisis:
        mdp = create_enhanced_mdp(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            fast_lead_time=fast_lead_time,
            slow_lead_time=slow_lead_time,
            fast_cost=fast_cost,
            slow_cost=slow_cost,
            enable_crisis=True,
            **mdp_kwargs
        )
    else:
        mdp = create_simple_mdp(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            fast_lead_time=fast_lead_time,
            slow_lead_time=slow_lead_time,
            fast_cost=fast_cost,
            slow_cost=slow_cost
        )
    
    return PerishableInventoryGymWrapper(
        mdp=mdp,
        reward_config=reward_config
    )
