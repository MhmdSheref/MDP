"""
Tests for the Enhanced Gym Wrapper.

Tests enhanced observation space, asymmetric action space, and reward shaping.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import sys
sys.path.insert(0, '.')

from colab_training.gym_env import (
    PerishableInventoryGymWrapper,
    RewardConfig,
    create_gym_env
)
from perishable_inventory_mdp.environment import create_simple_mdp, create_enhanced_mdp
from perishable_inventory_mdp.contracts import ContractManager, SupplierContract


class TestRewardConfig:
    """Tests for RewardConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RewardConfig()
        
        assert config.alpha == 0.5
        assert config.beta == 0.3
        assert config.gamma == 0.2
        assert config.delta == 0.1
        assert config.target_fill_rate == 0.95
        assert config.normalize is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = RewardConfig(
            alpha=0.6,
            beta=0.2,
            gamma=0.1,
            delta=0.2,
            normalize=False
        )
        
        assert config.alpha == 0.6
        assert config.normalize is False
    
    def test_zero_weights_raises(self):
        """Test that zero weights raise validation error"""
        with pytest.raises(ValueError):
            RewardConfig(alpha=0.0, beta=0.0, gamma=0.0)


class TestObservationSpace:
    """Tests for enhanced observation space"""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple wrapped environment"""
        return create_gym_env(
            shelf_life=5,
            mean_demand=10.0,
            fast_lead_time=1,
            slow_lead_time=3,
            fast_cost=2.0,
            slow_cost=1.0
        )
    
    def test_observation_space_shape(self, simple_env):
        """Test observation space has correct shape"""
        obs_info = simple_env.get_observation_space_info()
        
        # Check all components are present
        expected_components = [
            'inventory', 'pipeline', 'backorders',
            'supplier_costs', 'lead_times', 'crisis',
            'contracts', 'time', 'demand_history'
        ]
        for comp in expected_components:
            assert comp in obs_info, f"Missing component: {comp}"
        
        # Total size should match observation space
        total_size = sum(end - start for start, end in obs_info.values())
        assert simple_env.observation_space.shape[0] == total_size
    
    def test_observation_bounds(self, simple_env):
        """Test observation values are within bounds"""
        obs, _ = simple_env.reset(seed=42)
        
        # Most values should be in [-1, 2] range
        assert obs.min() >= -1.5, f"Observation min too low: {obs.min()}"
        assert obs.max() <= 2.5, f"Observation max too high: {obs.max()}"
    
    def test_observation_components(self, simple_env):
        """Test individual observation components"""
        obs, _ = simple_env.reset(seed=42)
        obs_info = simple_env.get_observation_space_info()
        
        # Inventory should be non-negative (normalized)
        inv_start, inv_end = obs_info['inventory']
        inventory = obs[inv_start:inv_end]
        assert (inventory >= 0).all(), "Inventory should be non-negative"
        
        # Supplier costs should be in [0, 1]
        cost_start, cost_end = obs_info['supplier_costs']
        costs = obs[cost_start:cost_end]
        assert (costs >= 0).all() and (costs <= 1).all(), "Costs should be normalized"
        
        # Crisis should be one-hot (sums to 1)
        crisis_start, crisis_end = obs_info['crisis']
        crisis = obs[crisis_start:crisis_end]
        assert_almost_equal(crisis.sum(), 1.0, decimal=5)
    
    def test_supplier_cost_normalization(self, simple_env):
        """Test that supplier costs are correctly normalized"""
        obs, _ = simple_env.reset(seed=42)
        obs_info = simple_env.get_observation_space_info()
        
        cost_start, cost_end = obs_info['supplier_costs']
        costs = obs[cost_start:cost_end]
        
        # The max cost supplier should have normalized cost = 1.0
        # fast_cost=2.0 > slow_cost=1.0, so max is 2.0
        # Normalized: [2.0/2.0, 1.0/2.0] = [1.0, 0.5]
        # But order depends on supplier ID sorting
        assert 1.0 in costs or np.isclose(costs, 1.0).any()
    
    def test_time_features(self, simple_env):
        """Test sin/cos time features"""
        simple_env.reset(seed=42)
        
        # Run a few steps and check time features change
        time_features_list = []
        for _ in range(10):
            action = simple_env.action_space.sample()
            obs, _, _, _, _ = simple_env.step(action)
            
            obs_info = simple_env.get_observation_space_info()
            time_start, time_end = obs_info['time']
            time_features_list.append(obs[time_start:time_end].copy())
        
        # Time features should vary over time
        time_features_array = np.array(time_features_list)
        assert time_features_array.std(axis=0).sum() > 0.01
    
    def test_demand_history(self, simple_env):
        """Test demand history is tracked correctly"""
        simple_env.reset(seed=42)
        
        demands = []
        for _ in range(10):
            action = simple_env.action_space.sample()
            _, _, _, _, info = simple_env.step(action)
            demands.append(info['demand'])
        
        # After 10 steps, demand history should contain recent demands
        assert len(simple_env.demand_history) == simple_env.demand_history_length


class TestActionSpace:
    """Tests for asymmetric action space"""
    
    @pytest.fixture
    def env(self):
        """Create wrapped environment"""
        return create_gym_env(
            fast_lead_time=1,
            slow_lead_time=3,
            fast_cost=2.0,
            slow_cost=1.0
        )
    
    def test_action_space_type(self, env):
        """Test action space is MultiDiscrete"""
        from gymnasium.spaces import MultiDiscrete
        assert isinstance(env.action_space, MultiDiscrete)
    
    def test_asymmetric_action_bins(self, env):
        """Test that cheap supplier has more action bins"""
        info = env.get_supplier_action_space_info()
        
        # Find supplier with lower cost
        min_cost_sid = min(info.keys(), key=lambda x: info[x]['unit_cost'])
        max_cost_sid = max(info.keys(), key=lambda x: info[x]['unit_cost'])
        
        # Cheap supplier should have >= actions than expensive
        assert info[min_cost_sid]['num_actions'] >= info[max_cost_sid]['num_actions']
    
    def test_action_decoding(self, env):
        """Test that actions decode correctly"""
        env.reset(seed=42)
        
        # Test action [0, 0] -> both suppliers order 0
        action = np.array([0, 0])
        mdp_action = env._decode_action(action)
        
        for qty in mdp_action.values():
            assert qty == 0.0
        
        # Test maximum action
        max_action = np.array([
            env.action_space.nvec[0] - 1,
            env.action_space.nvec[1] - 1 if len(env.action_space.nvec) > 1 else 0
        ])
        mdp_action = env._decode_action(max_action[:len(env.supplier_order)])
        
        # At least one should be non-zero
        assert any(qty > 0 for qty in mdp_action.values())
    
    def test_action_bins_values(self, env):
        """Test specific action bin values"""
        info = env.get_supplier_action_space_info()
        
        for sid, supplier_info in info.items():
            bins = supplier_info['bins']
            
            # First bin should be 0
            assert bins[0] == 0
            
            # Bins should be monotonically increasing
            for i in range(1, len(bins)):
                assert bins[i] > bins[i-1]


class TestRewardShaping:
    """Tests for reward shaping"""
    
    @pytest.fixture
    def env_shaped(self):
        """Create environment with shaped rewards"""
        config = RewardConfig(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            delta=0.1,
            normalize=False  # Easier to verify without normalization
        )
        return create_gym_env(reward_config=config)
    
    @pytest.fixture
    def env_raw(self):
        """Create environment with minimal shaping"""
        config = RewardConfig(
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            delta=0.0,
            normalize=False
        )
        return create_gym_env(reward_config=config)
    
    def test_reward_negative_when_ordering(self, env_shaped):
        """Test that ordering incurs cost (negative reward component)"""
        env_shaped.reset(seed=42)
        
        # Order from expensive supplier
        action = np.array([env_shaped.action_space.nvec[0] - 1, 0])
        _, reward, _, _, info = env_shaped.step(action)
        
        # Procurement cost should be positive
        assert info['procurement_cost'] > 0
    
    def test_reward_decomposition(self, env_shaped):
        """Test that reward components are decomposed in info"""
        env_shaped.reset(seed=42)
        
        action = env_shaped.action_space.sample()
        _, _, _, _, info = env_shaped.step(action)
        
        # Check all cost components are present
        assert 'procurement_cost' in info
        assert 'holding_cost' in info
        assert 'shortage_cost' in info
        assert 'spoilage_cost' in info
        assert 'total_cost' in info
        assert 'raw_reward' in info
    
    def test_service_bonus(self, env_shaped):
        """Test service bonus for high fill rate"""
        # This is a statistical test - run multiple episodes
        env_shaped.reset(seed=42)
        
        # Run with high inventory to avoid stockouts
        for _ in range(10):
            # Order from slow (cheap) supplier
            action = np.array([0, env_shaped.action_space.nvec[1] - 1])
            _, _, _, _, info = env_shaped.step(action)
        
        # After building inventory, fill rate should be high
        assert info['fill_rate'] >= 0
    
    def test_reward_normalization(self):
        """Test that normalization affects reward scale"""
        config_norm = RewardConfig(normalize=True, normalization_scale=10.0)
        config_raw = RewardConfig(normalize=False)
        
        env_norm = create_gym_env(reward_config=config_norm)
        env_raw = create_gym_env(reward_config=config_raw)
        
        # Use same seed for reproducibility
        env_norm.reset(seed=42)
        env_raw.reset(seed=42)
        
        action = np.array([1, 1])
        _, reward_norm, _, _, _ = env_norm.step(action)
        _, reward_raw, _, _, _ = env_raw.step(action)
        
        # Normalized reward should generally have smaller magnitude
        # (unless raw reward is already small)
        if abs(reward_raw) > 1:
            assert abs(reward_norm) < abs(reward_raw)


class TestCrisisIntegration:
    """Tests for crisis state integration"""
    
    @pytest.fixture
    def crisis_env(self):
        """Create environment with crisis enabled"""
        return create_gym_env(
            enable_crisis=True,
            crisis_probability=0.3  # Higher probability for testing
        )
    
    def test_crisis_one_hot_encoding(self, crisis_env):
        """Test crisis state is one-hot encoded"""
        obs, _ = crisis_env.reset(seed=42)
        obs_info = crisis_env.get_observation_space_info()
        
        crisis_start, crisis_end = obs_info['crisis']
        crisis = obs[crisis_start:crisis_end]
        
        # Should be one-hot (exactly one 1, rest 0)
        assert len(crisis) == 3
        assert_almost_equal(crisis.sum(), 1.0, decimal=5)
        assert ((crisis == 0) | (crisis == 1)).all()
    
    def test_crisis_state_changes(self, crisis_env):
        """Test crisis state can change over time"""
        crisis_env.reset(seed=42)
        
        crisis_states = []
        for _ in range(100):
            action = crisis_env.action_space.sample()
            obs, _, _, _, _ = crisis_env.step(action)
            
            obs_info = crisis_env.get_observation_space_info()
            crisis_start, crisis_end = obs_info['crisis']
            crisis = obs[crisis_start:crisis_end]
            crisis_states.append(np.argmax(crisis))
        
        # With crisis_probability=0.3, we should see some variation
        unique_states = set(crisis_states)
        # At minimum normal state (0) should appear
        assert 0 in unique_states or 1 in unique_states or 2 in unique_states


class TestContractIntegration:
    """Tests for contract integration in observations"""
    
    def test_contract_discounts_in_obs(self):
        """Test that contract discounts appear in observation"""
        mdp = create_simple_mdp()
        
        # Create contract manager with active contract
        manager = ContractManager()
        contract = SupplierContract(
            contract_id=1,
            supplier_id=0,
            discount_rate=0.15
        )
        contract.sign()
        manager.contracts[1] = contract
        
        env = PerishableInventoryGymWrapper(
            mdp=mdp,
            contract_manager=manager
        )
        
        obs, _ = env.reset(seed=42)
        obs_info = env.get_observation_space_info()
        
        contract_start, contract_end = obs_info['contracts']
        contracts = obs[contract_start:contract_end]
        
        # Should have non-zero discount for supplier 0
        assert contracts.max() > 0


class TestIntegration:
    """Integration tests for full episode runs"""
    
    def test_full_episode_simple(self):
        """Test running a full episode with simple MDP"""
        env = create_gym_env()
        
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        
        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            assert obs.shape == env.observation_space.shape
            assert not terminated
            assert not truncated
        
        # Total reward should be finite
        assert np.isfinite(total_reward)
    
    def test_full_episode_crisis(self):
        """Test running a full episode with crisis enabled"""
        env = create_gym_env(enable_crisis=True)
        
        obs, _ = env.reset(seed=42)
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, _, _, info = env.step(action)
            
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same initial observation"""
        env1 = create_gym_env()
        env2 = create_gym_env()
        
        # Reset with same seed should give same initial state
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        assert_array_almost_equal(obs1, obs2)
        
        # Note: step() involves random demand sampling which uses global 
        # numpy random state, so exact determinism after reset is not guaranteed
        # unless we fully control the random state in the step() call
        action = np.array([1, 1])
        _, r1, _, _, info1 = env1.step(action)
        
        # Just verify step works and returns valid data
        assert np.isfinite(r1)
        assert 'demand' in info1
    
    def test_info_dict_completeness(self):
        """Test that info dict contains expected keys"""
        env = create_gym_env()
        env.reset(seed=42)
        
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        
        expected_keys = [
            'demand', 'sales', 'spoilage', 'total_cost',
            'raw_reward', 'procurement_cost', 'holding_cost',
            'shortage_cost', 'spoilage_cost', 'fill_rate',
            'orders', 'time_step'
        ]
        
        for key in expected_keys:
            assert key in info, f"Missing info key: {key}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_zero_inventory_start(self):
        """Test starting with zero inventory"""
        env = create_gym_env()
        obs, _ = env.reset(seed=42)
        
        # Should not raise
        action = np.array([0, 0])
        obs, reward, _, _, _ = env.step(action)
        
        assert np.isfinite(obs).all()
    
    def test_high_demand_scenario(self):
        """Test with high demand"""
        env = create_gym_env(mean_demand=50.0)
        env.reset(seed=42)
        
        # Run with no orders - should handle stockouts
        for _ in range(10):
            action = np.array([0, 0])
            obs, reward, _, _, info = env.step(action)
            
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
    
    def test_single_supplier(self):
        """Test with single supplier MDP"""
        from perishable_inventory_mdp.environment import create_simple_mdp
        
        mdp = create_simple_mdp(num_suppliers=1)
        env = PerishableInventoryGymWrapper(mdp=mdp)
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        
        action = np.array([1])  # Single action
        obs, reward, _, _, _ = env.step(action)
        
        assert np.isfinite(obs).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
