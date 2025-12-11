"""
Tests for the Environment Suite.

Validates environment generation, uniqueness, and configuration.
"""

import pytest
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, '.')

from colab_training.environment_suite import (
    EnvironmentConfig,
    EnvironmentSuite,
    ComplexityLevel,
    build_environment_from_config,
    generate_simple_environments,
    generate_moderate_environments,
    generate_complex_environments,
    generate_extreme_environments,
    create_environment_suite,
    get_canonical_suite,
    iter_environments
)
from perishable_inventory_mdp.environment import PerishableInventoryMDP, EnhancedPerishableInventoryMDP


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EnvironmentConfig()
        
        assert config.shelf_life == 5
        assert config.mean_demand == 10.0
        assert config.num_suppliers == 2
        assert config.demand_type == "stationary"
        assert config.complexity == "simple"
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = EnvironmentConfig(
            shelf_life=7,
            mean_demand=15.0,
            num_suppliers=3,
            lead_times=(1, 2, 4),
            unit_costs=(3.0, 2.0, 1.0),
            demand_type="seasonal",
            complexity="complex"
        )
        
        assert config.shelf_life == 7
        assert config.num_suppliers == 3
        assert len(config.lead_times) == 3
        assert config.complexity == "complex"
    
    def test_unique_id_generation(self):
        """Test that different configs get different IDs"""
        config1 = EnvironmentConfig(shelf_life=5)
        config2 = EnvironmentConfig(shelf_life=6)
        
        assert config1.env_id != config2.env_id
    
    def test_same_config_same_id(self):
        """Test that identical configs get same ID"""
        config1 = EnvironmentConfig(
            shelf_life=5,
            mean_demand=10.0,
            lead_times=(1, 3)
        )
        config2 = EnvironmentConfig(
            shelf_life=5,
            mean_demand=10.0,
            lead_times=(1, 3)
        )
        
        assert config1.env_id == config2.env_id
    
    def test_to_dict_and_back(self):
        """Test serialization round-trip"""
        config = EnvironmentConfig(
            shelf_life=6,
            mean_demand=12.0,
            num_suppliers=3,
            lead_times=(1, 2, 4),
            unit_costs=(3.0, 2.0, 1.0),
            complexity="moderate"
        )
        
        data = config.to_dict()
        restored = EnvironmentConfig.from_dict(data)
        
        assert restored.shelf_life == config.shelf_life
        assert restored.mean_demand == config.mean_demand
        assert restored.num_suppliers == config.num_suppliers
        assert restored.lead_times == config.lead_times
        assert restored.complexity == config.complexity


class TestEnvironmentSuite:
    """Tests for EnvironmentSuite class"""
    
    def test_empty_suite(self):
        """Test empty suite initialization"""
        suite = EnvironmentSuite()
        
        assert len(suite) == 0
        assert suite.get_summary() == {}
    
    def test_add_config(self):
        """Test adding configurations"""
        suite = EnvironmentSuite()
        config = EnvironmentConfig(shelf_life=5)
        
        result = suite.add_config(config)
        
        assert result is True
        assert len(suite) == 1
    
    def test_no_duplicates(self):
        """Test duplicate detection"""
        suite = EnvironmentSuite()
        config1 = EnvironmentConfig(shelf_life=5, mean_demand=10.0)
        config2 = EnvironmentConfig(shelf_life=5, mean_demand=10.0)  # Same
        
        suite.add_config(config1)
        result = suite.add_config(config2)
        
        assert result is False
        assert len(suite) == 1
    
    def test_get_by_complexity(self):
        """Test filtering by complexity"""
        suite = EnvironmentSuite()
        suite.add_config(EnvironmentConfig(shelf_life=4, complexity="simple"))
        suite.add_config(EnvironmentConfig(shelf_life=5, complexity="simple"))
        suite.add_config(EnvironmentConfig(shelf_life=6, complexity="complex"))
        
        simple = suite.get_by_complexity("simple")
        complex_envs = suite.get_by_complexity("complex")
        
        assert len(simple) == 2
        assert len(complex_envs) == 1
    
    def test_get_summary(self):
        """Test summary generation"""
        suite = EnvironmentSuite()
        suite.add_config(EnvironmentConfig(shelf_life=4, complexity="simple"))
        suite.add_config(EnvironmentConfig(shelf_life=5, complexity="simple"))
        suite.add_config(EnvironmentConfig(shelf_life=6, complexity="complex"))
        
        summary = suite.get_summary()
        
        assert summary == {"simple": 2, "complex": 1}
    
    def test_save_and_load(self):
        """Test saving and loading suite"""
        suite = EnvironmentSuite(seed=123)
        suite.add_config(EnvironmentConfig(shelf_life=5, complexity="simple"))
        suite.add_config(EnvironmentConfig(shelf_life=6, complexity="moderate"))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            suite.save(filepath)
            loaded = EnvironmentSuite.load(filepath)
            
            assert len(loaded) == 2
            assert loaded.seed == 123
            assert loaded.get_summary() == {"simple": 1, "moderate": 1}
        finally:
            os.unlink(filepath)
    
    def test_iteration(self):
        """Test iterating over suite"""
        suite = EnvironmentSuite()
        suite.add_config(EnvironmentConfig(shelf_life=4))
        suite.add_config(EnvironmentConfig(shelf_life=5))
        
        configs = list(suite)
        assert len(configs) == 2
    
    def test_indexing(self):
        """Test indexing into suite"""
        suite = EnvironmentSuite()
        config = EnvironmentConfig(shelf_life=7)
        suite.add_config(config)
        
        assert suite[0].shelf_life == 7


class TestEnvironmentGeneration:
    """Tests for environment generation functions"""
    
    def test_generate_simple(self):
        """Test simple environment generation"""
        rng = np.random.RandomState(42)
        configs = generate_simple_environments(rng, count=10)
        
        assert len(configs) == 10
        for config in configs:
            assert config.complexity == "simple"
            assert config.num_suppliers == 2
            assert config.demand_type == "stationary"
    
    def test_generate_moderate(self):
        """Test moderate environment generation"""
        rng = np.random.RandomState(42)
        configs = generate_moderate_environments(rng, count=10)
        
        assert len(configs) == 10
        for config in configs:
            assert config.complexity == "moderate"
    
    def test_generate_complex(self):
        """Test complex environment generation"""
        rng = np.random.RandomState(42)
        configs = generate_complex_environments(rng, count=10)
        
        assert len(configs) == 10
        for config in configs:
            assert config.complexity == "complex"
            assert config.num_suppliers >= 2
    
    def test_generate_extreme(self):
        """Test extreme environment generation"""
        rng = np.random.RandomState(42)
        configs = generate_extreme_environments(rng, count=10)
        
        assert len(configs) == 10
        for config in configs:
            assert config.complexity == "extreme"
            assert config.num_suppliers >= 3
            assert config.demand_type == "composite"


class TestCanonicalSuite:
    """Tests for the canonical environment suite"""
    
    def test_canonical_suite_size(self):
        """Test canonical suite has 100+ environments"""
        suite = get_canonical_suite()
        
        assert len(suite) >= 100
    
    def test_canonical_suite_reproducibility(self):
        """Test canonical suite is reproducible"""
        suite1 = get_canonical_suite()
        suite2 = get_canonical_suite()
        
        assert len(suite1) == len(suite2)
        
        for c1, c2 in zip(suite1, suite2):
            assert c1.env_id == c2.env_id
    
    def test_canonical_suite_complexity_distribution(self):
        """Test canonical suite has all complexity levels"""
        suite = get_canonical_suite()
        summary = suite.get_summary()
        
        assert "simple" in summary
        assert "moderate" in summary
        assert "complex" in summary
        assert "extreme" in summary
        
        # Each category should have environments
        assert summary["simple"] >= 15
        assert summary["moderate"] >= 25
        assert summary["complex"] >= 30
        assert summary["extreme"] >= 15
    
    def test_no_duplicates_in_canonical(self):
        """Test canonical suite has no duplicates"""
        suite = get_canonical_suite()
        ids = [c.env_id for c in suite]
        
        assert len(ids) == len(set(ids))


class TestBuildEnvironment:
    """Tests for building environments from config"""
    
    def test_build_simple_environment(self):
        """Test building a simple environment"""
        config = EnvironmentConfig(
            shelf_life=5,
            mean_demand=10.0,
            num_suppliers=2,
            lead_times=(1, 3),
            unit_costs=(2.0, 1.0)
        )
        
        mdp = build_environment_from_config(config)
        
        assert isinstance(mdp, PerishableInventoryMDP)
        assert mdp.shelf_life == 5
        assert len(mdp.suppliers) == 2
    
    def test_build_environment_with_crisis(self):
        """Test building environment with crisis enabled"""
        config = EnvironmentConfig(
            shelf_life=5,
            enable_crisis=True,
            crisis_probability=0.05
        )
        
        mdp = build_environment_from_config(config)
        
        assert isinstance(mdp, EnhancedPerishableInventoryMDP)
        assert mdp.crisis_process is not None
    
    def test_build_environment_seasonal_demand(self):
        """Test building environment with seasonal demand"""
        config = EnvironmentConfig(
            demand_type="seasonal",
            seasonality_amplitude=0.2
        )
        
        mdp = build_environment_from_config(config)
        
        # Should not raise
        state = mdp.reset(seed=42)
        assert state is not None
    
    def test_build_environment_with_stochastic_lead_times(self):
        """Test building environment with stochastic lead times"""
        config = EnvironmentConfig(
            stochastic_lead_times=True,
            lead_time_variance=0.2
        )
        
        mdp = build_environment_from_config(config)
        
        assert mdp.stochastic_lead_times is not None
        assert len(mdp.stochastic_lead_times) > 0
    
    def test_build_3_supplier_environment(self):
        """Test building environment with 3 suppliers"""
        config = EnvironmentConfig(
            num_suppliers=3,
            lead_times=(1, 2, 4),
            unit_costs=(3.0, 2.0, 1.0),
            capacities=(100.0, 100.0, 100.0)
        )
        
        mdp = build_environment_from_config(config)
        
        assert len(mdp.suppliers) == 3
    
    def test_built_environment_runs(self):
        """Test that built environment can run simulation"""
        config = EnvironmentConfig(
            shelf_life=5,
            mean_demand=10.0,
            demand_type="composite",
            seasonality_amplitude=0.2,
            spike_probability=0.05,
            spike_multiplier=2.0
        )
        
        mdp = build_environment_from_config(config)
        state = mdp.reset(seed=42)
        
        # Run a few steps
        for _ in range(10):
            action = {s['id']: 5.0 for s in mdp.suppliers}
            result = mdp.step(state, action)
            state = result.next_state
            
            assert np.isfinite(result.costs.total_cost)


class TestIterEnvironments:
    """Tests for iter_environments utility"""
    
    def test_iter_all_environments(self):
        """Test iterating all environments"""
        count = 0
        for config, mdp in iter_environments():
            count += 1
            assert isinstance(config, EnvironmentConfig)
            assert isinstance(mdp, PerishableInventoryMDP)
            
            if count >= 5:  # Just test first 5
                break
        
        assert count == 5
    
    def test_iter_by_complexity(self):
        """Test iterating filtered by complexity"""
        for config, mdp in iter_environments(complexity="simple"):
            assert config.complexity == "simple"
            break  # Just test first one


class TestSuiteValidation:
    """Validation tests for the complete suite"""
    
    def test_all_environments_buildable(self):
        """Test that all environments in canonical suite can be built"""
        suite = get_canonical_suite()
        
        errors = []
        for i, config in enumerate(suite):
            try:
                mdp = build_environment_from_config(config)
                # Try to reset
                state = mdp.reset(seed=42)
                assert state is not None
            except Exception as e:
                errors.append(f"Config {i} ({config.env_id}): {e}")
        
        assert len(errors) == 0, f"Failed to build {len(errors)} environments: {errors[:5]}"
    
    def test_all_environments_steppable(self):
        """Test that all environments can execute steps"""
        suite = get_canonical_suite()
        
        # Test a sample of 20 environments
        sample_indices = np.random.RandomState(42).choice(
            len(suite), size=min(20, len(suite)), replace=False
        )
        
        errors = []
        for idx in sample_indices:
            config = suite[idx]
            try:
                mdp = build_environment_from_config(config)
                state = mdp.reset(seed=42)
                
                # Execute one step
                action = {s['id']: 5.0 for s in mdp.suppliers}
                result = mdp.step(state, action)
                
                assert np.isfinite(result.costs.total_cost)
            except Exception as e:
                errors.append(f"Config {idx} ({config.env_id}): {e}")
        
        assert len(errors) == 0, f"Failed to step {len(errors)} environments: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
