"""
Tests for the Extended Demand Processes.

Tests DemandSpikeProcess, TrendDemand, and CompositeDemand classes.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.demand import (
    DemandSpikeProcess, TrendDemand, CompositeDemand,
    PoissonDemand, create_demand_scenario
)
from perishable_inventory_mdp.exceptions import InvalidDemandError


class TestDemandSpikeProcess:
    """Tests for DemandSpikeProcess"""
    
    def test_basic_properties(self):
        """Test basic construction and properties"""
        base = PoissonDemand(10.0)
        spiky = DemandSpikeProcess(base, spike_prob=0.1, spike_multiplier=3.0)
        
        assert spiky.base_process is base
        assert spiky.spike_prob == 0.1
        assert spiky.spike_multiplier == 3.0
    
    def test_mean_calculation(self):
        """Test expected demand accounts for spikes"""
        base = PoissonDemand(10.0)
        spiky = DemandSpikeProcess(base, spike_prob=0.1, spike_multiplier=3.0)
        
        # E[D] = base * (1 - p + p * multiplier)
        # E[D] = 10 * (0.9 + 0.1 * 3) = 10 * 1.2 = 12
        expected_mean = 10.0 * (1 - 0.1 + 0.1 * 3.0)
        assert_almost_equal(spiky.mean(), expected_mean)
    
    def test_spike_increases_demand(self):
        """Test that spikes lead to higher demand on average"""
        np.random.seed(42)
        base = PoissonDemand(10.0)
        spiky = DemandSpikeProcess(base, spike_prob=0.2, spike_multiplier=3.0)
        
        samples = [spiky.sample() for _ in range(1000)]
        empirical_mean = np.mean(samples)
        theoretical_mean = spiky.mean()
        
        # Should be reasonably close (within 15% for 1000 samples)
        assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.15
    
    def test_invalid_spike_prob(self):
        """Test validation of spike probability"""
        base = PoissonDemand(10.0)
        
        with pytest.raises(InvalidDemandError):
            DemandSpikeProcess(base, spike_prob=-0.1)
        
        with pytest.raises(InvalidDemandError):
            DemandSpikeProcess(base, spike_prob=1.5)
    
    def test_invalid_spike_multiplier(self):
        """Test validation of spike multiplier"""
        base = PoissonDemand(10.0)
        
        with pytest.raises(InvalidDemandError):
            DemandSpikeProcess(base, spike_multiplier=0.5)
    
    def test_exogenous_state_delegation(self):
        """Test that exogenous state update is delegated to base"""
        base = PoissonDemand(10.0)
        spiky = DemandSpikeProcess(base, spike_prob=0.1, spike_multiplier=2.0)
        
        # PoissonDemand returns current_state unchanged
        state = np.array([5.0])
        new_state = spiky.update_exogenous_state(state)
        assert new_state is state


class TestTrendDemand:
    """Tests for TrendDemand"""
    
    def test_basic_properties(self):
        """Test basic construction"""
        trend = TrendDemand(10.0, trend_rate=0.02, trend_power=1.0)
        
        assert trend.base_rate == 10.0
        assert trend.trend_rate == 0.02
        assert trend.trend_power == 1.0
    
    def test_trend_at_time_zero(self):
        """Test demand at t=0 equals base rate"""
        trend = TrendDemand(10.0, trend_rate=0.05)
        
        # At t=0, should return base rate
        assert_almost_equal(trend.mean(np.array([0.0])), 10.0)
    
    def test_positive_trend(self):
        """Test demand increases with positive trend"""
        trend = TrendDemand(10.0, trend_rate=0.1)
        
        # At t=10: rate = 10 * (1 + 0.1 * 10) = 10 * 2 = 20
        expected = 10.0 * (1 + 0.1 * 10)
        assert_almost_equal(trend.mean(np.array([10.0])), expected)
    
    def test_negative_trend(self):
        """Test demand decreases with negative trend, respecting min_rate"""
        trend = TrendDemand(10.0, trend_rate=-0.05, min_rate=2.0)
        
        # At t=0: rate = 10
        assert_almost_equal(trend.mean(np.array([0.0])), 10.0)
        
        # At t=100: rate = 10 * (1 - 5) = 10 * (-4) -> clipped to min_rate = 2
        assert_almost_equal(trend.mean(np.array([100.0])), 2.0)
    
    def test_max_rate_ceiling(self):
        """Test demand respects max_rate ceiling"""
        trend = TrendDemand(10.0, trend_rate=0.5, max_rate=30.0)
        
        # At t=10: rate = 10 * (1 + 5) = 60 -> clipped to 30
        assert_almost_equal(trend.mean(np.array([10.0])), 30.0)
    
    def test_exogenous_state_update(self):
        """Test time advances in exogenous state"""
        trend = TrendDemand(10.0, trend_rate=0.01)
        
        state = np.array([5.0])
        new_state = trend.update_exogenous_state(state)
        
        assert new_state[0] == 6.0
    
    def test_null_state_returns_base(self):
        """Test None state returns base rate"""
        trend = TrendDemand(15.0, trend_rate=0.1)
        
        assert_almost_equal(trend.mean(None), 15.0)
    
    def test_invalid_base_rate(self):
        """Test validation of base rate"""
        with pytest.raises(InvalidDemandError):
            TrendDemand(-5.0)
    
    def test_quadratic_trend(self):
        """Test non-linear trend with power > 1"""
        trend = TrendDemand(10.0, trend_rate=0.1, trend_power=2.0)
        
        # At t=5: rate = 10 * (1 + 0.1 * 5)^2 = 10 * 1.5^2 = 22.5
        expected = 10.0 * (1.5 ** 2)
        assert_almost_equal(trend.mean(np.array([5.0])), expected)


class TestCompositeDemand:
    """Tests for CompositeDemand"""
    
    def test_basic_properties(self):
        """Test basic construction"""
        comp = CompositeDemand(
            10.0,
            seasonality_amplitude=0.3,
            seasonality_period=12,
            trend_rate=0.01,
            spike_prob=0.05
        )
        
        assert comp.base_rate == 10.0
        assert comp.seasonality_amplitude == 0.3
        assert comp.trend_rate == 0.01
        assert comp.spike_prob == 0.05
    
    def test_seasonality_only(self):
        """Test seasonality pattern"""
        comp = CompositeDemand(10.0, seasonality_amplitude=0.5, seasonality_period=4)
        
        # At t=0: rate = 10 * (1 + 0.5 * sin(0)) = 10
        assert_almost_equal(comp._get_rate(np.array([0.0, 0.0])), 10.0)
        
        # At t=1 (quarter period): rate = 10 * (1 + 0.5 * sin(π/2)) = 10 * 1.5 = 15
        at_t1 = comp._get_rate(np.array([1.0, 0.0]))
        assert_almost_equal(at_t1, 15.0, decimal=5)
    
    def test_crisis_modulation(self):
        """Test crisis state multiplies demand"""
        comp = CompositeDemand(10.0, crisis_multipliers=(1.0, 1.5, 3.0))
        
        # Normal state
        assert_almost_equal(comp._get_rate(np.array([0.0, 0.0])), 10.0)
        
        # Elevated state
        assert_almost_equal(comp._get_rate(np.array([0.0, 1.0])), 15.0)
        
        # Crisis state
        assert_almost_equal(comp._get_rate(np.array([0.0, 2.0])), 30.0)
    
    def test_combined_effects(self):
        """Test multiple effects combine"""
        comp = CompositeDemand(
            10.0,
            seasonality_amplitude=0.0,
            trend_rate=0.1,
            crisis_multipliers=(1.0, 2.0, 3.0)
        )
        
        # At t=5 in crisis: rate = 10 * (1 + 0.1*5) * 3 = 10 * 1.5 * 3 = 45
        rate = comp._get_rate(np.array([5.0, 2.0]))
        assert_almost_equal(rate, 45.0)
    
    def test_spike_mean_adjustment(self):
        """Test mean accounts for spike probability"""
        comp = CompositeDemand(10.0, spike_prob=0.1, spike_multiplier=2.0)
        
        # Mean = base * (1 - p + p * mult) = 10 * (0.9 + 0.1*2) = 10 * 1.1 = 11
        expected = 10.0 * (1 - 0.1 + 0.1 * 2.0)
        assert_almost_equal(comp.mean(np.array([0.0, 0.0])), expected)
    
    def test_exogenous_state_update(self):
        """Test time advances, crisis state preserved"""
        comp = CompositeDemand(10.0)
        
        state = np.array([3.0, 1.0])
        new_state = comp.update_exogenous_state(state)
        
        assert new_state[0] == 4.0  # Time advanced
        assert new_state[1] == 1.0  # Crisis state preserved
    
    def test_null_state_initialization(self):
        """Test None state initializes correctly"""
        comp = CompositeDemand(10.0)
        
        new_state = comp.update_exogenous_state(None)
        assert len(new_state) == 2
        assert new_state[0] == 1.0  # Time = 1
        assert new_state[1] == 0.0  # Normal crisis state
    
    def test_invalid_seasonality_amplitude(self):
        """Test validation of seasonality amplitude"""
        with pytest.raises(InvalidDemandError):
            CompositeDemand(10.0, seasonality_amplitude=-0.1)
        
        with pytest.raises(InvalidDemandError):
            CompositeDemand(10.0, seasonality_amplitude=1.0)
    
    def test_invalid_spike_prob(self):
        """Test validation of spike probability"""
        with pytest.raises(InvalidDemandError):
            CompositeDemand(10.0, spike_prob=1.5)


class TestExtendedDemandScenarioFactory:
    """Tests for extended create_demand_scenario factory"""
    
    def test_spiky_scenario(self):
        """Test spiky scenario creation"""
        demand = create_demand_scenario(10.0, "spiky", spike_prob=0.1, spike_multiplier=2.5)
        
        assert isinstance(demand, DemandSpikeProcess)
        assert demand.spike_prob == 0.1
        assert demand.spike_multiplier == 2.5
    
    def test_trending_scenario(self):
        """Test trending scenario creation"""
        demand = create_demand_scenario(10.0, "trending", trend_rate=0.05, min_rate=5.0)
        
        assert isinstance(demand, TrendDemand)
        assert demand.trend_rate == 0.05
        assert demand.min_rate == 5.0
    
    def test_composite_scenario(self):
        """Test composite scenario creation"""
        demand = create_demand_scenario(
            10.0, "composite",
            seasonality_amplitude=0.3,
            trend_rate=0.01,
            spike_prob=0.05
        )
        
        assert isinstance(demand, CompositeDemand)
        assert demand.seasonality_amplitude == 0.3
        assert demand.trend_rate == 0.01
        assert demand.spike_prob == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
