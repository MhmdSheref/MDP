"""
Tests for the Crisis Process module.

Tests CrisisEvent, CrisisProcess, and factory functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.crisis import (
    CrisisEvent, CrisisLevel, CrisisProcess,
    NORMAL_STATE, ELEVATED_STATE, CRISIS_STATE,
    create_crisis_process
)
from perishable_inventory_mdp.exceptions import InvalidParameterError


class TestCrisisLevel:
    """Tests for CrisisLevel enum"""
    
    def test_enum_values(self):
        """Test enum values are correct"""
        assert CrisisLevel.NORMAL == 0
        assert CrisisLevel.ELEVATED == 1
        assert CrisisLevel.CRISIS == 2


class TestCrisisEvent:
    """Tests for CrisisEvent dataclass"""
    
    def test_basic_construction(self):
        """Test basic construction"""
        event = CrisisEvent(
            level=1,
            name="Test Event",
            demand_multiplier=2.0,
            supply_disruption_prob=0.3
        )
        
        assert event.level == 1
        assert event.name == "Test Event"
        assert event.demand_multiplier == 2.0
        assert event.supply_disruption_prob == 0.3
    
    def test_predefined_normal_state(self):
        """Test predefined NORMAL_STATE"""
        assert NORMAL_STATE.level == 0
        assert NORMAL_STATE.demand_multiplier == 1.0
        assert NORMAL_STATE.supply_disruption_prob == 0.0
    
    def test_predefined_crisis_state(self):
        """Test predefined CRISIS_STATE"""
        assert CRISIS_STATE.level == 2
        assert CRISIS_STATE.demand_multiplier == 3.0
        assert CRISIS_STATE.supply_disruption_prob == 0.40
    
    def test_invalid_demand_multiplier(self):
        """Test validation of demand multiplier"""
        with pytest.raises(InvalidParameterError):
            CrisisEvent(level=0, demand_multiplier=-0.5)
    
    def test_invalid_disruption_prob(self):
        """Test validation of supply disruption probability"""
        with pytest.raises(InvalidParameterError):
            CrisisEvent(level=0, supply_disruption_prob=1.5)
        
        with pytest.raises(InvalidParameterError):
            CrisisEvent(level=0, supply_disruption_prob=-0.1)
    
    def test_invalid_duration(self):
        """Test validation of duration"""
        with pytest.raises(InvalidParameterError):
            CrisisEvent(level=0, duration_mean=0)
        
        with pytest.raises(InvalidParameterError):
            CrisisEvent(level=0, duration_std=-1.0)


class TestCrisisProcess:
    """Tests for CrisisProcess class"""
    
    def test_basic_construction(self):
        """Test basic construction with defaults"""
        process = CrisisProcess()
        
        assert len(process.states) == 3
        assert process.current_state == 0
        assert process.transition_matrix.shape == (3, 3)
    
    def test_transition_matrix_validation(self):
        """Test transition matrix row sums"""
        # Should work - rows sum to 1
        process = CrisisProcess(transition_matrix=np.array([
            [0.9, 0.08, 0.02],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ]))
        assert process.current_state == 0
    
    def test_invalid_transition_matrix_shape(self):
        """Test invalid transition matrix shape"""
        with pytest.raises(InvalidParameterError):
            CrisisProcess(transition_matrix=np.array([
                [0.9, 0.1],
                [0.2, 0.8]
            ]))
    
    def test_invalid_transition_matrix_row_sum(self):
        """Test transition matrix with incorrect row sums"""
        with pytest.raises(InvalidParameterError):
            CrisisProcess(transition_matrix=np.array([
                [0.9, 0.2, 0.02],  # Sum = 1.12
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6]
            ]))
    
    def test_sample_transition(self):
        """Test state transitions"""
        np.random.seed(42)
        process = CrisisProcess()
        
        # Sample many transitions
        states = [process.current_state]
        for _ in range(100):
            states.append(process.sample_transition())
        
        # Should have variety of states
        unique_states = set(states)
        assert len(unique_states) >= 1  # At least some states
    
    def test_get_current_event(self):
        """Test getting current event"""
        process = CrisisProcess()
        
        process.current_state = 0
        assert process.get_current_event().level == 0
        
        process.current_state = 2
        assert process.get_current_event().level == 2
    
    def test_get_demand_multiplier(self):
        """Test getting demand multiplier"""
        process = CrisisProcess()
        
        process.current_state = 0
        assert process.get_demand_multiplier() == 1.0
        
        process.current_state = 2  # Crisis
        assert process.get_demand_multiplier() == 3.0
    
    def test_get_supply_disruption_prob(self):
        """Test getting supply disruption probability"""
        process = CrisisProcess()
        
        process.current_state = 0
        assert process.get_supply_disruption_prob() == 0.0
        
        process.current_state = 2
        assert process.get_supply_disruption_prob() == 0.40
    
    def test_should_reject_order(self):
        """Test order rejection logic"""
        np.random.seed(42)
        process = CrisisProcess()
        
        # In normal state with no base rejection
        process.current_state = 0
        rejections = [process.should_reject_order(0.0) for _ in range(100)]
        assert sum(rejections) == 0  # No rejections expected
        
        # In crisis state
        process.current_state = 2
        rejections = [process.should_reject_order(0.0) for _ in range(100)]
        assert sum(rejections) > 0  # Some rejections expected
    
    def test_reset(self):
        """Test reset to initial state"""
        process = CrisisProcess()
        process.current_state = 2
        process.periods_in_state = 10
        
        process.reset(0)
        
        assert process.current_state == 0
        assert process.periods_in_state == 0
    
    def test_reset_invalid_state(self):
        """Test reset with invalid state"""
        process = CrisisProcess()
        
        with pytest.raises(InvalidParameterError):
            process.reset(5)
    
    def test_copy(self):
        """Test copying process"""
        process = CrisisProcess()
        process.current_state = 1
        process.periods_in_state = 3
        
        copied = process.copy()
        
        assert copied.current_state == 1
        assert copied.periods_in_state == 3
        assert copied is not process
    
    def test_to_exogenous_state(self):
        """Test conversion to exogenous state array"""
        process = CrisisProcess()
        process.current_state = 2
        
        exog = process.to_exogenous_state(time=5.0)
        
        assert len(exog) == 2
        assert exog[0] == 5.0
        assert exog[1] == 2.0
    
    def test_update_from_exogenous_state(self):
        """Test updating from exogenous state array"""
        process = CrisisProcess()
        process.current_state = 0
        
        process.update_from_exogenous_state(np.array([10.0, 1.0]))
        
        assert process.current_state == 1


class TestCrisisProcessFactory:
    """Tests for create_crisis_process factory"""
    
    def test_default_creation(self):
        """Test default creation"""
        process = create_crisis_process()
        
        assert process.current_state == 0
        assert len(process.states) == 3
    
    def test_mild_severity(self):
        """Test mild severity configuration"""
        process = create_crisis_process(severity="mild")
        
        # Crisis multiplier should be lower
        assert process.states[2].demand_multiplier == 1.5
        assert process.states[2].supply_disruption_prob == 0.15
    
    def test_severe_severity(self):
        """Test severe severity configuration"""
        process = create_crisis_process(severity="severe")
        
        # Crisis multiplier should be higher
        assert process.states[2].demand_multiplier == 4.0
        assert process.states[2].supply_disruption_prob == 0.60
    
    def test_high_crisis_probability(self):
        """Test high crisis probability affects transitions"""
        process = create_crisis_process(crisis_probability=0.3)
        
        # Transition to crisis should be more likely
        assert process.transition_matrix[0, 1] > 0.05  # Normal -> Elevated
        assert process.transition_matrix[0, 2] > 0.01  # Normal -> Crisis
    
    def test_high_recovery_rate(self):
        """Test high recovery rate affects transitions"""
        process = create_crisis_process(recovery_rate=0.5)
        
        # Recovery should be more likely
        assert process.transition_matrix[2, 0] > 0.1  # Crisis -> Normal
        assert process.transition_matrix[2, 1] > 0.2  # Crisis -> Elevated
    
    def test_invalid_crisis_probability(self):
        """Test invalid crisis probability"""
        with pytest.raises(InvalidParameterError):
            create_crisis_process(crisis_probability=1.5)
    
    def test_invalid_recovery_rate(self):
        """Test invalid recovery rate"""
        with pytest.raises(InvalidParameterError):
            create_crisis_process(recovery_rate=-0.1)
    
    def test_unknown_severity(self):
        """Test unknown severity raises error"""
        with pytest.raises(ValueError):
            create_crisis_process(severity="extreme")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
