"""
Tests for Enhanced RL Training.

Tests curriculum learning, schedule callbacks, and benchmarking functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from collections import deque

import sys
sys.path.insert(0, '.')

from colab_training.callbacks import (
    LinearSchedule,
    ScheduleCallback,
    CurriculumCallback,
    BenchmarkCallback,
    create_lr_schedule,
    create_entropy_schedule
)
from colab_training.benchmark import (
    EvaluationResult,
    ComparisonReport,
    evaluate_policy,
    get_tbs_policy_for_env,
    get_basestock_policy_for_env,
    generate_performance_report
)
from colab_training.gym_env import create_gym_env


class TestLinearSchedule:
    """Tests for LinearSchedule."""
    
    def test_linear_schedule_start(self):
        """Test schedule returns initial value at start."""
        schedule = LinearSchedule(initial_value=1.0, final_value=0.0)
        
        # progress_remaining=1.0 means start
        assert schedule(1.0) == 1.0
    
    def test_linear_schedule_end(self):
        """Test schedule returns final value at end."""
        schedule = LinearSchedule(initial_value=1.0, final_value=0.0)
        
        # progress_remaining=0.0 means end
        assert schedule(0.0) == 0.0
    
    def test_linear_schedule_midpoint(self):
        """Test schedule returns midpoint at half progress."""
        schedule = LinearSchedule(initial_value=1.0, final_value=0.0)
        
        # Midpoint
        assert schedule(0.5) == 0.5
    
    def test_linear_schedule_custom_values(self):
        """Test schedule with custom initial and final values."""
        schedule = LinearSchedule(initial_value=0.003, final_value=0.0003)
        
        assert np.isclose(schedule(1.0), 0.003)
        assert np.isclose(schedule(0.0), 0.0003)
        assert np.isclose(schedule(0.5), 0.00165)


class TestScheduleCallback:
    """Tests for ScheduleCallback."""
    
    def test_schedule_callback_init(self):
        """Test callback initialization."""
        callback = ScheduleCallback(
            initial_lr=3e-4,
            final_lr=0.0,
            initial_ent_coef=0.01,
            final_ent_coef=0.001
        )
        
        assert callback.initial_lr == 3e-4
        assert callback.final_lr == 0.0
        assert callback.initial_ent_coef == 0.01
        assert callback.final_ent_coef == 0.001
    
    def test_schedule_callback_lr_schedule(self):
        """Test learning rate schedule computation."""
        callback = ScheduleCallback(
            initial_lr=1.0,
            final_lr=0.0
        )
        
        # Test the internal schedule
        assert callback.lr_schedule(1.0) == 1.0
        assert callback.lr_schedule(0.0) == 0.0


class TestCurriculumCallback:
    """Tests for CurriculumCallback."""
    
    def test_curriculum_initial_complexity(self):
        """Test curriculum starts at simple complexity."""
        mock_factory = MagicMock()
        
        callback = CurriculumCallback(
            env_factory=mock_factory,
            thresholds={"simple": -5.0, "moderate": -8.0},
            min_episodes_per_level=10
        )
        
        assert callback.current_level_idx == 0
        assert callback.current_complexity == "simple"
    
    def test_curriculum_complexity_order(self):
        """Test complexity levels are in correct order."""
        assert CurriculumCallback.COMPLEXITY_ORDER == [
            "simple", "moderate", "complex", "extreme"
        ]
    
    def test_should_advance_false_when_not_enough_episodes(self):
        """Test advancement blocked when not enough episodes."""
        mock_factory = MagicMock()
        
        callback = CurriculumCallback(
            env_factory=mock_factory,
            thresholds={"simple": -5.0},
            min_episodes_per_level=50,
            window_size=100
        )
        
        # Add few episodes
        callback.episodes_at_level = 10
        callback.episode_rewards = deque([0.0] * 10)
        
        assert callback._should_advance() is False
    
    def test_should_advance_false_when_below_threshold(self):
        """Test advancement blocked when reward below threshold."""
        mock_factory = MagicMock()
        
        callback = CurriculumCallback(
            env_factory=mock_factory,
            thresholds={"simple": -5.0},
            min_episodes_per_level=10,
            window_size=20
        )
        
        # Add enough episodes with low rewards
        callback.episodes_at_level = 50
        callback.episode_rewards = deque([-10.0] * 20)  # Below threshold
        
        assert callback._should_advance() == False


class TestBenchmarkCallback:
    """Tests for BenchmarkCallback."""
    
    def test_benchmark_callback_init(self):
        """Test benchmark callback initialization."""
        mock_env = MagicMock()
        
        callback = BenchmarkCallback(
            eval_env=mock_env,
            benchmark_freq=100000,
            n_eval_episodes=10
        )
        
        assert callback.benchmark_freq == 100000
        assert callback.n_eval_episodes == 10
        assert callback.benchmark_results == []
    
    def test_benchmark_stores_results(self):
        """Test that benchmark stores results."""
        mock_env = MagicMock()
        
        callback = BenchmarkCallback(
            eval_env=mock_env,
            benchmark_freq=100,
            n_eval_episodes=1
        )
        
        # Initially empty
        assert len(callback.benchmark_results) == 0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test creating evaluation result."""
        result = EvaluationResult(
            env_id="test_env",
            complexity="simple",
            policy_name="RL",
            mean_cost=100.0,
            std_cost=10.0,
            mean_fill_rate=0.95,
            std_fill_rate=0.02
        )
        
        assert result.env_id == "test_env"
        assert result.mean_cost == 100.0
        assert result.mean_fill_rate == 0.95
    
    def test_evaluation_result_to_dict(self):
        """Test converting result to dict."""
        result = EvaluationResult(
            env_id="test",
            complexity="moderate",
            policy_name="TBS",
            mean_cost=50.0,
            std_cost=5.0,
            mean_fill_rate=0.98,
            std_fill_rate=0.01,
            n_episodes=10
        )
        
        d = result.to_dict()
        
        assert d["env_id"] == "test"
        assert d["policy_name"] == "TBS"
        assert d["mean_cost"] == 50.0
        assert d["n_episodes"] == 10


class TestComparisonReport:
    """Tests for ComparisonReport."""
    
    def test_report_add_result(self):
        """Test adding results to report."""
        report = ComparisonReport()
        
        result = EvaluationResult(
            env_id="env1",
            complexity="simple",
            policy_name="RL",
            mean_cost=100.0,
            std_cost=10.0,
            mean_fill_rate=0.95,
            std_fill_rate=0.02
        )
        
        report.add_result(result)
        
        assert len(report.results) == 1
    
    def test_report_to_dataframe(self):
        """Test converting report to DataFrame."""
        report = ComparisonReport()
        
        report.add_result(EvaluationResult(
            env_id="env1",
            complexity="simple",
            policy_name="RL",
            mean_cost=100.0,
            std_cost=10.0,
            mean_fill_rate=0.95,
            std_fill_rate=0.02
        ))
        
        report.add_result(EvaluationResult(
            env_id="env1",
            complexity="simple",
            policy_name="TBS",
            mean_cost=50.0,
            std_cost=5.0,
            mean_fill_rate=0.98,
            std_fill_rate=0.01
        ))
        
        df = report.to_dataframe()
        
        assert len(df) == 2
        assert "policy_name" in df.columns
        assert "mean_cost" in df.columns


class TestGetPolicies:
    """Tests for policy factory functions."""
    
    def test_get_tbs_policy(self):
        """Test getting TBS policy for environment."""
        env = create_gym_env(
            shelf_life=5,
            mean_demand=10.0,
            fast_lead_time=1,
            slow_lead_time=3,
            fast_cost=2.0,
            slow_cost=1.0
        )
        
        tbs = get_tbs_policy_for_env(env)
        
        assert tbs is not None
        assert hasattr(tbs, 'get_action')
        assert hasattr(tbs, 'base_stock_level')
        assert hasattr(tbs, 'reorder_point')
    
    def test_get_basestock_policy(self):
        """Test getting base stock policy for environment."""
        env = create_gym_env()
        
        bs = get_basestock_policy_for_env(env, target_level=60.0)
        
        assert bs is not None
        assert hasattr(bs, 'get_action')
        assert bs.target_level == 60.0


class TestScheduleFunctions:
    """Tests for schedule helper functions."""
    
    def test_create_lr_schedule(self):
        """Test LR schedule creation."""
        schedule = create_lr_schedule(3e-4, 0.0)
        
        assert callable(schedule)
        assert schedule(1.0) == 3e-4
        assert schedule(0.0) == 0.0
    
    def test_create_entropy_schedule(self):
        """Test entropy schedule creation."""
        schedule = create_entropy_schedule(0.01, 0.001)
        
        assert callable(schedule)
        assert np.isclose(schedule(1.0), 0.01)
        assert np.isclose(schedule(0.0), 0.001)


class TestGenerateReport:
    """Tests for report generation."""
    
    def test_generate_performance_report(self):
        """Test generating human-readable report."""
        report = ComparisonReport()
        
        # Add some test data
        for complexity in ["simple", "moderate"]:
            for policy in ["RL", "TBS"]:
                report.add_result(EvaluationResult(
                    env_id=f"env_{complexity}",
                    complexity=complexity,
                    policy_name=policy,
                    mean_cost=100.0 if policy == "RL" else 50.0,
                    std_cost=10.0,
                    mean_fill_rate=0.95,
                    std_fill_rate=0.02
                ))
        
        text_report = generate_performance_report(report)
        
        assert "RL vs TBS" in text_report
        assert "simple" in text_report
        assert "moderate" in text_report


class TestIntegration:
    """Integration tests for training components."""
    
    def test_evaluate_policy_with_gym_env(self):
        """Test evaluating a random policy on gym env."""
        env = create_gym_env()
        
        # Create a mock random policy
        class RandomPolicy:
            def predict(self, obs, deterministic=False):
                if hasattr(env, 'action_space'):
                    return env.action_space.sample(), None
                return np.array([1, 1]), None
        
        result = evaluate_policy(
            policy=RandomPolicy(),
            env=env,
            n_episodes=2,
            max_steps=10,
            policy_name="Random",
            env_id="test",
            complexity="simple"
        )
        
        assert result.policy_name == "Random"
        assert result.n_episodes == 2
        assert result.mean_cost >= 0  # Costs should be non-negative
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        env = create_gym_env()
        
        # Get TBS policy
        tbs = get_tbs_policy_for_env(env)
        
        # Evaluate it
        result = evaluate_policy(
            policy=tbs,
            env=env,
            n_episodes=2,
            max_steps=20,
            policy_name="TBS",
            env_id="test_env",
            complexity="simple"
        )
        
        assert result.policy_name == "TBS"
        assert 0.0 <= result.mean_fill_rate <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
