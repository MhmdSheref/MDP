"""
Custom Callbacks for Enhanced RL Training.

Implements:
1. CurriculumCallback: Progressive complexity training
2. ScheduleCallback: Learning rate and entropy annealing
3. BenchmarkCallback: Periodic RL vs TBS comparison
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class LinearSchedule:
    """Linear schedule for hyperparameter annealing.
    
    Value decreases linearly from initial_value to final_value
    over the course of training (based on remaining progress).
    """
    
    def __init__(
        self,
        initial_value: float,
        final_value: float = 0.0
    ):
        """
        Args:
            initial_value: Starting value at progress=1.0
            final_value: Ending value at progress=0.0
        """
        self.initial_value = initial_value
        self.final_value = final_value
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Compute value at given progress.
        
        Args:
            progress_remaining: Float from 1.0 (start) to 0.0 (end)
        
        Returns:
            Current value based on linear interpolation
        """
        return (
            self.final_value + 
            progress_remaining * (self.initial_value - self.final_value)
        )


class ScheduleCallback(BaseCallback):
    """Callback for logging and tracking hyperparameter schedules.
    
    Logs current learning rate and entropy coefficient to tensorboard.
    Useful for monitoring training dynamics.
    """
    
    def __init__(
        self,
        initial_lr: float = 3e-4,
        final_lr: float = 0.0,
        initial_ent_coef: float = 0.01,
        final_ent_coef: float = 0.001,
        log_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Args:
            initial_lr: Starting learning rate
            final_lr: Final learning rate
            initial_ent_coef: Starting entropy coefficient
            final_ent_coef: Final entropy coefficient
            log_freq: How often to log values
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.lr_schedule = LinearSchedule(initial_lr, final_lr)
        self.ent_schedule = LinearSchedule(initial_ent_coef, final_ent_coef)
        self.log_freq = log_freq
        
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
    
    def _on_step(self) -> bool:
        """Log schedule values periodically."""
        if self.n_calls % self.log_freq == 0:
            progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
            
            current_lr = self.lr_schedule(progress)
            current_ent = self.ent_schedule(progress)
            
            self.logger.record("schedule/learning_rate", current_lr)
            self.logger.record("schedule/entropy_coef", current_ent)
            self.logger.record("schedule/progress", 1.0 - progress)
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: LR={current_lr:.2e}, Ent={current_ent:.4f}")
        
        return True


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning across complexity levels.
    
    Progressively increases environment complexity as the agent improves.
    Starts with simple environments and advances to more complex ones
    when performance thresholds are met.
    
    Complexity levels: simple -> moderate -> complex -> extreme
    """
    
    COMPLEXITY_ORDER = ["simple", "moderate", "complex", "extreme"]
    
    def __init__(
        self,
        env_factory: Callable[[str], VecEnv],
        thresholds: Dict[str, float] = None,
        window_size: int = 100,
        min_episodes_per_level: int = 50,
        verbose: int = 0
    ):
        """
        Args:
            env_factory: Function that creates VecEnv for given complexity
            thresholds: Dict mapping complexity -> reward threshold to advance
            window_size: Rolling window size for computing mean reward
            min_episodes_per_level: Minimum episodes before advancing
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env_factory = env_factory
        self.thresholds = thresholds or {
            "simple": -5.0,      # Need mean reward > -5 to advance
            "moderate": -8.0,   # Need mean reward > -8 to advance
            "complex": -12.0,   # Need mean reward > -12 to advance
        }
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level
        
        self.current_level_idx = 0
        self.episode_rewards: deque = deque(maxlen=window_size)
        self.episodes_at_level = 0
        self.level_history: List[Dict[str, Any]] = []
    
    @property
    def current_complexity(self) -> str:
        """Get current complexity level name."""
        return self.COMPLEXITY_ORDER[self.current_level_idx]
    
    def _on_training_start(self) -> None:
        """Initialize with first complexity level."""
        self._set_environment(self.current_complexity)
        self.level_history.append({
            "complexity": self.current_complexity,
            "start_step": 0,
            "episodes": 0
        })
    
    def _on_step(self) -> bool:
        """Check for level advancement."""
        # Check for episode completions via info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                self.episodes_at_level += 1
                
                if self.verbose > 1:
                    print(f"Episode reward: {ep_reward:.2f}")
        
        # Check if we should advance
        if self._should_advance():
            self._advance_level()
        
        return True
    
    def _should_advance(self) -> bool:
        """Check if agent should advance to next level."""
        # Can't advance beyond extreme
        if self.current_level_idx >= len(self.COMPLEXITY_ORDER) - 1:
            return False
        
        # Need minimum episodes
        if self.episodes_at_level < self.min_episodes_per_level:
            return False
        
        # Need enough episodes for reliable mean
        if len(self.episode_rewards) < self.window_size // 2:
            return False
        
        # Check threshold
        mean_reward = np.mean(self.episode_rewards)
        threshold = self.thresholds.get(self.current_complexity, float('-inf'))
        
        return mean_reward > threshold
    
    def _advance_level(self) -> None:
        """Advance to next complexity level."""
        old_level = self.current_complexity
        
        # Update history for current level
        self.level_history[-1]["episodes"] = self.episodes_at_level
        self.level_history[-1]["end_step"] = self.num_timesteps
        self.level_history[-1]["final_mean_reward"] = np.mean(self.episode_rewards)
        
        # Advance
        self.current_level_idx += 1
        new_level = self.current_complexity
        
        if self.verbose > 0:
            print(f"\n{'='*50}")
            print(f"CURRICULUM: Advancing {old_level} -> {new_level}")
            print(f"Mean reward: {np.mean(self.episode_rewards):.2f}")
            print(f"{'='*50}\n")
        
        # Log to tensorboard
        self.logger.record("curriculum/level", self.current_level_idx)
        self.logger.record("curriculum/advancement_step", self.num_timesteps)
        
        # Reset tracking
        self.episode_rewards.clear()
        self.episodes_at_level = 0
        
        # Start new level
        self.level_history.append({
            "complexity": new_level,
            "start_step": self.num_timesteps,
            "episodes": 0
        })
        
        # Set new environment
        self._set_environment(new_level)
    
    def _set_environment(self, complexity: str) -> None:
        """Set training environment for given complexity.
        
        Important: After calling set_env(), we must reset _last_obs
        to prevent 'No previous observation was provided' errors.
        """
        new_env = self.env_factory(complexity)
        self.model.set_env(new_env)
        
        # CRITICAL: Reset the environment and update model's internal observation
        # This prevents the "No previous observation was provided" assertion error
        self.model._last_obs = new_env.reset()
        # Also need to reset these internal buffers for on-policy algorithms
        self.model._last_episode_starts = np.ones((new_env.num_envs,), dtype=bool)
        
        if self.verbose > 0:
            print(f"Curriculum: Set environment to {complexity}")


class BenchmarkCallback(BaseCallback):
    """Callback for periodic benchmarking against baseline policies.
    
    Compares RL agent against TBS and other baselines at regular intervals.
    Logs performance metrics to tensorboard and saves comparison data.
    """
    
    def __init__(
        self,
        eval_env: VecEnv,
        benchmark_freq: int = 100000,
        n_eval_episodes: int = 10,
        baseline_policies: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        verbose: int = 0
    ):
        """
        Args:
            eval_env: Environment for evaluation
            benchmark_freq: How often to benchmark (in timesteps)
            n_eval_episodes: Episodes per evaluation
            baseline_policies: Dict of {name: policy} for comparison
            save_path: Path to save benchmark results
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.benchmark_freq = benchmark_freq
        self.n_eval_episodes = n_eval_episodes
        self.baseline_policies = baseline_policies or {}
        self.save_path = save_path
        
        self.benchmark_results: List[Dict[str, Any]] = []
    
    def _on_step(self) -> bool:
        """Run benchmark at specified frequency."""
        if self.num_timesteps % self.benchmark_freq == 0 and self.num_timesteps > 0:
            self._run_benchmark()
        return True
    
    def _run_benchmark(self) -> None:
        """Execute full benchmark comparison."""
        if self.verbose > 0:
            print(f"\nRunning benchmark at step {self.num_timesteps}...")
        
        results = {
            "timestep": self.num_timesteps,
            "policies": {}
        }
        
        # Evaluate RL agent
        rl_metrics = self._evaluate_policy(self.model, "RL")
        results["policies"]["RL"] = rl_metrics
        
        # Log RL metrics
        self.logger.record("benchmark/rl_mean_cost", rl_metrics["mean_cost"])
        self.logger.record("benchmark/rl_fill_rate", rl_metrics["fill_rate"])
        
        # Evaluate baselines
        for name, policy in self.baseline_policies.items():
            baseline_metrics = self._evaluate_policy(policy, name)
            results["policies"][name] = baseline_metrics
            
            self.logger.record(f"benchmark/{name.lower()}_mean_cost", baseline_metrics["mean_cost"])
            self.logger.record(f"benchmark/{name.lower()}_fill_rate", baseline_metrics["fill_rate"])
        
        # Compute relative performance
        if "TBS" in self.baseline_policies:
            tbs_cost = results["policies"]["TBS"]["mean_cost"]
            rl_cost = rl_metrics["mean_cost"]
            if tbs_cost > 0:
                cost_ratio = rl_cost / tbs_cost
                results["rl_vs_tbs_ratio"] = cost_ratio
                self.logger.record("benchmark/rl_vs_tbs_ratio", cost_ratio)
        
        self.benchmark_results.append(results)
        
        if self.verbose > 0:
            print(f"  RL: cost={rl_metrics['mean_cost']:.2f}, fill={rl_metrics['fill_rate']:.2%}")
            for name, metrics in results["policies"].items():
                if name != "RL":
                    print(f"  {name}: cost={metrics['mean_cost']:.2f}, fill={metrics['fill_rate']:.2%}")
    
    def _evaluate_policy(
        self,
        policy: Any,
        name: str
    ) -> Dict[str, float]:
        """Evaluate a single policy."""
        total_costs = []
        total_fill_rates = []
        total_spoilage = []
        
        # Handle both SB3 models and custom policies
        is_sb3_model = hasattr(policy, 'predict')
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode_cost = 0.0
            episode_demand = 0.0
            episode_sales = 0.0
            episode_spoilage = 0.0
            done = False
            steps = 0
            
            while not done and steps < 500:  # Max 500 steps per episode
                if is_sb3_model:
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    # Custom policy interface
                    # Need to get underlying env state
                    if hasattr(self.eval_env, 'envs'):
                        env = self.eval_env.envs[0]
                        if hasattr(env, 'current_state') and hasattr(env, 'mdp'):
                            action_dict = policy.get_action(env.current_state, env.mdp)
                            # Convert to encoded action (simplified)
                            action = self._encode_action(env, action_dict)
                        else:
                            action = self.eval_env.action_space.sample()
                    else:
                        action = self.eval_env.action_space.sample()
                
                obs, reward, done, info = self.eval_env.step(action)
                
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                
                episode_cost += info.get('total_cost', 0)
                episode_demand += info.get('demand', 0)
                episode_sales += info.get('sales', 0)
                episode_spoilage += info.get('spoilage', 0)
                steps += 1
                
                if isinstance(done, np.ndarray):
                    done = done[0]
            
            total_costs.append(episode_cost)
            fill_rate = episode_sales / max(episode_demand, 1e-6)
            total_fill_rates.append(fill_rate)
            total_spoilage.append(episode_spoilage)
        
        return {
            "mean_cost": np.mean(total_costs),
            "std_cost": np.std(total_costs),
            "fill_rate": np.mean(total_fill_rates),
            "mean_spoilage": np.mean(total_spoilage)
        }
    
    def _encode_action(self, env: Any, action_dict: Dict[int, float]) -> np.ndarray:
        """Convert policy action dict to environment action."""
        # This is a simplified encoding - matches gym_env_v2 decoding logic
        action = []
        for sid in env.supplier_order:
            qty = action_dict.get(sid, 0)
            bins = env.supplier_action_bins[sid]
            # Find closest bin
            idx = np.argmin(np.abs(np.array(bins) - qty))
            action.append(idx)
        return np.array([action])


def create_lr_schedule(
    initial_lr: float = 3e-4,
    final_lr: float = 0.0
) -> Callable[[float], float]:
    """Create learning rate schedule function for SB3.
    
    Args:
        initial_lr: Starting learning rate
        final_lr: Final learning rate
    
    Returns:
        Schedule function: progress_remaining -> lr
    """
    schedule = LinearSchedule(initial_lr, final_lr)
    return schedule


def create_entropy_schedule(
    initial_ent: float = 0.01,
    final_ent: float = 0.001
) -> Callable[[float], float]:
    """Create entropy coefficient schedule.
    
    Args:
        initial_ent: Starting entropy coefficient
        final_ent: Final entropy coefficient
    
    Returns:
        Schedule function: progress_remaining -> ent_coef
    """
    schedule = LinearSchedule(initial_ent, final_ent)
    return schedule
