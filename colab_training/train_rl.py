"""
Enhanced RL Training Script.

Improvements:
1. 5M training timesteps (configurable)
2. Linear learning rate annealing
3. Entropy coefficient decay
4. Curriculum learning (simple -> complex environments)
5. Periodic benchmarking against TBS baseline
6. Uses gym_env (cost-aware wrapper)
7. Multi-environment vectorized training

Usage:
    python train_rl.py                    # Full training
    python train_rl.py --test-mode        # Quick test run
    python train_rl.py --timesteps 1000000  # Custom timesteps
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit

from colab_training.gym_env import (
    PerishableInventoryGymWrapper,
    RewardConfig,
    create_gym_env
)
from colab_training.environment_suite import (
    EnvironmentSuite,
    EnvironmentConfig,
    create_environment_suite,
    build_environment_from_config
)
from colab_training.callbacks import (
    ScheduleCallback,
    CurriculumCallback,
    BenchmarkCallback,
    create_lr_schedule,
    create_entropy_schedule
)
from colab_training.benchmark import get_tbs_policy_for_env, get_basestock_policy_for_env


# Default training configuration
DEFAULT_CONFIG = {
    "total_timesteps": 5_000_000,
    "initial_learning_rate": 3e-4,
    "final_learning_rate": 0.0,
    "initial_entropy_coef": 0.01,
    "final_entropy_coef": 0.001,
    "curriculum_enabled": True,
    "curriculum_thresholds": {
        "simple": -5.0,
        "moderate": -8.0,
        "complex": -12.0
    },
    "n_envs": 8,
    "episode_length": 500,
    "checkpoint_freq": 100_000,
    "eval_freq": 50_000,
    "n_eval_episodes": 10,
    "benchmark_freq": 100_000,
    "seed": 42,
    "policy_kwargs": {
        "net_arch": [256, 256]
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load training configuration.
    
    Args:
        config_path: Path to JSON config file (optional)
    
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            custom = json.load(f)
        
        # Merge training_params if present
        if "training_params" in custom:
            config.update(custom["training_params"])
        else:
            config.update(custom)
    
    return config


def create_env_from_config(
    env_config: EnvironmentConfig,
    reward_config: Optional[RewardConfig] = None,
    episode_length: int = 500
) -> PerishableInventoryGymWrapper:
    """Create gym environment from EnvironmentConfig.
    
    Args:
        env_config: Environment configuration
        reward_config: Reward shaping configuration
        episode_length: Max steps per episode
    
    Returns:
        Wrapped gym environment
    """
    mdp = build_environment_from_config(env_config)
    
    env = PerishableInventoryGymWrapper(
        mdp=mdp,
        reward_config=reward_config or RewardConfig()
    )
    
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = Monitor(env)
    
    return env


def make_curriculum_env_factory(
    suite: EnvironmentSuite,
    n_envs: int,
    episode_length: int,
    seed: int
) -> Callable[[str], SubprocVecEnv]:
    """Create factory function for curriculum environments.
    
    Args:
        suite: Environment suite
        n_envs: Number of parallel environments
        episode_length: Max steps per episode
        seed: Random seed
    
    Returns:
        Function that creates VecEnv for given complexity
    """
    def env_factory(complexity: str) -> SubprocVecEnv:
        configs = suite.get_by_complexity(complexity)
        
        if not configs:
            raise ValueError(f"No environments for complexity: {complexity}")
        
        # Sample n_envs configs (with replacement if needed)
        rng = np.random.RandomState(seed)
        selected_configs = rng.choice(configs, size=min(n_envs, len(configs)), replace=False).tolist()
        
        # Pad if needed
        while len(selected_configs) < n_envs:
            selected_configs.append(rng.choice(configs))
        
        def make_env(env_config):
            def _init():
                return create_env_from_config(env_config, episode_length=episode_length)
            return _init
        
        env_fns = [make_env(cfg) for cfg in selected_configs]
        
        if n_envs > 1:
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)
    
    return env_factory


def create_eval_env(
    suite: EnvironmentSuite,
    complexity: str = "simple",
    episode_length: int = 500
) -> DummyVecEnv:
    """Create evaluation environment.
    
    Args:
        suite: Environment suite
        complexity: Complexity level
        episode_length: Max episode length
    
    Returns:
        Vectorized evaluation environment
    """
    configs = suite.get_by_complexity(complexity)
    if not configs:
        configs = list(suite.configs)[:1]
    
    config = configs[0]
    
    def _init():
        return create_env_from_config(config, episode_length=episode_length)
    
    return DummyVecEnv([_init])


def train(args: argparse.Namespace) -> PPO:
    """Run enhanced training pipeline.
    
    Args:
        args: Command line arguments
    
    Returns:
        Trained PPO model
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.test_mode:
        config["total_timesteps"] = 10_000
        config["eval_freq"] = 2_000
        config["checkpoint_freq"] = 5_000
        config["benchmark_freq"] = 5_000
        config["n_envs"] = 2
    
    print("=" * 60)
    print("Enhanced RL Training")
    print("=" * 60)
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Learning rate: {config['initial_learning_rate']} -> {config['final_learning_rate']}")
    print(f"Entropy coef: {config['initial_entropy_coef']} -> {config['final_entropy_coef']}")
    print(f"Curriculum: {config['curriculum_enabled']}")
    print(f"Parallel envs: {config['n_envs']}")
    print("=" * 60)
    
    # Create environment suite
    suite = create_environment_suite(seed=config["seed"])
    print(f"Environment suite created: {len(suite)} environments")
    print(f"  - Simple: {len(suite.get_by_complexity('simple'))}")
    print(f"  - Moderate: {len(suite.get_by_complexity('moderate'))}")
    print(f"  - Complex: {len(suite.get_by_complexity('complex'))}")
    print(f"  - Extreme: {len(suite.get_by_complexity('extreme'))}")
    
    # Create environment factory for curriculum
    env_factory = make_curriculum_env_factory(
        suite=suite,
        n_envs=config["n_envs"],
        episode_length=config["episode_length"],
        seed=config["seed"]
    )
    
    # Start with simple environments
    initial_complexity = "simple"
    train_env = env_factory(initial_complexity)
    print(f"\nStarting with {initial_complexity} environments")
    
    # Create evaluation environment
    eval_env = create_eval_env(suite, "simple", config["episode_length"])
    
    # Setup output directories
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)
    (log_dir / "best_model").mkdir(exist_ok=True)
    (log_dir / "benchmark").mkdir(exist_ok=True)
    
    # Create learning rate schedule
    lr_schedule = create_lr_schedule(
        config["initial_learning_rate"],
        config["final_learning_rate"]
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr_schedule,
        ent_coef=config["initial_entropy_coef"],  # Will be managed by callback
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=config["seed"],
        **{"policy_kwargs": config.get("policy_kwargs", {})}
    )
    
    print(f"\nModel created: {model.policy}")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
    # Setup callbacks
    callbacks = []
    
    # 1. Schedule callback for logging LR/entropy
    schedule_callback = ScheduleCallback(
        initial_lr=config["initial_learning_rate"],
        final_lr=config["final_learning_rate"],
        initial_ent_coef=config["initial_entropy_coef"],
        final_ent_coef=config["final_entropy_coef"],
        log_freq=1000,
        verbose=1 if args.verbose else 0
    )
    callbacks.append(schedule_callback)
    
    # 2. Curriculum callback (if enabled)
    if config["curriculum_enabled"]:
        curriculum_callback = CurriculumCallback(
            env_factory=env_factory,
            thresholds=config.get("curriculum_thresholds", {}),
            window_size=100,
            min_episodes_per_level=50 if not args.test_mode else 5,
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    # 3. Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=config["eval_freq"] // config["n_envs"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # 4. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"] // config["n_envs"],
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_perishable"
    )
    callbacks.append(checkpoint_callback)
    
    # 5. Benchmark callback (compare with TBS)
    try:
        tbs_policy = get_tbs_policy_for_env(eval_env)
        baseline_policies = {"TBS": tbs_policy}
        
        benchmark_callback = BenchmarkCallback(
            eval_env=eval_env,
            benchmark_freq=config["benchmark_freq"] // config["n_envs"],
            n_eval_episodes=config["n_eval_episodes"],
            baseline_policies=baseline_policies,
            save_path=str(log_dir / "benchmark"),
            verbose=1
        )
        callbacks.append(benchmark_callback)
        print("Benchmark callback enabled with TBS baseline")
    except Exception as e:
        print(f"Warning: Could not create TBS baseline: {e}")
    
    callback_list = CallbackList(callbacks)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=True
    )
    
    # Save final model
    model_path = log_dir / "final_model"
    model.save(str(model_path))
    print(f"\nFinal model saved to: {model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Enhanced RL Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--test-mode", action="store_true", help="Quick test run with fewer timesteps")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        model = train(args)
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
