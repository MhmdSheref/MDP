
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

# Add path
sys.path.insert(0, ".")

from colab_training.environment_suite import create_environment_suite
from colab_training.train_rl import create_env_from_config
from colab_training.benchmark import get_tbs_policy_for_env

def _encode_action(env, action_dict):
    """Callback's _encode_action logic - EXACT COPY."""
    action = []
    if not hasattr(env, 'supplier_order'):
         return np.zeros(1)
         
    for sid in env.supplier_order:
        qty = action_dict.get(sid, 0)
        bins = env.supplier_action_bins[sid]
        # Find closest bin
        idx = np.argmin(np.abs(np.array(bins) - qty))
        action.append(idx) # NO int() cast
    return np.array([action])

def main():
    print("Setting up environment...")
    suite = create_environment_suite()
    config = suite.get_by_complexity("simple")[0]
    
    def make_env():
        return create_env_from_config(config, episode_length=500)
    
    eval_env = DummyVecEnv([make_env])
    
    print("Getting TBS policy...")
    tbs_policy = get_tbs_policy_for_env(eval_env)
    
    print("Running evaluation loop...")
    env = eval_env.envs[0]
    obs = eval_env.reset()
    
    # Run loop
    for step_i in range(5):
        try:
            if hasattr(env, 'current_state') and hasattr(env, 'mdp'):
                action_dict = tbs_policy.get_action(env.current_state, env.mdp)
                encoded_action = _encode_action(env, action_dict)
                print(f"Step {step_i}: Action shape: {encoded_action.shape}, Dtype: {encoded_action.dtype}")
                
                result = eval_env.step(encoded_action)
                
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                
                print(f"  Result Done type: {type(done)}")
                if isinstance(done, np.ndarray):
                    print(f"  Done shape: {done.shape}")
                    d = done[0] # Test indexing
                    print(f"  Done[0]: {d}")
                
                # Verify action received in gym_env (we can't easily hook, but if no error, it worked)
                
            else:
                 print("Env structure mismatch")

        except Exception as e:
            print(f"CAUGHT ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()
