import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def train_model(env, total_timesteps=20000, save_path="models/ppo_stock_trader"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Wrap env for SB3
    env = DummyVecEnv([lambda: env])

    # Evaluation environment
    eval_env = DummyVecEnv([lambda: env.envs[0]])

    # Evaluation callback to save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path="logs",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    policy_kwargs = dict(net_arch=[256, 256])


    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,               # Slightly lower LR = more stable learning
        n_steps=2048,
        batch_size=128,                   # Larger batch = smoother gradients
        ent_coef=0.001,                   # Lower = less random exploration
        gamma=0.995,                      # Higher = longer-term reward focus
        gae_lambda=0.95,                  # Good default
        clip_range=0.2,
        max_grad_norm=0.5,                # Prevents exploding gradients
        vf_coef=0.4,                      # Value function loss weight
        verbose=1,
    )


    print("\n⚙️ Training started...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    print(f"✅ Final model saved to {save_path}")

    return model

def evaluate_model(model, env, num_episodes=1):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        print(f"Step {step} | Action: {action} | Reward: {reward} | Net Worth: {env.net_worth}")
        step += 1
        total_reward += reward

    print(f"✅ Total Reward: {total_reward}")
