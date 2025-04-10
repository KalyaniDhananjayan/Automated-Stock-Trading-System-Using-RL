import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, env, n_steps=None):
    print("🔍 Running model evaluation...")
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        if n_steps and step_count >= n_steps:
            break

    print(f"✅ Evaluation completed in {step_count} steps.")
    return env

def test_environment(env, model=None, n_steps=100):
    obs = env.reset()
    for _ in range(n_steps):
        if model:
            action, _ = model.predict(obs)
        else:
            action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            break
    return env


def analyze_performance(env):
    print("📊 Computing performance metrics...")
    os.makedirs("data", exist_ok=True)  # make sure this is here if you're saving files

    net_worths = np.array(env.equity_curve)

    if len(net_worths) < 2:
        return {
            "Sharpe Ratio": 0,
            "Max Drawdown": 0,
            "Final Net Worth": net_worths[-1] if len(net_worths) else 0
        }

    daily_returns = np.diff(net_worths) / (net_worths[:-1] + 1e-10)
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)
    sharpe_ratio *= np.sqrt(252)

    drawdown = (net_worths - np.maximum.accumulate(net_worths)) / (np.maximum.accumulate(net_worths) + 1e-10)
    max_drawdown = np.min(drawdown)
    final_net_worth = net_worths[-1]

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Final Net Worth": final_net_worth
    }


def plot_performance(env):
    if len(env.equity_curve) < 2:
        print("⚠️ Not enough data to plot performance.")
        return
    
    print("Equity Curve Data:", env.equity_curve[:10])  # First 10 values
    print("Total Points:", len(env.equity_curve))


    plt.figure(figsize=(12, 6))
    plt.plot(env.equity_curve, label="Net Worth")
    plt.title("📈 Equity Curve (Net Worth Over Time)")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/equity_curve.png")
    print("📷 Saved plot to data/equity_curve.png")
    plt.show()
