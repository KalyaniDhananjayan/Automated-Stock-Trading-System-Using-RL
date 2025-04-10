import os
import pandas as pd
from data_preprocessing import download_stock_data, preprocess_data, load_guaranteed_numeric_data
from enhanced_trading_env import EnhancedStockTradingEnv
from training import train_model, evaluate_model
from evaluation import test_environment, analyze_performance, plot_performance

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data_path = "data/preprocessed_stock_data.csv"
    verified_data_path = "data/verified_numeric_only.csv"

    if not os.path.exists(data_path):
        print("📥 Downloading stock data...")
        raw_data = download_stock_data(ticker="TSLA")
        preprocess_data(raw_data, save_path=data_path)

    print("📊 Loading and verifying data...")
    try:
        df = load_guaranteed_numeric_data(data_path, verified_data_path)
        print("✅ Data loaded successfully.")
        print(df.head())
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    train_env = EnhancedStockTradingEnv(train_df)
    test_env = EnhancedStockTradingEnv(test_df)

    print("\n🧪 Initial test environment run...")
    _ = test_environment(train_env, n_steps=10)

    print("\n🚀 Training model...")
    model = train_model(train_env, total_timesteps=50000, save_path="models/ppo_stock_trader_best")

    print("\n🧪 Evaluating model...")
    test_env_eval = EnhancedStockTradingEnv(test_df, initial_balance=10000)
    _ = evaluate_model(model, test_env_eval)

    print("\n🧪 Final test run...")
    test_env_live = EnhancedStockTradingEnv(test_df, initial_balance=10000)
    _ = test_environment(test_env_live, n_steps=200)

    print("\n📈 Analyzing performance...")
    results = analyze_performance(test_env_live)
    for k, v in results.items():
        print(f"{k:>20}: {v if not isinstance(v, float) else f'{v:.4f}'}")

    print("\n📉 Plotting performance...")
    plot_performance(test_env_live)

    print("\n✅ Project completed successfully!")

if __name__ == "__main__":
    main()
