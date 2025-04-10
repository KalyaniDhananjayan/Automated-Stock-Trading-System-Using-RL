
# Stock Trading Automation using Reinforcement Learning

A fully functional and modular Reinforcement Learning project designed to automate stock trading using state-of-the-art algorithms like PPO. This project integrates technical indicators, handles real market data, and provides a dashboard for visualization and monitoring.

---

## Project Description

This project leverages the power of Reinforcement Learning to build an automated stock trading agent. It uses historical stock data, computes key technical indicators, builds a custom OpenAI Gym environment, trains using the PPO algorithm from Stable-Baselines3, and visualizes performance using a Streamlit dashboard.

---

## 📂 Project Structure

```
StockRL_Project/
│
├── best_models/                 # Stores best performing models
├── checkpoints/                # Periodic model checkpoints
├── data/                       # Raw and preprocessed stock data
├── logs/                       # TensorBoard training logs
├── models/                     # Trained RL models
├── myenv/                      # Python environment folder (optional)
├── results/                    # Evaluation metrics and CSVs
├── rl_env/                     # Custom environment scripts (if separated)
├── src/                        # All main source code
│   ├── data_preprocessing.py   # Fetches and processes data, adds indicators
│   ├── enhanced_trading_env.py # Custom Gym environment (EnhancedStockTradingEnv)
│   ├── evaluation.py           # Evaluates trained models with performance metrics
│   ├── main.py                 # Main execution pipeline (train + eval + plot)
│   ├── training.py             # Training script with PPO & SB3 callbacks
│   └── dashboard.py            # Streamlit dashboard to visualize results
│
├── stockRL_env/                # (Optional) Virtualenv or Conda env
├── tensorboard/                # TensorBoard logs for live training feedback
├── requirements.txt            # Required packages
```

---

## 📊 Features

- 📉 **Technical Indicators**: SMA, RSI, MACD, CCI, OBV, ADX, EMA, Williams %R, and more.
- 💡 **Custom Gym Environment**: Tailored trading logic, reward functions, and observation space.
- ⚙️ **Training Engine**: PPO algorithm from Stable-Baselines3 with support for checkpoints and tensorboard logging.
- 📈 **Performance Evaluation**: Metrics like Sharpe Ratio, Max Drawdown, Win Rate, and Total Return.
- 🧼 **Robust Preprocessing**: Handles NaNs, divide-by-zero, non-numeric data, and normalization.
- 🌐 **Streamlit Dashboard**: Interactive interface to view performance and retrain the model.

---

## 🛠 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-rl-automation.git
cd stock-rl-automation

# 2. Create virtual environment (optional but recommended)
python -m venv stockRL_env
source stockRL_env/bin/activate    # On Windows: stockRL_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ✅ Usage

1. **Preprocess Data**

   ```bash
   python src/data_preprocessing.py
   ```

2. **Train the PPO Model**

   ```bash
   python src/training.py
   ```

3. **Evaluate the Model**

   ```bash
   python src/evaluation.py
   ```

4. **Launch the Dashboard**

   ```bash
   streamlit run src/dashboard.py
   ```

---

## 📈 Performance Metrics

| Metric           | Description                                  |
| ---------------- | -------------------------------------------- |
| **Total Return** | Overall % gain/loss from the agent's trades. |
| **Sharpe Ratio** | Risk-adjusted return — higher is better.     |
| **Max Drawdown** | Biggest observed loss from peak value.       |
| **Win Rate**     | % of trades that were profitable.            |

---

## 📦 Requirements

Read the `requirements.txt`
---

## Acknowledgements

- Built with guidance from teachers and real-time stock market data.
- Inspired by various reinforcement learning finance papers.
- Libraries used: `ta`, `yfinance`, `Stable-Baselines3`, `Streamlit`, `PyTorch`.

---

