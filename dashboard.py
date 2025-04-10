
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

from src.data_preprocessing import load_guaranteed_numeric_data
from src.enhanced_trading_env import EnhancedStockTradingEnv
from src.training import train_model
from src.evaluation import analyze_performance, test_environment

# Set page configuration
st.set_page_config(
    page_title="RL Stock Trader Dashboard",
    page_icon="📊",
    layout="wide"
)

# Create required folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
@st.cache_data
def load_data():
    DATA_PATH = "data/preprocessed_stock_data.csv"
    VERIFIED_PATH = "data/verified_numeric_only.csv"
    df = load_guaranteed_numeric_data(DATA_PATH, VERIFIED_PATH)
    return df

df = load_data()
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

st.title("📊 RL Stock Trader Dashboard")
st.markdown("View model performance, equity curve, indicators, and optionally train new models.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Model Evaluation", "Technical Indicators", "Train New Model"])

def plot_indicators(dataframe, indicators, window_start=0, window_end=None):
    if window_end is None:
        window_end = len(dataframe)
    
    df_window = dataframe.iloc[window_start:window_end].copy()
    
    # First, let's determine what our price column is named
    # Common names for price columns in financial datasets
    possible_price_columns = ['close', 'Close', 'price', 'Price', 'adj_close', 'Adj Close', 'closing_price']
    
    price_col = None
    for col in possible_price_columns:
        if col in df_window.columns:
            price_col = col
            break
    
    # If none of the common names are found, use the first numeric column
    if price_col is None:
        numeric_cols = df_window.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            raise ValueError("Could not identify a suitable price column in the dataframe")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot close price as default
    ax.plot(df_window.index, df_window[price_col], label=f'{price_col}', color='black', alpha=0.7)
    
    # Plot selected indicators
    for indicator in indicators:
        if indicator == 'SMA':
            df_window['SMA'] = df_window[price_col].rolling(window=20).mean()
            ax.plot(df_window.index, df_window['SMA'], label='SMA (20)', color='blue')
        elif indicator == 'EMA':
            df_window['EMA'] = df_window[price_col].ewm(span=20, adjust=False).mean()
            ax.plot(df_window.index, df_window['EMA'], label='EMA (20)', color='red')
        elif indicator == 'Bollinger Bands':
            # Calculate Bollinger Bands
            df_window['SMA'] = df_window[price_col].rolling(window=20).mean()
            df_window['STD'] = df_window[price_col].rolling(window=20).std()
            df_window['Upper'] = df_window['SMA'] + (df_window['STD'] * 2)
            df_window['Lower'] = df_window['SMA'] - (df_window['STD'] * 2)
            
            ax.plot(df_window.index, df_window['Upper'], label='Upper Band', color='green', linestyle='--')
            ax.plot(df_window.index, df_window['SMA'], label='Middle Band', color='blue', linestyle='-')
            ax.plot(df_window.index, df_window['Lower'], label='Lower Band', color='red', linestyle='--')
        elif indicator == 'RSI':
            # RSI calculation needs to be on a secondary axis
            delta = df_window[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            RS = gain / loss
            df_window['RSI'] = 100 - (100 / (1 + RS))
            
            ax2 = ax.twinx()
            ax2.plot(df_window.index, df_window['RSI'], label='RSI (14)', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            ax2.legend(loc='upper right')
        elif indicator == 'MACD':
            # Calculate MACD
            df_window['EMA12'] = df_window[price_col].ewm(span=12, adjust=False).mean()
            df_window['EMA26'] = df_window[price_col].ewm(span=26, adjust=False).mean()
            df_window['MACD'] = df_window['EMA12'] - df_window['EMA26']
            df_window['Signal'] = df_window['MACD'].ewm(span=9, adjust=False).mean()
            
            # Create a secondary axis for MACD
            ax2 = ax.twinx()
            ax2.plot(df_window.index, df_window['MACD'], label='MACD', color='blue')
            ax2.plot(df_window.index, df_window['Signal'], label='Signal', color='red')
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper right')
    
    ax.set_title("Price and Selected Indicators")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    return fig


# Page: Model Evaluation
if page == "Model Evaluation":
    st.header("📥 Load Existing Model")
    
    model_files = [f for f in os.listdir("models") if f.endswith(".zip")]
    
    if not model_files:
        st.warning("No models found in the 'models' directory. Please train a model first.")
    else:
        selected_model_file = st.selectbox("Choose a model to evaluate:", model_files)
        load_model_btn = st.button("Load and Evaluate Model")
        
        if load_model_btn:
            try:
                with st.spinner("Loading and evaluating model..."):
                    model_path = os.path.join("models", selected_model_file)
                    model = PPO.load(model_path)
                    test_env = EnhancedStockTradingEnv(test_df.copy(), initial_balance=10000)
                    test_environment(test_env, model=model, n_steps=200)
                    results = analyze_performance(test_env)
                    
                    st.success(f"Model `{selected_model_file}` loaded successfully!")
                    
                    # Create two columns for visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Equity Curve")
                        fig1, ax1 = plt.subplots(figsize=(10, 4))
                        ax1.plot(test_env.equity_curve, label='Net Worth', color='blue')
                        ax1.set_title("Equity Curve")
                        ax1.set_xlabel("Steps")
                        ax1.set_ylabel("Net Worth")
                        ax1.grid(True)
                        ax1.legend()
                        st.pyplot(fig1)
                    
                    with col2:
                        st.subheader("📊 Performance Metrics")
                        
                        # Format the results nicely
                        metrics_df = pd.DataFrame({
                            'Metric': list(results.keys()),
                            'Value': list(results.values())
                        })
                        
                        st.table(metrics_df)
                        
                    # Add a section for trade history
                    if hasattr(test_env, 'trade_history') and test_env.trade_history:
                        st.subheader("💹 Trade History")
                        trade_df = pd.DataFrame(test_env.trade_history)
                        st.dataframe(trade_df)
                
            except Exception as e:
                st.error(f"🚨 Failed to load and evaluate model: {e}")

# Page: Technical Indicators
elif page == "Technical Indicators":
    st.header("📉 Technical Indicators Analysis")
    
    # Select which part of the data to visualize
    data_section = st.radio("Select data section to analyze:", ["Training Data"])
    df_to_use = train_df 
    
    # Configure visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        window_size = st.slider("Select data window size:", 
                              min_value=50, 
                              max_value=min(500, len(df_to_use)), 
                              value=200)
        
        window_start = st.slider("Window start position:", 
                               min_value=0, 
                               max_value=max(0, len(df_to_use) - window_size), 
                               value=0)
    
    with col2:
        available_indicators = ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD"]
        selected_indicators = st.multiselect("Select indicators to display:", 
                                          available_indicators, 
                                          default=["SMA", "Bollinger Bands"])
    
    # Plot the indicators
    if selected_indicators:
        fig = plot_indicators(df_to_use, selected_indicators, window_start, window_start + window_size)
        st.pyplot(fig)
    else:
        st.info("Please select at least one indicator to display.")
    
    # Additional information about indicators
    with st.expander("Learn about Technical Indicators"):
        st.markdown("""
        ### Technical Indicators
        
        - **SMA (Simple Moving Average)**: Calculates the average of a selected range of prices over a specified number of periods.
        - **EMA (Exponential Moving Average)**: Gives more weight to recent prices in its calculation, making it more responsive to new information.
        - **Bollinger Bands**: Consists of a middle band (SMA) with upper and lower bands that represent standard deviations from the middle band.
        - **RSI (Relative Strength Index)**: Measures the speed and change of price movements on a scale of 0 to 100, identifying overbought (>70) or oversold (<30) conditions.
        - **MACD (Moving Average Convergence Divergence)**: Shows the relationship between two EMAs of a price, helping to identify momentum changes.
        """)

# Page: Train New Model
elif page == "Train New Model":
    st.header("🛠️ Train New PPO Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        timesteps = st.number_input("Training Timesteps", min_value=1000, value=20000, step=1000)
        learning_rate = st.select_slider("Learning Rate", 
                                       options=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01],
                                       value=0.0003)
    
    with col2:
        n_steps = st.number_input("Steps per Update", min_value=32, value=2048, step=32)
        model_name = st.text_input("Model Name (optional)", 
                                 value=f"ppo_model_{datetime.now().strftime('%Y%m%d')}")
    
    do_train = st.button("Start Training")
    
    if do_train:
        with st.spinner("Training model... This may take a while."):
            st.info("Training model...")
            train_env = EnhancedStockTradingEnv(train_df.copy())
            
            # Auto-generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.zip" if model_name else f"ppo_model_{timestamp}.zip"
            save_path = os.path.join("models", model_filename)
            
            # Train with custom parameters
# Train with parameters that match your function definition
            model = train_model(
                train_env, 
                total_timesteps=timesteps,
                save_path=save_path
            )
            st.success(f"✅ Model trained and saved as `{model_filename}`")
            
            with st.spinner("Evaluating newly trained model..."):
                st.info("Evaluating newly trained model...")
                test_env = EnhancedStockTradingEnv(test_df.copy(), initial_balance=10000)
                test_environment(test_env, model=model, n_steps=200)
                results = analyze_performance(test_env)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Equity Curve (New Model)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(test_env.equity_curve, label='Net Worth', color='green')
                    ax2.set_title("Equity Curve - New Model")
                    ax2.set_xlabel("Steps")
                    ax2.set_ylabel("Net Worth")
                    ax2.grid(True)
                    ax2.legend()
                    st.pyplot(fig2)
                
                with col2:
                    st.subheader("📊 Performance Metrics (New Model)")
                    metrics_df = pd.DataFrame({
                        'Metric': list(results.keys()),
                        'Value': list(results.values())
                    })
                    st.table(metrics_df)

st.markdown("---")
st.caption("Built with ❤️ by Arjun, Arsha, Kalyani, Vivek")