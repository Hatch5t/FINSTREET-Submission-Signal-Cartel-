import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

BACKTEST_START = os.getenv("BACKTEST_START", "2022-01-01")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "3.0"))
MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"
OS_END = os.getenv("OS_END", "2023-12-31")
OS_START = os.getenv("OS_START", "2022-07-01")
START_DATE_DATA = os.getenv("START_DATE_DATA", "2021-01-01")
TARGET_VOL = float(os.getenv("TARGET_VOL", "0.02"))
TICKER: str = os.getenv("TICKER", "NSE:SONATSOFTW-EQ")

access_token = None

try:
    token_path = os.path.join(os.getcwd(), "Data", "access_token.txt")
    with open(token_path, "r") as f:
        access_token = f.read().strip()

except FileNotFoundError:
    print("WARNING: Access Token file not found. Run the Notebook first!")


# ==========================================
# 2. DATA FETCHING (FYERS API)
# ==========================================
def fetch_fyers_data(
    client_id: str, symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetches historical data using Fyers API v3 and formats it for the strategy.
    """
    if MOCK_MODE:
        print("[WARNING] Running in MOCK MODE (Fake Data)")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return pd.DataFrame(
            {
                "Open": 100 + np.random.randn(len(dates)),
                "High": 105 + np.random.randn(len(dates)),
                "Low": 95 + np.random.randn(len(dates)),
                "Close": 100 + np.cumsum(np.random.randn(len(dates))),
                "Volume": 10000 + np.random.randint(0, 1000, len(dates)),
            },
            index=dates,
        )

    try:
        if access_token is None:
            raise ValueError("Access Token is not set. Cannot fetch data.")

        fyers = fyersModel.FyersModel(
            client_id=client_id, token=access_token, is_async=False, log_path=""
        )

        data_input = {
            "symbol": symbol,
            "resolution": "D",  # Daily resolution
            "date_format": "1",  # YYYY-MM-DD
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1",
        }

        response: Dict[str, Any] = fyers.history(data=data_input)

        if response.get("s") != "ok":
            raise ValueError(
                f"Fyers API Error: {response.get('message', 'Unknown Error')}"
            )

        # Fyers returns: [epoch, open, high, low, close, volume]
        cols = ["epoch", "Open", "High", "Low", "Close", "Volume"]
        df = pd.DataFrame(response["candles"], columns=cols)

        # Process Dates
        df["Date"] = pd.to_datetime(df["epoch"], unit="s").dt.date
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # Drop epoch and ensure numeric types
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        return df

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return pd.DataFrame()


print(f"--- FETCHING DATA FOR {TICKER} (via FYERS) ---")
# Extend end date slightly to ensure we capture the full OS period
fetch_end = (pd.to_datetime(OS_END) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
df = fetch_fyers_data(CLIENT_ID, TICKER, START_DATE_DATA, fetch_end)

if len(df) == 0:
    raise ValueError(
        "No data fetched. Check your API Keys, Token, or Internet Connection."
    )

# Clean duplicate indices if any
df = df[~df.index.duplicated(keep="first")]

# ==========================================
# 3. ADVANCED FEATURE ENGINEERING (V2)
# ==========================================
data = df.copy()

# A. Standard Features
data["Returns"] = data["Close"].pct_change()
data["Range"] = (data["High"] - data["Low"]) / data["Close"]

# B. Technical Indicators

# 1. RSI
delta = data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# 2. MACD (Momentum)
data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = data["EMA_12"] - data["EMA_26"]
data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_Hist"] = data["MACD"] - data["Signal_Line"]

# 3. Bollinger Bands (Volatility)
data["SMA_20"] = data["Close"].rolling(window=20).mean()
std_20 = data["Close"].rolling(window=20).std()
data["Upper_Band"] = data["SMA_20"] + (std_20 * 2)
data["Lower_Band"] = data["SMA_20"] - (std_20 * 2)
data["Pct_B"] = (data["Close"] - data["Lower_Band"]) / (
    data["Upper_Band"] - data["Lower_Band"]
)

# 4. Volatility (for Risk)
data["Rolling_Vol"] = data["Returns"].rolling(window=20).std() * np.sqrt(252)

# Shift Features (Predicting Tomorrow using Today's data)
features = ["Returns", "Range", "RSI", "MACD_Hist", "Pct_B"]
for feat in features:
    data[f"{feat}_Lag1"] = data[feat].shift(1)

# Target: 1 if Tomorrow's Close > Today's Close
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

feature_cols = [f"{col}_Lag1" for col in features]
X = data[feature_cols]
y = data["Target"]

# ==========================================
# 4. WALK-FORWARD BACKTEST (RANDOM FOREST)
# ==========================================
print(f"--- RUNNING RANDOM FOREST MODEL ({BACKTEST_START} to {OS_END}) ---")

# Filter data to include both Backtest and Out-of-Sample periods
mask_dates = (data.index >= BACKTEST_START) & (data.index <= OS_END)
full_period = data.index[mask_dates]

trade_log = []

for current_date in full_period:
    # Train on all data available BEFORE current_date
    train_mask = data.index < (current_date - pd.Timedelta(days=1))
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_today = X.loc[[current_date]]

    # Need enough history to train
    if len(X_train) < 50:
        continue

    # Random Forest Model
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=5, random_state=42
    )
    model.fit(X_train, y_train)

    # Predict Probability for UP move
    prob_up = model.predict_proba(X_today)[0][1]

    # Threshold Logic
    if prob_up > CONFIDENCE_THRESHOLD:
        action = "LONG"
    elif prob_up < (1.0 - CONFIDENCE_THRESHOLD):
        action = "SHORT"
    else:
        action = "CASH"

    # Position Sizing (Volatility Targeting)
    # Using previous day's volatility to avoid look-ahead bias
    prev_idx: int = data.index.get_loc(current_date) - 1
    if prev_idx >= 0:
        vol_yesterday = data.iloc[prev_idx]["Rolling_Vol"]
    else:
        vol_yesterday = 0

    if vol_yesterday == 0 or np.isnan(vol_yesterday):
        size = 0
    else:
        size = min(TARGET_VOL / vol_yesterday, MAX_LEVERAGE)

    # Calculate PnL
    # Retrieve scalar value safely
    actual_ret_val = data.loc[current_date, "Returns"]
    if isinstance(actual_ret_val, pd.Series):
        actual_ret = float(actual_ret_val.iloc[0])
    else:
        actual_ret = float(actual_ret_val)

    if action == "LONG":
        strategy_ret = size * actual_ret
    elif action == "SHORT":
        strategy_ret = size * (-1 * actual_ret)
    else:
        strategy_ret = 0.0
        size = 0.0

    trade_log.append(
        {
            "Date": current_date,
            "Prob_Up": round(prob_up, 3),
            "Action": action,
            "Size": round(size, 2),
            "Actual_Ret": actual_ret,
            "Strat_Ret": strategy_ret,
        }
    )

# ==========================================
# 5. RESULTS & VISUALIZATION
# ==========================================
results = pd.DataFrame(trade_log)

if results.empty:
    print("No trades generated. Check date ranges or data availability.")
else:
    results.set_index("Date", inplace=True)
    results["Cum_Mkt"] = (1 + results["Actual_Ret"]).cumprod()
    results["Cum_Strat"] = (1 + results["Strat_Ret"]).cumprod()

    total_ret = (results["Cum_Strat"].iloc[-1] - 1) * 100
    mkt_ret = (results["Cum_Mkt"].iloc[-1] - 1) * 100

    active_trades = results[results["Action"] != "CASH"]
    if len(active_trades) > 0:
        win_trades = active_trades[active_trades["Strat_Ret"] > 0]
        win_rate = len(win_trades) / len(active_trades)
    else:
        win_rate = 0.0

    # --- SHARPE CALCULATION (MODIFIED) ---

    # 1. Total Sharpe
    mean_daily_ret = results["Strat_Ret"].mean()
    std_daily_ret = results["Strat_Ret"].std()
    if std_daily_ret != 0:
        sharpe_total = (mean_daily_ret / std_daily_ret) * np.sqrt(252)
    else:
        sharpe_total = 0.0

    # 2. IS / OS Separation
    os_split = pd.Timestamp(OS_START)

    # In-Sample Data
    is_data = results[results.index < os_split]
    if not is_data.empty and is_data["Strat_Ret"].std() != 0:
        sharpe_is = (
            is_data["Strat_Ret"].mean() / is_data["Strat_Ret"].std()
        ) * np.sqrt(252)
    else:
        sharpe_is = 0.0

    # Out-of-Sample Data
    os_data = results[results.index >= os_split]
    if not os_data.empty and os_data["Strat_Ret"].std() != 0:
        sharpe_os = (
            os_data["Strat_Ret"].mean() / os_data["Strat_Ret"].std()
        ) * np.sqrt(252)
    else:
        sharpe_os = 0.0
    # -------------------------------------

    cumulative_returns = results["Cum_Strat"]
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100

    print("\n" + "=" * 50)
    print(" PERFORMANCE REPORT (RANDOM FOREST)")
    print("=" * 50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Total Return (PnL)':<25} | {total_ret:.2f}%")
    print(f"{'Market Return':<25} | {mkt_ret:.2f}%")
    print(f"{'Win Rate':<25} | {win_rate * 100:.2f}%")
    print("-" * 50)
    print(f"{'Sharpe Ratio (Total)':<25} | {sharpe_total:.2f}")
    print(f"{'Sharpe Ratio (IS)':<25} | {sharpe_is:.2f}")
    print(f"{'Sharpe Ratio (OS)':<25} | {sharpe_os:.2f}")
    print("-" * 50)
    print(f"{'Max Drawdown':<25} | {max_drawdown:.2f}%")
    print(f"{'Trades Taken':<25} | {len(active_trades)} / {len(results)}")
    print("=" * 50)

    pd.set_option("display.max_rows", None)
    print("\n" + "=" * 50)
    print("               FULL TRADE LOG               ")
    print("=" * 50)
    print(results[["Action", "Prob_Up", "Size", "Actual_Ret", "Strat_Ret"]])
    print("=" * 50)
    pd.reset_option("display.max_rows")

    # PLOTS
    _, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(
        results.index,
        results["Cum_Mkt"],
        label="Buy & Hold",
        color="gray",
        linestyle="--",
        alpha=0.7,
    )
    axes[0].plot(
        results.index,
        results["Cum_Strat"],
        label="Random Forest",
        color="blue",
        linewidth=2,
    )
    try:
        axes[0].axvline(
            pd.Timestamp(OS_START), color="orange", linestyle=":", label="OS Start"
        )
    except Exception:
        pass
    axes[0].set_title(f"Equity Curve: Random Forest ({TICKER})")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results.index, results["Size"], color="orange", label="Leverage")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Dynamic Position Sizing")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    colors = ["green" if x > 0 else "red" for x in results["Strat_Ret"]]
    axes[2].bar(results.index, results["Strat_Ret"], color=colors, alpha=0.6)
    axes[2].set_title("Daily Strategy PnL")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("strategy_performance.png")
