# ML-Driven Algo Trading Strategy (Kshitij 2026 - FinStreet)

**Team Name:** Signal Cartel  
**Stock Selected:** `NSE:SONATSOFTW-EQ` (Sonata Software Ltd)

## Overview

This project is an end-to-end algorithmic trading system developed for the **Round 2 Submission of FinStreet (Kshitij 2026)**.

The system leverages a **Random Forest Classifier** to predict short-term price movements and executes trades automatically via the **Fyers API**. It features a robust data pipeline, advanced feature engineering, and strict risk management rules (Volatility Targeting).

---

## Setup & Installation

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Install Dependencies

Run the following command to install the required libraries, make sure if you are not keeping the files in the same folder put the whole file address in the command:

```bash
pip install -r requirements.txt
```

### 3. API Configuration

1. Open `Data/config.py`.
2. Add **Client ID, Client Secret, Redirect URI** to `.env` file.
3. The system automatically reads the **Access Token** from `Data/access_token.txt` (which is generated in the next step).

---

## How to Run

### Step 1: Authentication & Data Fetching

Run the Jupyter Notebook located in the `Data/` folder:

1. Open **`Data/FYERS_API_Integration.ipynb`**.
2. Follow the steps to generate your Auth Code.
3. Run the cells to generate the `access_token.txt` and download the stock data CSV.

### Step 2: Run the Strategy

Execute the main script from the root folder to process features, run the backtest, and generate signals:

```bash
python main.py
```

* **Output**: This will print the performance metrics (Sharpe Ratio, Total Return) and save a performance chart as `strategy_performance.png`.

---

## Strategy Logic

### 1. Feature Engineering

We transform raw price data into predictive signals using:

* **RSI (14)**: Measures momentum and overbought/oversold conditions.
* **MACD**: Identifies trend reversals.
* **Bollinger Bands (%B)**: Measures price relative to volatility bands.
* **Volatility (Rolling)**: Used for risk-adjusted position sizing.

### 2. Machine Learning Model

* **Algorithm**: Random Forest Classifier (`n_estimators=100`, `max_depth=5`).
* **Training Method**: Walk-Forward (The model is retrained every day on all past data available up to that point).
* **Target**: Predicts if **Tomorrow's Close > Today's Close**.

### 3. Execution Rules

* **Long Signal**: If Model Probability > 0.55.
* **Short Signal**: If Model Probability < 0.45.
* **Cash (Neutral)**: If Probability is between 0.45 and 0.55.
* **Position Sizing**: Inverse to Volatility (Higher Volatility = Smaller Position).

---

## Performance Metrics (Nov-Dec 2025 Backtest)

* **Sharpe Ratio**: > 1.5 (Target Met)
* **Max Drawdown**: Controlled via Volatility Targeting.
* **Execution**: Validated via Fyers API payload generation.

![metrics](./strategy_performance.png)

---

### Disclaimer

> This project is for educational and hackathon evaluation purposes. All API calls in `execution.py` are set to **Print-Only Mode** to prevent accidental real-money trades during testing.
