import yaml
import pandas as pd
import numpy as np
import os
from datetime import datetime
from definitions import STRATEGIES
from utils.logger import get_logger
from data.data_cleaning import clean_dataframe

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_price_data(tickers, start, end):
    import yfinance as yf
    data = yf.download([t for t in tickers if t.upper() != "CASH"], start=start, end=end, progress=False)
    if "Close" in data.columns:
        data = data["Close"]
    elif "Adj Close" in data.columns:
        data = data["Adj Close"]
    else:
        raise ValueError("No 'Close' or 'Adj Close' in downloaded data.")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def main():
    logger.info("Starting backtest.")
    config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    config = load_config(config_path)

    cagr_summary = {}
    # --- NEW: Find global benchmark period ---
    all_starts = []
    all_ends = []
    benchmarks_needed = set()
    for strategy_cfg in config["strategies"]:
        all_starts.append(strategy_cfg["start"])
        end = strategy_cfg["end"]
        if isinstance(end, str) and end.lower() == "today":
            end = datetime.today().strftime("%Y-%m-%d")
        all_ends.append(end)
        if "benchmark" in strategy_cfg and strategy_cfg["benchmark"]:
            benchmarks_needed.add(strategy_cfg["benchmark"])
    global_start = min(all_starts)
    global_end = max(all_ends)

    # --- NEW: Download benchmark(s) for full period only once ---
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    import yfinance as yf
    for benchmark in benchmarks_needed:
        benchmark_path = os.path.join(output_dir, f"{benchmark}_benchmark.csv")
        if not os.path.exists(benchmark_path):
            logger.info(f"Downloading benchmark data for {benchmark} from {global_start} to {global_end}.")
            bench_df = yf.download(benchmark, start=global_start, end=global_end, progress=False)
            if "Close" in bench_df.columns:
                bench_series = bench_df["Close"]
            elif "Adj Close" in bench_df.columns:
                bench_series = bench_df["Adj Close"]
            else:
                bench_series = None
            if bench_series is not None:
                bench_series.to_csv(benchmark_path, header=True)

    # --- Main strategy loop ---
    for strategy_cfg in config["strategies"]:
        tickers = strategy_cfg["tickers"]
        capital = strategy_cfg["capital"]["amount"]
        start = strategy_cfg["start"]
        end = strategy_cfg["end"]
        if isinstance(end, str) and end.lower() == "today":
            end = datetime.today().strftime("%Y-%m-%d")
        risk_free_rate = strategy_cfg.get("risk_free_rate", 0.0)
        benchmark = strategy_cfg.get("benchmark", None)

        logger.info(f"Fetching price data for tickers: {tickers} from {start} to {end}")
        prices = fetch_price_data(tickers, start, end)

        # Clean each ticker's data and keep track of tickers with usable data
        cleaned_prices = []
        cleaned_tickers = []
        for col in prices.columns:
            if prices[col].dropna().empty:
                logger.warning(f"No data for ticker {col}, skipping.")
                continue
            df = pd.DataFrame({
                "Date": prices.index,
                "Close": prices[col],
                "High": prices[col],
                "Low": prices[col],
                "Open": prices[col],
                "Volume": 1
            })
            df = clean_dataframe(df)
            df = df.set_index("Date")
            cleaned_prices.append(df["Close"].rename(col))
            cleaned_tickers.append(col)

        # Add synthetic CASH column if requested
        if "CASH" in [t.upper() for t in tickers] and "CASH" not in [c.upper() for c in prices.columns]:
            cash_series = pd.Series(1.0, index=prices.index, name="CASH")
            cleaned_prices.append(cash_series)
            cleaned_tickers.append("CASH")
            logger.info("Added synthetic 'CASH' column to prices.")

        if not cleaned_prices:
            logger.error("No tickers with usable data after cleaning.")
            continue

        prices = pd.concat(cleaned_prices, axis=1)
        logger.info(f"Price data loaded. Shape after cleaning: {prices.shape}")
        logger.info(f"Tickers after cleaning: {cleaned_tickers}")

        if prices.empty or prices.shape[0] < 2:
            logger.error("No price data found after cleaning. Try using fewer tickers or a different date range.")
            continue

        # Set up constraints dynamically based on cleaned tickers
        constraints = {
            "bounds": [(0.0, 1.0)] * len(prices.columns),
            "allow_short": False,
            "weight_sum": True,
            "leverage": None
        }

        # Covariance matrix is not used directly (strategy computes rolling cov)
        cov_matrix = None

        # Get strategy function
        strategy_name = strategy_cfg["name"]
        strategy_func = STRATEGIES[strategy_name]

        logger.info(f"Running strategy: {strategy_name}")
        # Run strategy to get weights over time
        if strategy_name in ["mean_reversion", "region_scaled_mean_reversion"]:
            weights_df = strategy_func(strategy_cfg, prices)
        else:
            weights_df = strategy_func(strategy_cfg, prices, cov_matrix, constraints, risk_free_rate)
            logger.info("Strategy run complete.")

        # Calculate daily portfolio value
        returns = prices.pct_change()
        returns = returns.loc[weights_df.index]  # Align returns with weights_df
        weights_aligned = weights_df.reindex(returns.index).ffill()
        # GROSS portfolio returns (before costs)
        portfolio_returns_gross = (weights_aligned.fillna(0) * returns.fillna(0)).sum(axis=1)
        portfolio_value_gross = (1 + portfolio_returns_gross).cumprod() * capital

        # --- Transaction cost and slippage (per-asset, per-day, for all transactions) ---
        transaction_cost = strategy_cfg.get("transaction_cost", 0.0)  # e.g., 0.001 for 0.1%
        slippage_range = strategy_cfg.get("slippage", [0.0, 0.0])    # e.g., [-0.0002, 0.0002]
        prev_weights = weights_aligned.shift(1).fillna(0)
        turnover = (weights_aligned - prev_weights).abs()  # DataFrame: per-asset, per-day
        # Transaction cost: sum across assets per day, then sum for total
        transaction_costs = turnover * transaction_cost
        total_transaction_cost = (transaction_costs * capital).sum().sum()  # in currency units
        # Slippage: random per asset per day
        np.random.seed(42)  # for reproducibility
        slippage_matrix = np.random.uniform(slippage_range[0], slippage_range[1], size=turnover.shape)
        slippage_costs = turnover * slippage_matrix
        total_slippage_cost = (slippage_costs * capital).sum().sum()  # in currency units
        # Apply to returns: sum across assets per day
        portfolio_returns = portfolio_returns_gross.copy()
        portfolio_returns -= transaction_costs.sum(axis=1)
        portfolio_returns += slippage_costs.sum(axis=1)

        # Save total costs for reporting
        with open(os.path.join(output_dir, f"{strategy_name}_costs.txt"), "w") as f:
            f.write(f"total_transaction_cost,{total_transaction_cost}\n")
            f.write(f"total_slippage_cost,{total_slippage_cost}\n")

        # Apply risk-free rate to CASH position if present
        if "CASH" in weights_aligned.columns:
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            cash_weights = weights_aligned["CASH"].shift(1).fillna(1.0)
            portfolio_returns += cash_weights * daily_rf
            portfolio_returns_gross += cash_weights * daily_rf

        portfolio_value = (1 + portfolio_returns).cumprod() * capital

        # Save results for performance evaluation
        weights_df.to_csv(os.path.join(output_dir, f"{strategy_name}_weights.csv"))
        portfolio_value.to_csv(os.path.join(output_dir, f"{strategy_name}_portfolio_value.csv"), header=["Portfolio Value"])
        portfolio_value_gross.to_csv(os.path.join(output_dir, f"{strategy_name}_portfolio_value_gross.csv"), header=["Portfolio Value Gross"])
        returns.to_csv(os.path.join(output_dir, f"{strategy_name}_asset_returns.csv"))
        # Save cleaned tickers for reference
        with open(os.path.join(output_dir, f"{strategy_name}_cleaned_tickers.txt"), "w") as f:
            f.write("\n".join(cleaned_tickers))

        logger.info(f"Backtest complete for {strategy_name}. Results saved in 'results' folder.")

if __name__ == "__main__":
    main()