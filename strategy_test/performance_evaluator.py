import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

def load_results(results_dir, strategy_name):
    portfolio_value = pd.read_csv(
        os.path.join(results_dir, f"{strategy_name}_portfolio_value.csv"),
        index_col=0, parse_dates=True
    ).squeeze("columns")
    weights = pd.read_csv(
        os.path.join(results_dir, f"{strategy_name}_weights.csv"),
        index_col=0, parse_dates=True
    )
    asset_returns = pd.read_csv(
        os.path.join(results_dir, f"{strategy_name}_asset_returns.csv"),
        index_col=0, parse_dates=True
    )
    return portfolio_value, weights, asset_returns

def compute_performance_metrics(portfolio_value, asset_returns, risk_free_rate=0.0, benchmark=None):
    returns = portfolio_value.pct_change().dropna()
    n_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25

    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (returns.mean() - risk_free_rate / 252) / downside_std if downside_std > 0 else np.nan
    roll_max = portfolio_value.cummax()
    drawdown = (portfolio_value - roll_max) / roll_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    hit_rate = (returns > 0).mean()

    bench_metrics = {}
    if benchmark is not None:
        bench_returns = benchmark.pct_change().dropna()
        common_idx = returns.index.intersection(bench_returns.index)
        returns_aligned = returns.loc[common_idx]
        bench_returns_aligned = bench_returns.loc[common_idx]

        bench_cagr = (benchmark.loc[common_idx].iloc[-1] / benchmark.loc[common_idx].iloc[0]) ** (1 / n_years) - 1
        bench_vol = bench_returns_aligned.std() * np.sqrt(252)
        bench_sharpe = (bench_returns_aligned.mean() - risk_free_rate / 252) / bench_returns_aligned.std() * np.sqrt(252)
        beta = np.cov(returns_aligned, bench_returns_aligned)[0, 1] / np.var(bench_returns_aligned)
        tracking_error = np.std(returns_aligned - bench_returns_aligned) * np.sqrt(252)
        excess_return = cagr - bench_cagr
        bench_metrics = {
            "benchmark_cagr": bench_cagr,
            "benchmark_volatility": bench_vol,
            "benchmark_sharpe": bench_sharpe,
            "beta": beta,
            "tracking_error": tracking_error,
            "excess_return": excess_return
        }

    metrics = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Hit Rate": hit_rate
    }
    metrics.update(bench_metrics)
    return metrics, drawdown, returns

def plot_performance(portfolio_value, drawdown, returns, weights, asset_returns, benchmark=None, strategy_name=None):
    first_valid = portfolio_value.first_valid_index()
    pv_valid = portfolio_value.loc[first_valid:]
    weights_valid = weights.loc[first_valid:]
    logger.info(f"Plotting from {first_valid}")

    plt.figure(figsize=(12, 6))
    plt.plot(pv_valid, label=f"{strategy_name} (Portfolio Value)")
    if benchmark is not None:
        # Plot the full benchmark series, scaled to initial capital at portfolio start
        bench_plot = benchmark.copy()
        if pv_valid.index[0] in bench_plot.index:
            scale = pv_valid.iloc[0] / bench_plot.loc[pv_valid.index[0]]
            bench_plot = bench_plot * scale
        plt.plot(bench_plot, label="Benchmark (Full Period)", linestyle="--")
    plt.yscale("log")
    plt.title(f"Portfolio Value vs. Benchmark (Log Y Scale) - {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Pie chart of final weights
    logger.info("Plotting pie chart of final portfolio weights.")
    final_weights = weights_valid.iloc[-1].round(2)
    nonzero_weights = final_weights[final_weights > 0]
    plt.figure(figsize=(8, 8))
    plt.pie(nonzero_weights, labels=nonzero_weights.index, autopct="%.2f%%", startangle=90, counterclock=False)
    plt.title(f"Final Portfolio Weights - {strategy_name}")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting drawdown curve.")
    drawdown_valid = drawdown.loc[first_valid:]
    plt.figure(figsize=(12, 4))
    plt.plot(drawdown_valid, color="red")
    plt.title(f"Drawdown Over Time - {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting rolling risk metrics.")
    returns_valid = returns.loc[first_valid:]
    rolling_window = 63
    rolling_vol = returns_valid.rolling(rolling_window).std() * np.sqrt(252)
    rolling_sharpe = (returns_valid.rolling(rolling_window).mean()) / returns_valid.rolling(rolling_window).std() * np.sqrt(252)
    downside = returns_valid.copy()
    downside[downside > 0] = 0
    rolling_sortino = (returns_valid.rolling(rolling_window).mean()) / downside.rolling(rolling_window).std() * np.sqrt(252)

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_vol, label="Rolling Volatility")
    plt.plot(rolling_sharpe, label="Rolling Sharpe")
    plt.plot(rolling_sortino, label="Rolling Sortino")
    plt.title(f"Rolling Risk Metrics (63-day window) - {strategy_name}")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

    logger.info("Plotting histogram of returns.")
    plt.figure(figsize=(8, 4))
    sns.histplot(returns_valid, bins=100, kde=True)
    plt.title(f"Distribution of Daily Returns - {strategy_name}")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.show()

    logger.info("Plotting asset allocation over time.")
    plt.figure(figsize=(12, 6))
    weights_valid.plot.area(ax=plt.gca(), cmap="tab20", stacked=True)
    plt.title(f"Asset Allocation Over Time - {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    plt.tight_layout()
    plt.show()

def main():
    logger.info("Starting performance evaluation.")
    config_path = os.path.join(os.path.dirname(__file__), "test_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    portfolios = {}
    metrics_dict = {}

    for strategy_cfg in config["strategies"]:
        strategy_name = strategy_cfg["name"]
        benchmark_ticker = strategy_cfg.get("benchmark", None)
        risk_free_rate = strategy_cfg.get("risk_free_rate", 0.0)
        start = strategy_cfg["start"]
        end = strategy_cfg["end"]
        if isinstance(end, str) and end.lower() == "today":
            end = datetime.today().strftime("%Y-%m-%d")

        portfolio_value, weights, asset_returns = load_results(results_dir, strategy_name)
        logger.info(f"Loaded backtest results for {strategy_name}.")

        # Load total transaction and slippage costs
        cost_path = os.path.join(results_dir, f"{strategy_name}_costs.txt")
        total_transaction_cost = None
        total_slippage_cost = None
        if os.path.exists(cost_path):
            with open(cost_path, "r") as f:
                for line in f:
                    if line.startswith("total_transaction_cost"):
                        total_transaction_cost = float(line.strip().split(",")[1])
                    elif line.startswith("total_slippage_cost"):
                        total_slippage_cost = float(line.strip().split(",")[1])

        # Load or fetch benchmark
        benchmark = None
        if benchmark_ticker:
            benchmark_path = os.path.join(results_dir, f"{benchmark_ticker}_benchmark.csv")
            if os.path.exists(benchmark_path):
                benchmark = pd.read_csv(benchmark_path, index_col=0, parse_dates=True).squeeze("columns")
            else:
                import yfinance as yf
                logger.info(f"Downloading benchmark data for {benchmark_ticker}.")
                bench_df = yf.download(benchmark_ticker, start=start, end=end, progress=False)
                if "Close" in bench_df.columns:
                    benchmark = bench_df["Close"]
                elif "Adj Close" in bench_df.columns:
                    benchmark = bench_df["Adj Close"]
                benchmark = benchmark.reindex(portfolio_value.index).ffill()
                benchmark.to_csv(benchmark_path, header=True)
            benchmark = benchmark.reindex(portfolio_value.index).ffill()
            capital = strategy_cfg["capital"]["amount"]
            if benchmark.first_valid_index() is not None:
                initial_bench = benchmark.loc[benchmark.first_valid_index()]
                benchmark = benchmark * (capital / initial_bench)

        metrics, drawdown, returns = compute_performance_metrics(
            portfolio_value, asset_returns, risk_free_rate, benchmark
        )
        metrics_dict[strategy_name] = metrics
        portfolios[strategy_name] = portfolio_value

        # Load gross (before costs) portfolio value and compute gross CAGR
        gross_path = os.path.join(results_dir, f"{strategy_name}_portfolio_value_gross.csv")
        gross_cagr = None
        net_cagr = None
        cagr_diff = None
        if os.path.exists(gross_path):
            portfolio_value_gross = pd.read_csv(gross_path, index_col=0, parse_dates=True).squeeze("columns")
            gross_returns = portfolio_value_gross.pct_change().dropna()
            n_years_gross = (portfolio_value_gross.index[-1] - portfolio_value_gross.index[0]).days / 365.25
            gross_cagr = (portfolio_value_gross.iloc[-1] / portfolio_value_gross.iloc[0]) ** (1 / n_years_gross) - 1
            # Net CAGR (after costs)
            n_years_net = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
            net_cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / n_years_net) - 1
            cagr_diff = gross_cagr - net_cagr
            metrics["Gross CAGR (before costs)"] = gross_cagr
            metrics["Net CAGR (after costs)"] = net_cagr
            metrics["CAGR Difference (cost impact)"] = cagr_diff
        if "CAGR" in metrics:
            # Remove 'CAGR' from metrics to avoid duplicate/confusing output
            del metrics["CAGR"]
        logger.info(f"Performance Metrics for {strategy_name}:")
        for k, v in metrics.items():
            if k in ["Gross CAGR (before costs)", "Net CAGR (after costs)", "CAGR Difference (cost impact)", "Volatility", "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Hit Rate", "benchmark_cagr", "benchmark_volatility", "excess_return"]:
                logger.info(f"{k}: {v:.4%}")
            else:
                logger.info(f"{k}: {v:.4f}")

        # Plot all performance charts for this strategy
        plot_performance(portfolio_value, drawdown, returns, weights, asset_returns, benchmark, strategy_name)

    # Plot all strategies together for comparison
    plt.figure(figsize=(12, 6))
    for name, pv in portfolios.items():
        first_valid = pv.first_valid_index()
        pv_valid = pv.loc[first_valid:]
        plt.plot(pv_valid, label=name)
    plt.yscale("log")
    plt.title("Portfolio Value Comparison (Log Y Scale)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()