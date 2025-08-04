import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import yaml
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_end_date(end):
    if isinstance(end, str) and end.lower() == "today":
        return datetime.date.today().strftime("%Y-%m-%d")
    return end

def fetch_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    # Use "Close" instead of "Adj Close" due to yfinance update
    if "Close" in data.columns:
        data = data["Close"]
    elif "Adj Close" in data.columns:
        data = data["Adj Close"]
    else:
        raise ValueError("Neither 'Close' nor 'Adj Close' found in downloaded data.")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def compute_returns(prices):
    return prices.pct_change().dropna()

def annualize_volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def annualize_sharpe(returns, risk_free_rate=0.0, periods_per_year=252):
    excess = returns - risk_free_rate / periods_per_year
    return (excess.mean() / returns.std()) * np.sqrt(periods_per_year)

def annualize_sortino(returns, risk_free_rate=0.0, periods_per_year=252, ticker=None):
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year)
    excess = returns.mean() - risk_free_rate / periods_per_year
    if downside_std == 0:
        logger.warning(f"Sortino ratio is NaN for {ticker}: downside standard deviation is zero (window too short or no downside moves).")
        return np.nan
    return excess / downside_std

def max_drawdown(prices):
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return drawdown.min()

def rolling_risk_metrics(returns, window=63, risk_free_rate=0.0, ticker=None):
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(window).mean() - risk_free_rate/252) / returns.rolling(window).std() * np.sqrt(252)
    def rolling_sortino(x):
        downside = x[x < 0]
        downside_std = downside.std() * np.sqrt(252)
        excess = x.mean() - risk_free_rate / 252
        if downside_std == 0:
            logger.warning(f"Rolling Sortino ratio is NaN for {ticker}: downside standard deviation is zero (window too short or no downside moves).")
            return np.nan
        return excess / downside_std
    rolling_sortino_ratio = returns.rolling(window).apply(rolling_sortino, raw=False)
    rolling_drawdown = returns.rolling(window).apply(lambda x: max_drawdown((1 + x).cumprod()), raw=False)
    return rolling_vol, rolling_sharpe, rolling_sortino_ratio, rolling_drawdown

def compute_beta_alpha(returns, benchmark_returns, risk_free_rate=0.0):
    cov = np.cov(returns, benchmark_returns)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
    alpha = returns.mean() - beta * benchmark_returns.mean() - risk_free_rate / 252
    return beta, alpha

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "portfolio.yaml")
    config = load_config(config_path)

    tickers = config["tickers"]
    start = config["date_range"]["start"]
    end = get_end_date(config["date_range"]["end"])
    risk_free_rate = config.get("objective", {}).get("risk_free_rate", 0.0)
    risk_metrics = config.get("risk_metrics", [])

    # Set output directory and paths
    output_dir = os.path.join(os.path.dirname(__file__), "risk_metrics_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "risk_metrics_output.csv")
    cov_path = os.path.join(output_dir, "risk_metrics_output_covariance.csv")
    prices_path = os.path.join(output_dir, "risk_metrics_output_prices.csv")

    # Use benchmark from config if present, else fallback
    benchmark_ticker = config.get("benchmark", "SPY")
    if benchmark_ticker not in tickers:
        logger.warning(f"Benchmark ticker {benchmark_ticker} not in tickers list, using first ticker as benchmark.")
        benchmark_ticker = tickers[0]

    logger.info(f"Fetching price data for tickers: {tickers} from {start} to {end}")
    prices = fetch_price_data(tickers, start, end)
    returns = compute_returns(prices)

    results = []

    # Use benchmark for beta/alpha
    benchmark_returns = returns[benchmark_ticker]

    for ticker in tickers:
        ticker_data = {}
        ticker_data["ticker"] = ticker
        ticker_returns = returns[ticker]
        ticker_prices = prices[ticker]

        if "volatility" in risk_metrics:
            ticker_data["volatility"] = annualize_volatility(ticker_returns)
        if "sharpe_ratio" in risk_metrics:
            ticker_data["sharpe_ratio"] = annualize_sharpe(ticker_returns, risk_free_rate)
        if "max_drawdown" in risk_metrics:
            ticker_data["max_drawdown"] = max_drawdown(ticker_prices)
        if "sortino_ratio" in risk_metrics:
            ticker_data["sortino_ratio"] = annualize_sortino(ticker_returns, risk_free_rate, ticker=ticker)
        if "beta" in risk_metrics or "alpha" in risk_metrics:
            beta, alpha = compute_beta_alpha(ticker_returns, benchmark_returns, risk_free_rate)
            if "beta" in risk_metrics:
                ticker_data["beta"] = beta
            if "alpha" in risk_metrics:
                ticker_data["alpha"] = alpha

        results.append(ticker_data)

    # Covariance matrix
    if "covariance" in config:
        cov_method = config["covariance"].get("method", "sample")
        if cov_method == "sample":
            cov_matrix = returns.cov()
        elif cov_method == "ledoit-wolf":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        elif cov_method == "ewma":
            lam = config["covariance"].get("ewma_lambda", 0.94)
            cov_matrix = returns.ewm(alpha=1-lam).cov().iloc[-len(tickers):, :]
        else:
            cov_matrix = returns.cov()
    else:
        cov_matrix = returns.cov()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Risk metrics saved to {output_path}")

    # Save covariance matrix
    cov_matrix.to_csv(cov_path)
    logger.info(f"Covariance matrix saved to {cov_path}")

    # Save prices 
    prices.to_csv(prices_path)  
    logger.info(f"Prices saved to {prices_path}")  

    # Rolling metrics
    rolling_cfg = config.get("rolling_metrics", {})
    if rolling_cfg.get("enabled", False):
        window = rolling_cfg.get("window", 63)
        rolling_output_path = os.path.join(output_dir, "risk_metrics_rolling.csv")
        logger.info(f"Calculating rolling risk metrics with window {window}")
        rolling_results = {}
        for ticker in tickers:
            ticker_returns = returns[ticker]
            rolling_vol, rolling_sharpe, rolling_sortino, rolling_drawdown = rolling_risk_metrics(
                ticker_returns, window=window, risk_free_rate=risk_free_rate, ticker=ticker
            )
            rolling_results[f"{ticker}_rolling_volatility"] = rolling_vol
            rolling_results[f"{ticker}_rolling_sharpe"] = rolling_sharpe
            rolling_results[f"{ticker}_rolling_sortino"] = rolling_sortino
            rolling_results[f"{ticker}_rolling_drawdown"] = rolling_drawdown

        rolling_df = pd.DataFrame(rolling_results, index=returns.index)
        rolling_df.to_csv(rolling_output_path)
        logger.info(f"Rolling risk metrics saved to {rolling_output_path}")

if __name__ == "__main__":
    main()