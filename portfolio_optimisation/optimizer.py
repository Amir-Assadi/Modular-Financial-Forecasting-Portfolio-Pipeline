import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.optimize import minimize
from utils.logger import get_logger


logger = get_logger(__name__)

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_risk_metrics(risk_metrics_path):
    return pd.read_csv(risk_metrics_path)

def load_covariance_matrix(cov_path):
    return pd.read_csv(cov_path, index_col=0)

def load_constraints(constraints_path):
    from constraints import get_constraints_from_yaml
    return get_constraints_from_yaml(constraints_path)

def get_capital(config):
    cap = config.get("capital", {})
    return cap.get("amount", 1.0), cap.get("currency", "GBP")

def get_expected_returns(config, prices=None, full_prices=None):
    method = config["expected_returns"]["method"]
    manual_values = config["expected_returns"].get("manual_values", {})
    tickers = config["tickers"]
    if method == "manual":
        # Manual expected return as percentage
        return {t: manual_values[t] for t in tickers}
    elif method == "manual_price":
        # Manual expected return as future price
        if prices is None:
            raise ValueError("Current prices are required for manual_price expected returns.")
        expected_prices = np.array([manual_values[t] for t in tickers])
        current_prices = np.array([prices[t] for t in tickers])
        expected_returns = (expected_prices - current_prices) / current_prices
        return dict(zip(tickers, expected_returns))
    elif method == "mean_return":
        # Historical mean return (annualized)
        if full_prices is None:
            raise ValueError("Full price history is required for mean_return expected returns.")
        returns = full_prices.pct_change().dropna()
        mean_daily = returns.mean()
        mean_annual = (1 + mean_daily) ** 252 - 1
        return mean_annual.to_dict()
    else:
        raise NotImplementedError("Only manual, manual_price, and mean_return expected returns supported in this template.")
    
def objective_function(weights, mu, cov, risk_free_rate, obj_type):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    if obj_type == "max_sharpe":
        if port_vol == 0:
            return -1e6
        return -(port_return - risk_free_rate) / port_vol
    elif obj_type == "min_volatility":
        return port_vol
    elif obj_type == "risk_parity":
        # Risk parity: minimize difference between asset risk contributions
        asset_contrib = weights * np.dot(cov, weights)
        return np.std(asset_contrib / asset_contrib.sum())
    else:
        raise ValueError(f"Unknown objective type: {obj_type}")

def solve_optimization(mu, cov, constraints, risk_free_rate, obj_type):
    n = len(mu)
    bounds = constraints["bounds"]
    allow_short = constraints["allow_short"]
    weight_sum = constraints["weight_sum"]
    leverage = constraints.get("leverage", None)

    x0 = np.ones(n) / n

    cons = []
    if weight_sum:
        cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    if leverage is not None:
        cons.append({'type': 'ineq', 'fun': lambda w: leverage - np.sum(np.abs(w))})
    if allow_short:
        # Use bounds as specified in YAML (could be negative)
        bounds = [b for b in bounds]
    else:
        # Enforce lower bounds >= 0 for long-only
        bounds = [(max(0.0, b[0]), b[1]) for b in bounds]

    result = minimize(
        objective_function,
        x0,
        args=(mu, cov, risk_free_rate, obj_type),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': False}
    )
    if not result.success:
        logger.error(f"Optimization failed: {result.message}")
        raise RuntimeError("Optimization failed")
    return result.x

def compute_portfolio_metrics(weights, mu, cov, risk_free_rate, prices, benchmark_ticker=None):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan

    # Use time series for Sortino, beta, alpha
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (portfolio_returns.mean() - risk_free_rate / 252) / downside_std if downside_std > 0 else np.nan

    beta = alpha = np.nan
    if benchmark_ticker and benchmark_ticker in returns.columns:
        benchmark_returns = returns[benchmark_ticker]
        cov = np.cov(portfolio_returns, benchmark_returns)
        var_bench = np.var(benchmark_returns)
        beta = cov[0, 1] / var_bench if var_bench > 0 else np.nan
        alpha = portfolio_returns.mean() - beta * benchmark_returns.mean() if not np.isnan(beta) else np.nan

    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "beta": beta,
        "alpha": alpha
    }


def plot_weights(weights, tickers, capital, currency):
    plt.figure(figsize=(8, 6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title(f"Portfolio Weights (Total Capital: {capital} {currency})")
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier(mu, cov, risk_free_rate, bounds, n_points=50):
    n = len(mu)
    results = np.zeros((n_points, 2))
    for i, ret in enumerate(np.linspace(min(mu), max(mu), n_points)):
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - ret}]
        res = minimize(lambda w: np.sqrt(np.dot(w, np.dot(cov, w))),
                       np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            results[i, 0] = res.fun
            results[i, 1] = ret
        else:
            results[i, :] = np.nan
    plt.figure(figsize=(8, 6))
    plt.plot(results[:, 0], results[:, 1], 'b-', label='Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_risk_return_scatter(mu, std, tickers):
    plt.figure(figsize=(8, 6))
    plt.scatter(std, mu, c='blue')
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (std[i], mu[i]))
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Scatter')
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(prices, weights, capital, currency):
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    cumulative = (1 + portfolio_returns).cumprod() * capital
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label='Portfolio')
    plt.title('Cumulative Portfolio Value (Backtest)')
    plt.xlabel('Date')
    plt.ylabel(f'Portfolio Value ({currency})')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.show()

def plot_portfolio_allocation_over_time(prices, weights, capital, currency, tickers):
    returns = prices.pct_change().dropna()
    asset_cum = (1 + returns).cumprod()
    # Each asset's value over time
    asset_values = asset_cum.multiply(weights * capital, axis=1)
    # Stacked area plot
    plt.figure(figsize=(12, 6))
    asset_values.plot.area(ax=plt.gca(), stacked=True, cmap='tab20')
    plt.title('Portfolio Value Allocation Over Time')
    plt.xlabel('Date')
    plt.ylabel(f'Value ({currency})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(cov):
    # Ensure cov is a DataFrame with matching index and columns
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(cov)
    # Compute correlation matrix
    corr = cov.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_constraints(bounds, tickers):
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    x = np.arange(len(tickers))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, lower, width=0.4, label='Lower Bound')
    plt.bar(x + 0.2, upper, width=0.4, label='Upper Bound')
    plt.xticks(x, tickers)
    plt.ylabel('Weight')
    plt.title('Per-Asset Weight Constraints')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    config_path = os.path.join(os.path.dirname(__file__), "portfolio.yaml")
    config = load_yaml_config(config_path)
    tickers = config["tickers"]
    capital, currency = get_capital(config)

    # Use correct output directory for all files
    output_dir = os.path.join(os.path.dirname(__file__), "risk_metrics_output")
    risk_metrics_path = os.path.join(output_dir, "risk_metrics_output.csv")
    cov_path = os.path.join(output_dir, "risk_metrics_output_covariance.csv")
    prices_path = os.path.join(output_dir, "risk_metrics_output_prices.csv")

    # Load prices first (needed for manual_price and mean_return)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    latest_prices = prices[tickers].iloc[-1]

    # Get expected returns (supports manual, manual_price, mean_return)
    method = config["expected_returns"]["method"]
    if method == "manual":
        expected_returns_dict = get_expected_returns(config)
    elif method == "manual_price":
        expected_returns_dict = get_expected_returns(config, prices=latest_prices)
    elif method == "mean_return":
        expected_returns_dict = get_expected_returns(config, full_prices=prices[tickers])
    else:
        raise NotImplementedError(f"Unknown expected_returns method: {method}")

    mu = np.array([expected_returns_dict[t] for t in tickers])

    # Load other data
    risk_metrics = load_risk_metrics(risk_metrics_path)
    cov = load_covariance_matrix(cov_path).loc[tickers, tickers].values
    constraints = load_constraints(config_path)
    risk_free_rate = config.get("objective", {}).get("risk_free_rate", 0.0)
    obj_type = config.get("objective", {}).get("type", "max_sharpe")
    benchmark_ticker = config.get("benchmark", None)

    logger.info(f"Solving portfolio optimization with objective: {obj_type}")
    weights = solve_optimization(mu, cov, constraints, risk_free_rate, obj_type)
    weights = weights / weights.sum()    # Normalize

    # Output weights
    weights_pct = weights * 100
    allocation = weights * capital
    output_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": np.round(weights_pct, 2),
        f"Allocation ({currency})": np.round(allocation, 2)
    })
    output_path = os.path.join(os.path.dirname(__file__), "optimized_portfolio.csv")
    output_df.to_csv(output_path, index=False)
    logger.info(f"Optimized portfolio saved to {output_path}")
    print(output_df)

    # Show per-asset risk metrics
    print("\nPer-Asset Risk Metrics:")
    print(risk_metrics.set_index("ticker").loc[tickers])

    # Portfolio metrics
    port_metrics = compute_portfolio_metrics(
        weights, mu, cov, risk_free_rate, prices[tickers], benchmark_ticker
    )
    logger.info(f"Portfolio metrics: {port_metrics}")
    print("\nPortfolio Metrics:")
    for k, v in port_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Visualizations
    plot_weights(weights, tickers, capital, currency)
    plot_efficient_frontier(mu, cov, risk_free_rate, constraints["bounds"])
    std = np.sqrt(np.diag(cov))
    plot_risk_return_scatter(mu, std, tickers)

    # Backtest cumulative returns
    plot_cumulative_returns(prices[tickers], weights, capital, currency)
    plot_portfolio_allocation_over_time(prices[tickers], weights, capital, currency, tickers)

    # Correlation heatmap
    plot_correlation_heatmap(pd.DataFrame(cov, index=tickers, columns=tickers))

    # Constraints
    plot_constraints(constraints["bounds"], tickers)
    
if __name__ == "__main__":
    main()