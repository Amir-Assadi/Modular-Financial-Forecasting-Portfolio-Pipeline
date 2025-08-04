import os
import pandas as pd
import numpy as np
import yaml
import subprocess
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_feature_engineering(base_dir):
    feature_engineering_py = os.path.join(base_dir, "feature_engineering.py")
    logger.info("Running feature engineering to update features...")
    subprocess.run(['python', feature_engineering_py], check=True)
    logger.info("Feature engineering complete.")

def retrain_model(new_end_date, base_dir):
    config_data_path = os.path.join(base_dir, "..", "data", "config_data.yaml")
    with open(config_data_path, "r") as f:
        config_data = yaml.safe_load(f)
    config_data["end_date"] = new_end_date.strftime("%Y-%m-%d")
    with open(config_data_path, "w") as f:
        yaml.safe_dump(config_data, f)
    main_py = os.path.join(base_dir, "..", "main.py")
    logger.info(f"Retraining model with new end date: {new_end_date.strftime('%Y-%m-%d')}")
    subprocess.run(['python', main_py], check=True)
    run_feature_engineering(base_dir)

def fetch_price_data(ticker, start_date, end_date, save_path):
    import yfinance as yf
    from data.data_cleaning import clean_dataframe

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True
    )
    if df.empty:
        logger.error(f"No data fetched for {ticker}!")
        raise ValueError(f"No data fetched for {ticker}!")
    df = df.reset_index()
    # Drop first three rows if present (extra headers)
    df = df.drop(df.index[[0, 1, 2]], errors='ignore').reset_index(drop=True)
    # Set columns to expected order and names
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df = clean_dataframe(df)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved {ticker} data to {save_path}")

def compute_metrics(portfolio, benchmark):
    returns = portfolio.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    n_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / n_years) - 1
    bench_cagr = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / n_years) - 1
    excess_cagr = cagr - bench_cagr
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_dd = ((portfolio / portfolio.cummax()) - 1).min()
    hit_rate = (returns > 0).mean()
    # Sortino
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std() * np.sqrt(252)) if downside.std() > 0 else np.nan
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    # Benchmark Sharpe
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(252)
    # Tracking error
    tracking_error = np.std(returns - bench_returns) * np.sqrt(252)
    # Beta
    if len(returns) > 1 and len(bench_returns) > 1:
        beta = np.cov(returns, bench_returns)[0, 1] / np.var(bench_returns)
    else:
        beta = np.nan
    return {
        "CAGR": cagr,
        "Benchmark CAGR": bench_cagr,
        "Excess CAGR": excess_cagr,
        "Sharpe": sharpe,
        "Benchmark Sharpe": bench_sharpe,
        "Volatility": vol,
        "Max Drawdown": max_dd,
        "Sortino": sortino,
        "Calmar": calmar,
        "Hit Rate": hit_rate,
        "Tracking Error": tracking_error,
        "Beta": beta
    }

def plot_performance(perf_df, bench_df, ml_strategy_data_dir, ticker):
    # Equity curve
    plt.figure(figsize=(12, 6))
    perf_df["PortfolioValue"].plot(label="ML Strategy")
    bench_df["BenchmarkValue"].plot(label="Benchmark")
    plt.title("Portfolio Value vs Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Drawdown
    drawdown = (perf_df["PortfolioValue"] / perf_df["PortfolioValue"].cummax()) - 1
    plt.figure(figsize=(12, 4))
    drawdown.plot(color="red")
    plt.title("Drawdown")
    plt.tight_layout()
    plt.show()

    # Histogram of returns
    returns = perf_df["PortfolioValue"].pct_change().dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=100, alpha=0.7)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.show()

def plot_allocations(alloc_df, ticker):
    """Plot allocation weights over time"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Asset Weight over time
    axes[0].plot(alloc_df.index, alloc_df["Asset_Weight"], label=f"{ticker} Weight", color='blue', linewidth=1)
    axes[0].fill_between(alloc_df.index, 0, alloc_df["Asset_Weight"], alpha=0.3, color='blue')
    axes[0].set_ylabel("Asset Weight")
    axes[0].set_title(f"Portfolio Allocation Over Time - {ticker}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Cash vs Asset allocation stacked
    axes[1].fill_between(alloc_df.index, 0, alloc_df["Cash_Weight"], label="Cash", alpha=0.7, color='green')
    axes[1].fill_between(alloc_df.index, alloc_df["Cash_Weight"], 1, label=f"{ticker}", alpha=0.7, color='blue')
    axes[1].set_ylabel("Portfolio Weight")
    axes[1].set_title("Cash vs Asset Allocation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Plot 3: Trading signals
    signal_colors = ['red' if signal == 0 else 'green' for signal in alloc_df["Signal"]]
    axes[2].scatter(alloc_df.index, alloc_df["Signal"], c=signal_colors, alpha=0.6, s=1)
    axes[2].set_ylabel("Signal")
    axes[2].set_title("Trading Signals (0=Sell/Cash, 1=Buy/Hold)")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    total_days = len(alloc_df)
    days_in_position = len(alloc_df[alloc_df["Position"] == "Long"])
    days_in_cash = total_days - days_in_position
    
    print(f"\n--- Allocation Summary for {ticker} ---")
    print(f"Total trading days: {total_days}")
    print(f"Days in position: {days_in_position} ({days_in_position/total_days*100:.1f}%)")
    print(f"Days in cash: {days_in_cash} ({days_in_cash/total_days*100:.1f}%)")
    print(f"Average asset weight: {alloc_df['Asset_Weight'].mean():.3f}")
    print(f"Number of buy signals: {len(alloc_df[alloc_df['Signal'] == 1])}")
    print(f"Number of sell signals: {len(alloc_df[alloc_df['Signal'] == 0])}")
    
    # Position changes (trades)
    position_changes = alloc_df["Position"].ne(alloc_df["Position"].shift()).sum()
    print(f"Number of position changes (trades): {position_changes}")
    
    return {
        "total_days": total_days,
        "days_in_position": days_in_position,
        "days_in_cash": days_in_cash,
        "avg_asset_weight": alloc_df['Asset_Weight'].mean(),
        "position_changes": position_changes,
        "buy_signals": len(alloc_df[alloc_df['Signal'] == 1]),
        "sell_signals": len(alloc_df[alloc_df['Signal'] == 0])
    }

def main():
    logger.info("Starting ML strategy backtest...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(base_dir, "..", "main.py")
    ml_config_path = os.path.join(base_dir, "ml_strategy_config.yaml")
    ml_config = load_config(ml_config_path)
    ml_start_date = ml_config["start_date"]
    ml_end_date = ml_config["end_date"]
    ticker = ml_config["tickers"][0]

    # --- 1. Data and Feature Preparation (Full Range) - ONLY ONCE ---
    config_data_path = os.path.join(base_dir, "..", "data", "config_data.yaml")
    with open(config_data_path, "r") as f:
        config_data = yaml.safe_load(f)
    config_start = config_data["start_date"]
    
    # Create ml_strategy_data directory
    ml_strategy_data_dir = os.path.join(base_dir, "ml_strategy_data")
    os.makedirs(ml_strategy_data_dir, exist_ok=True)
    
    # Define paths for ML strategy data (these will be saved once and never overwritten)
    dst_price = os.path.join(ml_strategy_data_dir, f"{ticker}.csv")
    dst_feat = os.path.join(ml_strategy_data_dir, f"features_{ticker}.csv")
    
    # Check if ML strategy data already exists
    if not (os.path.exists(dst_price) and os.path.exists(dst_feat)):
        logger.info("ML strategy data files don't exist. Creating them for the full range...")
        
        # Set config data to cover full range from config start to ML strategy end
        original_config_start = config_data["start_date"]
        original_config_end = config_data["end_date"]
        config_data["start_date"] = config_start
        config_data["end_date"] = ml_end_date
        with open(config_data_path, "w") as f:
            yaml.safe_dump(config_data, f)
        logger.info(f"Patched config_data.yaml to cover full range: {config_start} - {ml_end_date}")

        # Run main pipeline ONCE to fetch all data and generate features for full range
        logger.info("Running main pipeline for full range data collection...")
        subprocess.run(['python', main_py], check=True)
        logger.info("Running feature engineering and model training for full range...")
        subprocess.run(['python', main_py, '--model'], check=True)
        logger.info("Main pipeline and feature/model step complete for full range.")

        # Copy data to ML strategy directory (this happens ONLY ONCE)
        src_price = os.path.join(base_dir, "..", "data", "historical_data", f"{ticker}.csv")
        src_feat = os.path.join(base_dir, "feature_engineering_output", f"features_{ticker}.csv")
        
        # Copy files directly (they already cover the full range)
        import shutil
        shutil.copy2(src_price, dst_price)
        shutil.copy2(src_feat, dst_feat)
        logger.info(f"Saved {dst_price} and {dst_feat} for ML strategy. These will NOT be overwritten.")
        
        # Restore original config
        config_data["start_date"] = original_config_start
        config_data["end_date"] = original_config_end
        with open(config_data_path, "w") as f:
            yaml.safe_dump(config_data, f)
    else:
        logger.info(f"ML strategy data files already exist. Using existing {dst_price} and {dst_feat}")

    # --- 2. Initial Model Training (Config Start to ML Strategy Start) ---
    logger.info("Performing initial model training...")
    config_data["start_date"] = config_start
    config_data["end_date"] = ml_start_date
    with open(config_data_path, "w") as f:
        yaml.safe_dump(config_data, f)
    logger.info(f"Training initial model for period: {config_start} - {ml_start_date}")
    
    # Run pipeline for initial training (this will train on config_start to ml_start_date)
    subprocess.run(['python', main_py], check=True)
    subprocess.run(['python', main_py, '--model'], check=True)
    logger.info("Initial model training complete.")

    # --- 3. Load ML Strategy Data (covers full range, used only for inference) ---
    logger.info("Loading ML strategy data for backtest...")
    if not os.path.exists(dst_feat):
        logger.error(f"Feature file {dst_feat} does not exist. Skipping ML strategy for {ticker}.")
        return
    if not os.path.exists(dst_price):
        logger.error(f"Price file {dst_price} does not exist. Skipping ML strategy for {ticker}.")
        return
    features_df = pd.read_csv(dst_feat, parse_dates=["Date"]).set_index("Date")
    prices_df = pd.read_csv(dst_price, parse_dates=["Date"]).set_index("Date")
    features_df.index = pd.to_datetime(features_df.index)
    prices_df.index = pd.to_datetime(prices_df.index)
    logger.info(f"ML strategy data covers {features_df.index.min().date()} to {features_df.index.max().date()}")

    # Initialize portfolio tracking variables
    capital = ml_config["capital"]["amount"]
    portfolio_value_history = []
    benchmark_value_history = []
    allocation_history = []  # Track allocations over time
    cash = capital
    asset_units = 0.0
    in_position = False
    # --- 4. Backtest Loop (Using Precomputed Data/Features) ---
    current_start = pd.to_datetime(ml_start_date)
    backtest_end = pd.to_datetime(ml_end_date)
    one_year = pd.DateOffset(days=252)
    model_dir = os.path.join(base_dir, ml_config.get("model_dir", "trained_model"))
    prediction_horizon = ml_config.get("prediction_horizon", 1)
    model_name = ml_config["model_name"]
    perf_out = os.path.join(base_dir, "ml_strategy_data", f"{ticker}_ml_strategy_performance.csv")

    while current_start < backtest_end:
        current_end = min(current_start + one_year, backtest_end)
        logger.info(f"Backtest segment: {current_start.date()} to {current_end.date()}")
        
        # Load model for this segment (model was trained up to current_end in previous iteration or initial training)
        model_path = os.path.join(
            model_dir,
            f"{ticker}_future_return_{prediction_horizon}d_class_{model_name}_classification_bundle.pkl"
        )
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist. Skipping segment {current_start.date()} to {current_end.date()} for {ticker}.")
            current_start = current_end
            continue
        from forcasting_ml.model_utils import load_model
        try:
            model, scaler, extra = load_model(model_path)
            feature_cols = extra["feature_cols"]
        except Exception as e:
            logger.error(f"Could not load model for {ticker}: {e}")
            break
            
        # Get segment dates for inference (using precomputed ML strategy data)
        segment_dates = features_df.index[(features_df.index >= current_start) & (features_df.index < current_end)]
        logger.info(f"Segment {current_start.date()} to {current_end.date()} has {len(segment_dates)} dates for inference.")
        
        if len(segment_dates) == 0:
            logger.warning(f"No dates found for segment {current_start.date()} to {current_end.date()}. Skipping.")
            current_start = current_end
            continue
            
        # Perform inference for this segment using precomputed features
        skipped = 0
        for date in segment_dates:
            # Check if required features are available
            missing_cols = [col for col in feature_cols if col not in features_df.columns]
            if missing_cols:
                logger.error(f"Missing columns in features for prediction on {date}: {missing_cols}. Skipping date.")
                skipped += 1
                continue
                
            try:
                # Extract features for this date
                X = features_df.loc[date, feature_cols].values.reshape(1, -1)
                
                # Check for NaN values
                if np.isnan(X).any():
                    nan_cols = [col for col, val in zip(feature_cols, X.flatten()) if np.isnan(val)]
                    logger.warning(f"NaN detected in features for {date}: {nan_cols}. Skipping date.")
                    skipped += 1
                    continue
                
                # Scale features and make prediction
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                
                # Get price for this date
                price = prices_df.loc[date, "Close"]
                
                # --- ALL-IN/ALL-OUT LOGIC FOR CLASSIFICATION ---
                if pred == 1 and not in_position:  # Buy signal
                    asset_units = cash / price
                    cash = 0.0
                    in_position = True
                    logger.debug(f"{date}: BUY signal - Going all-in at price {price:.2f}")
                elif pred == 0 and in_position:  # Sell signal
                    cash = asset_units * price
                    asset_units = 0.0
                    in_position = False
                    logger.debug(f"{date}: SELL signal - Going all-out at price {price:.2f}")
                
                # Calculate portfolio value
                port_val = cash + asset_units * price
                portfolio_value_history.append({"Date": date, "PortfolioValue": port_val})
                
                # Track allocation (weight in the asset vs cash)
                allocation_weight = (asset_units * price) / port_val if port_val > 0 else 0.0
                allocation_history.append({
                    "Date": date, 
                    "Asset_Weight": allocation_weight,
                    "Cash_Weight": 1.0 - allocation_weight,
                    "Position": "Long" if in_position else "Cash",
                    "Signal": pred
                })
                
                # Calculate benchmark value (buy and hold from start)
                if len(portfolio_value_history) == 1:  # First day
                    bench_start_price = price
                bench_val = capital * (price / bench_start_price)
                benchmark_value_history.append({"Date": date, "BenchmarkValue": bench_val})
                
            except Exception as e:
                logger.error(f"Failed to process date {date}: {e}. Skipping.")
                skipped += 1
                continue
        
        logger.info(f"Segment {current_start.date()} to {current_end.date()}: {skipped} dates skipped due to errors.")
        
        # --- Retrain model for next segment (if not at the end) ---
        if current_end < backtest_end:
            logger.info(f"Retraining model for next segment...")
            
            # Update config for retraining (train from config_start to current_end)
            config_data["start_date"] = config_start
            config_data["end_date"] = current_end.strftime("%Y-%m-%d")
            with open(config_data_path, "w") as f:
                yaml.safe_dump(config_data, f)
            
            logger.info(f"Retraining model for period: {config_start} - {current_end.date()}")
            
            # Run pipeline for retraining (this updates feature_engineering_output/ and trained_model/, NOT ml_strategy_data/)
            subprocess.run(['python', main_py], check=True)
            subprocess.run(['python', main_py, '--model'], check=True)
            logger.info("Model retraining complete.")
        
        # Move to next segment
        current_start = current_end
    # --- 5. Save Performance Results ---
    if not portfolio_value_history or not benchmark_value_history:
        logger.error(f"No portfolio or benchmark values were recorded!\n"
                     f"portfolio_value_history length: {len(portfolio_value_history)}\n"
                     f"benchmark_value_history length: {len(benchmark_value_history)}")
        raise RuntimeError("No portfolio or benchmark values were recorded during backtest.")
    
    # Create performance DataFrames
    perf_df = pd.DataFrame(portfolio_value_history).set_index("Date")
    bench_df = pd.DataFrame(benchmark_value_history).set_index("Date")
    alloc_df = pd.DataFrame(allocation_history).set_index("Date")
    
    # Join all data
    result = perf_df.join(bench_df).join(alloc_df)
    result.to_csv(perf_out)
    logger.info(f"Performance and allocation data saved to {perf_out}")
    
    # Save allocation history separately
    allocation_path = os.path.join(base_dir, "ml_strategy_data", f"{ticker}_allocation_history.csv")
    alloc_df.to_csv(allocation_path)
    logger.info(f"Allocation history saved to {allocation_path}")

    # --- 6. Compute and Save Performance Metrics ---
    metrics = compute_metrics(perf_df["PortfolioValue"], bench_df["BenchmarkValue"])
    metrics_path = os.path.join(base_dir, "ml_strategy_data", f"{ticker}_performance_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Performance metrics saved to {metrics_path}")

    # Print metrics to console
    print("\nML Strategy Performance Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "Ratio" in k or "Sharpe" in k or "Sortino" in k or "Calmar" in k or "Beta" in k:
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: {v}")

    # --- 7. Generate Plots ---
    plot_performance(perf_df, bench_df, os.path.join(base_dir, "ml_strategy_data"), ticker)
    allocation_stats = plot_allocations(alloc_df, ticker)
    
    # Save allocation statistics
    allocation_stats_path = os.path.join(base_dir, "ml_strategy_data", f"{ticker}_allocation_stats.csv")
    pd.DataFrame([allocation_stats]).to_csv(allocation_stats_path, index=False)
    logger.info(f"Allocation statistics saved to {allocation_stats_path}")
    
    logger.info("ML strategy backtest completed successfully!")

if __name__ == "__main__":
    main()