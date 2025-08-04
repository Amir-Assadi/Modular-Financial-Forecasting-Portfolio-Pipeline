from portfolio_optimisation.optimizer import solve_optimization 
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

def sharpe_ratio_rebalance_strategy(config, prices, cov_matrix, constraints, risk_free_rate):
    weights_history = []
    prev_weights = None
    rebalance = config.get("rebalance", 1)  # Default to daily if not set
    logger.info(f"Starting sharpe_ratio_rebalance_strategy with dynamic universe and rebalance={rebalance}.")
    lookback = config.get("lookback_mean_return", 252)
    max_weight = config.get("max_weights", 1.0)  # Use a single float for all tickers

    for i in range(lookback, len(prices)):
        if (i - lookback) % rebalance == 0:
            window_prices = prices.iloc[i-lookback:i]
            available_tickers = window_prices.dropna(axis=1).columns.tolist()
            if len(available_tickers) == 0:
                weights = pd.Series(np.nan, index=prices.columns)
            elif len(available_tickers) == 1:
                weights = pd.Series(0.0, index=prices.columns)
                weights[available_tickers[0]] = 1.0
            else:
                window_prices = window_prices[available_tickers]
                mean_returns = window_prices.pct_change().dropna().mean()
                mean_annual = (1 + mean_returns) ** 252 - 1
                mu = np.array([mean_annual[t] for t in available_tickers])
                cov = window_prices.pct_change().dropna().cov().values

                # Use the same max_weight for all tickers
                bounds = [(0.0, max_weight)] * len(available_tickers)
                local_constraints = dict(constraints)
                local_constraints["bounds"] = bounds

                try:
                    opt_weights = solve_optimization(
                        mu, cov, local_constraints, risk_free_rate, obj_type="max_sharpe"
                    )
                    opt_weights = opt_weights / opt_weights.sum()
                    weights = pd.Series(0.0, index=prices.columns)
                    weights[available_tickers] = opt_weights
                except Exception as e:
                    logger.error(f"Optimization failed at index {i}: {e}")
                    # fallback: hold previous weights or go to cash if available
                    if prev_weights is not None:
                        weights = prev_weights.copy()
                    elif "cash" in prices.columns:
                        weights = pd.Series(0.0, index=prices.columns)
                        weights["cash"] = 1.0
                    else:
                        weights = pd.Series(np.nan, index=prices.columns)
            prev_weights = weights
        else:
            weights = prev_weights if prev_weights is not None else pd.Series(np.nan, index=prices.columns)
        weights_history.append(weights)
        if (i - 252) % 50 == 0 or i == len(prices) - 1:
            logger.info(f"Processed day {i} / {len(prices)}")
    weights_df = pd.DataFrame(weights_history, index=prices.index[lookback:])
    logger.info("Completed sharpe_ratio_rebalance_strategy with dynamic universe and rebalance={rebalance}.")
    return weights_df

def mean_reversion_strategy(config, prices):
    lookback = config.get("lookback", 252)
    rebalance = config.get("rebalance", 5)
    max_weights = config.get("max_weights", {})
    lower_1st = config.get("lower_1st", 1)
    lower_2nd = config.get("lower_2nd", 2)
    upper_1st = config.get("upper_1st", 1)
    upper_2nd = config.get("upper_2nd", 2)
    fill_steps = config.get("fill_steps", 5)
    tickers = prices.columns.tolist()

    weights_history = []
    cash_col = "CASH" if "CASH" in prices.columns else None
    prev_weights = pd.Series(0.0, index=tickers)
    if cash_col:
        prev_weights[cash_col] = 1.0  # Start with 100% cash

    highest_allocation = {t: 0.0 for t in tickers}

    start_idx = lookback if lookback != "full_period" else 1
    for i in range(start_idx, len(prices)):
        if (i - start_idx) % rebalance != 0:
            if cash_col:
                prev_weights[cash_col] = prev_weights.get(cash_col, 0.0)
            weights_history.append(prev_weights.copy())
            continue

        if i == start_idx:
            weights = prev_weights.copy()
            weights_history.append(weights)
            continue

        if lookback == "full_period":
            window = prices.iloc[:i]
        else:
            window = prices.iloc[i-lookback:i]
        log_prices = np.log(window)
        weights = pd.Series(0.0, index=tickers)
        cash = prev_weights[cash_col] if cash_col else 0.0
        if cash_col:
            weights[cash_col] = cash

        for t in tickers:
            if t == cash_col or t not in max_weights:
                continue
            series = log_prices[t].dropna()
            if len(series) < 2:
                continue

            x = np.arange(len(series))
            y = series.values
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            trend = m * x + c
            mean = trend[-1]
            std = np.std(y - trend)
            if np.isnan(std) or std == 0:
                continue

            current_log_price = np.log(prices[t].iloc[i])
            l1 = mean - lower_1st * std
            l2 = mean - lower_2nd * std
            u1 = mean + upper_1st * std
            u2 = mean + upper_2nd * std

            max_w = max_weights[t]
            prev = prev_weights[t]
            pos = prev  # Default: hold previous

            # --- SELL LOGIC ---
            if prev > 0 and current_log_price >= u1:
                if current_log_price >= u2:
                    pos = 0.0
                else:
                    frac = (u2 - current_log_price) / (u2 - u1)
                    frac = np.clip(frac, 0, 1)
                    pos = 0.5 * prev * frac
                highest_allocation[t] = 0.0
                if pos < prev:
                    cash += (prev - pos)

            # --- BUY LOGIC ---
            elif current_log_price <= l2:
                pos = max_w
                highest_allocation[t] = max_w
                if pos > prev:
                    buy_amt = pos - prev
                    buy_amt = min(buy_amt, cash)
                    pos = prev + buy_amt
                    cash -= buy_amt
            elif current_log_price <= l1:
                step_size = (l1 - l2) / fill_steps
                steps_filled = int((l1 - current_log_price) // step_size) + 1
                alloc = 0.5 * max_w + 0.1 * max_w * steps_filled
                alloc = min(alloc, max_w)
                pos = max(prev, alloc, highest_allocation[t])
                highest_allocation[t] = pos
                if pos > prev:
                    buy_amt = pos - prev
                    buy_amt = min(buy_amt, cash)
                    pos = prev + buy_amt
                    cash -= buy_amt
            elif current_log_price <= l1 + 1e-8:
                pos = max(prev, 0.5 * max_w, highest_allocation[t])
                highest_allocation[t] = pos
                if pos > prev:
                    buy_amt = pos - prev
                    buy_amt = min(buy_amt, cash)
                    pos = prev + buy_amt
                    cash -= buy_amt
            # --- HOLD LOGIC ---
            else:
                pos = prev

            weights[t] = pos

        if cash_col:
            weights[cash_col] = max(cash, 0.0)
        weights = weights.fillna(0.0)

        # --- Ensure total allocation does not exceed 1.0 ---
        total_alloc = weights.sum()
        if total_alloc > 1.0:
            weights = weights / total_alloc

        prev_weights = weights
        weights_history.append(weights.copy())

        if (i - start_idx) % 50 == 0 or i == len(prices) - 1:
            logger.info(f"Processed day {i} / {len(prices)}")

    weights_df = pd.DataFrame(weights_history, index=prices.index[start_idx:])
    logger.info("Completed mean_reversion_strategy.")
    return weights_df

def region_scaled_mean_reversion(config, prices):
    lookback = config.get("lookback", 252)
    rebalance = config.get("rebalance", 5)
    max_weights = config.get("max_weights", {})
    lower_1st = config.get("lower_1st", 1)
    lower_2nd = config.get("lower_2nd", 2)
    upper_1st = config.get("upper_1st", 1)
    upper_2nd = config.get("upper_2nd", 2)
    region_multipliers = config.get("region_multipliers", {})
    tickers = prices.columns.tolist()

    weights_history = []
    cash_col = "CASH" if "CASH" in prices.columns else None
    prev_weights = pd.Series(0.0, index=tickers)
    if cash_col:
        prev_weights[cash_col] = 1.0

    start_idx = lookback if lookback != "full_period" else 1
    for i in range(start_idx, len(prices)):
        if (i - start_idx) % rebalance != 0:
            if cash_col:
                prev_weights[cash_col] = prev_weights.get(cash_col, 0.0)
            weights_history.append(prev_weights.copy())
            continue

        if i == start_idx:
            weights = prev_weights.copy()
            weights_history.append(weights)
            continue

        if lookback == "full_period":
            window = prices.iloc[:i]
        else:
            window = prices.iloc[i-lookback:i]
        log_prices = np.log(window)
        weights = pd.Series(0.0, index=tickers)
        cash = prev_weights[cash_col] if cash_col else 0.0
        if cash_col:
            weights[cash_col] = cash

        for t in tickers:
            if t == cash_col or t not in max_weights:
                continue
            series = log_prices[t].dropna()
            if len(series) < 2:
                continue

            x = np.arange(len(series))
            y = series.values
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            trend = m * x + c
            mean = trend[-1]
            std = np.std(y - trend)
            if np.isnan(std) or std == 0:
                continue

            current_log_price = np.log(prices[t].iloc[i])
            l1 = mean - lower_1st * std
            l2 = mean - lower_2nd * std
            u1 = mean + upper_1st * std
            u2 = mean + upper_2nd * std

            max_w = max_weights[t]
            prev = prev_weights[t]
            pos = prev

            # LOG-SPACE REGION LOGIC
            if current_log_price < l2:
                pct_drop = (l2 - current_log_price) / abs(l2) * 100
                buy_mult = region_multipliers.get("beyond_l2", {}).get("buy", 0)
                alloc = min(prev + buy_mult * pct_drop / 100 * max_w, max_w)
                pos = alloc
            elif l2 <= current_log_price < l1:
                pct_drop = (l1 - current_log_price) / abs(l1) * 100
                buy_mult = region_multipliers.get("l1_l2", {}).get("buy", 0)
                profit_mult = region_multipliers.get("l1_l2", {}).get("take_profit", 0)
                alloc = min(prev + buy_mult * pct_drop / 100 * max_w, max_w)
                if current_log_price > prev:
                    pct_up = (current_log_price - prev) / abs(prev) * 100 if prev > 0 else 0
                    alloc = max(alloc - profit_mult * pct_up / 100 * max_w, 0)
                pos = alloc
            elif l1 <= current_log_price < mean:
                pct_drop = (mean - current_log_price) / abs(mean) * 100
                buy_mult = region_multipliers.get("mean_l1", {}).get("buy", 0)
                profit_mult = region_multipliers.get("mean_l1", {}).get("take_profit", 0)
                alloc = min(prev + buy_mult * pct_drop / 100 * max_w, max_w)
                if current_log_price > prev:
                    pct_up = (current_log_price - prev) / abs(prev) * 100 if prev > 0 else 0
                    alloc = max(alloc - profit_mult * pct_up / 100 * max_w, 0)
                pos = alloc
            elif mean <= current_log_price < u1:
                pct_up = (current_log_price - mean) / abs(mean) * 100
                buy_mult = region_multipliers.get("mean_u1", {}).get("buy", 0)
                profit_mult = region_multipliers.get("mean_u1", {}).get("take_profit", 0)
                alloc = max(prev - profit_mult * pct_up / 100 * max_w, 0)
                pos = alloc
            elif u1 <= current_log_price < u2:
                pct_up = (current_log_price - u1) / abs(u1) * 100
                profit_mult = region_multipliers.get("u1_u2", {}).get("take_profit", 0)
                alloc = max(prev - profit_mult * pct_up / 100 * max_w, 0)
                pos = alloc
            elif current_log_price >= u2:
                pct_up = (current_log_price - u2) / abs(u2) * 100
                profit_mult = region_multipliers.get("beyond_u2", {}).get("take_profit", 0)
                alloc = max(prev - profit_mult * pct_up / 100 * max_w, 0)
                pos = alloc

            weights[t] = pos

        if cash_col:
            weights[cash_col] = max(cash, 0.0)
        weights = weights.fillna(0.0)

        # --- Ensure total allocation sums to exactly 1.0 ---
        total_alloc = weights.sum()
        if total_alloc > 1.0:
            weights = weights / total_alloc
        elif total_alloc < 1.0 and cash_col:
            # Assign any leftover to cash
            weights[cash_col] = 1.0 - weights.drop(cash_col).sum()

        prev_weights = weights
        weights_history.append(weights.copy())

        if (i - start_idx) % 50 == 0 or i == len(prices) - 1:
            logger.info(f"Processed day {i} / {len(prices)}")

    weights_df = pd.DataFrame(weights_history, index=prices.index[start_idx:])
    logger.info("Completed region_scaled_mean_reversion (log version).")
    return weights_df

STRATEGIES = {
    "sharpe_ratio_rebalance": sharpe_ratio_rebalance_strategy,
    "mean_reversion": mean_reversion_strategy,
    "region_scaled_mean_reversion": region_scaled_mean_reversion,
}