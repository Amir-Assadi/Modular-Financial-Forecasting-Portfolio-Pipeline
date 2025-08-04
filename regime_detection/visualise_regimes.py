import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_regimes_and_changepoints(ticker, markov_output_dir, markov_prob_output_dir, changepoint_output_dir):
    # Load data with correct filenames
    markov_path = os.path.join(markov_output_dir, f"{ticker}_regimes.csv")
    prob_path = os.path.join(markov_prob_output_dir, f"{ticker}_regime_probs.csv")
    changepoint_path = os.path.join(changepoint_output_dir, f"changepoints_{ticker}.csv")

    df = pd.read_csv(markov_path, parse_dates=["Date"])
    prob_df = pd.read_csv(prob_path, parse_dates=["Date"])
    cp_df = pd.read_csv(changepoint_path) if os.path.exists(changepoint_path) else None

    # Ensure 'returns' column exists for plotting
    if 'returns' not in df.columns and 'Close' in df.columns:
        df['returns'] = df['Close'].pct_change()
    elif 'returns' not in df.columns:
        raise ValueError("Neither 'returns' nor 'Close' found in Markov output for plotting.")

    # Map regime numbers to "Regime 0", "Regime 1", etc.
    regimes_raw = df["regime"].values
    regimes = np.array([int(r) if not pd.isna(r) else np.nan for r in regimes_raw])
    df["regime_int"] = regimes
    unique_regimes = sorted(set(r for r in regimes if not np.isnan(r)))
    regime_labels = {regime: f"Regime {regime}" for regime in unique_regimes}
    color_list = ["red", "blue", "green", "orange", "purple"]
    color_map = {f"Regime {i}": color_list[i % len(color_list)] for i in unique_regimes}

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    # 1. Plot returns and changepoints
    axs[0].plot(df["Date"], df["returns"], label="Returns", color="black")
    if cp_df is not None:
        for i, row in cp_df.iterrows():
            axs[0].axvline(pd.to_datetime(row["end_time"]), color="red", linestyle="--", alpha=0.7, label="Changepoint" if i == 0 else "")
    axs[0].set_ylabel("Returns")
    axs[0].set_title(f"{ticker} Returns with Changepoints")

    # 2. Plot regimes as colored background with consistent colors
    regimes = df["regime_int"].values
    already_labeled = set()
    for regime in unique_regimes:
        mask = regimes == regime
        label = regime_labels[regime]
        legend_label = label if label not in already_labeled else None
        axs[0].fill_between(
            df["Date"], df["returns"].min(), df["returns"].max(),
            where=mask, alpha=0.15, color=color_map[label], label=legend_label
        )
        already_labeled.add(label)
    axs[0].legend(loc="upper right")

    # 3. Plot all regime probability columns except 'Date', using actual names
    prob_cols = [col for col in prob_df.columns if col.lower() != "date"]
    if not prob_cols:
        print(f"[WARNING] No regime probability columns found for {ticker}. Skipping regime probability and uncertainty plots.")
        axs[1].text(0.5, 0.5, 'No regime probability columns found.', ha='center', va='center', fontsize=14, color='red', transform=axs[1].transAxes)
        axs[1].set_axis_off()
        axs[2].text(0.5, 0.5, 'No regime probability columns found.', ha='center', va='center', fontsize=14, color='red', transform=axs[2].transAxes)
        axs[2].set_axis_off()
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()
        return
    already_labeled_probs = set()
    for col in prob_cols:
        label = col
        color = None  # Use default color cycle for unknown names
        legend_label = label if label not in already_labeled_probs else None
        axs[1].plot(prob_df["Date"], prob_df[col], label=legend_label, color=color)
        already_labeled_probs.add(label)
    axs[1].set_ylabel("Regime Probability")
    axs[1].set_title("Markov Regime Probabilities")
    axs[1].legend()

    # 4. Highlight uncertainty (where max prob difference < threshold)
    threshold = 0.4
    probs = prob_df[prob_cols].values
    if probs.shape[1] > 1:
        max_probs = np.sort(probs, axis=1)[:, -1]
        second_max_probs = np.sort(probs, axis=1)[:, -2]
        uncertainty = (max_probs - second_max_probs) < threshold
        axs[2].plot(prob_df["Date"], max_probs - second_max_probs, label="Max Prob Difference")
        axs[2].axhline(threshold, color="red", linestyle="--", label="Uncertainty Threshold")
        axs[2].fill_between(prob_df["Date"], 0, 1, where=uncertainty, color="orange", alpha=0.3, label="Uncertain Regime")
        axs[2].set_ylabel("Max Prob Diff")
        axs[2].set_title("Regime Uncertainty (Low Difference = Uncertain)")
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, 'Not enough regime columns for uncertainty plot.', ha='center', va='center', fontsize=14, color='red', transform=axs[2].transAxes)
        axs[2].set_axis_off()

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    markov_output_dir = os.path.join(os.path.dirname(__file__), "markov_output")
    markov_prob_output_dir = os.path.join(os.path.dirname(__file__), "markov_prob_output")
    changepoint_output_dir = os.path.join(os.path.dirname(__file__), "changepoint_output")

    # Load tickers from config
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "data", "config_data.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tickers = config["tickers"]

    for ticker in tickers:
        print(f"Plotting for {ticker}...")
        plot_regimes_and_changepoints(ticker, markov_output_dir, markov_prob_output_dir, changepoint_output_dir)