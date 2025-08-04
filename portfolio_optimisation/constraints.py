import yaml
import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_weight_bounds(tickers, bounds_cfg):
    bounds = []
    for ticker in tickers:
        if ticker in bounds_cfg:
            bounds.append(tuple(bounds_cfg[ticker]))
        elif "default" in bounds_cfg:
            bounds.append(tuple(bounds_cfg["default"]))
        else:
            # If no bound specified, use [0, 1] as default
            bounds.append((0.0, 1.0))
    return bounds

def get_constraints_from_yaml(yaml_path):
    config = load_config(yaml_path)
    tickers = config["tickers"]
    constraints_cfg = config.get("constraints", {})
    bounds_cfg = constraints_cfg.get("bounds", {})
    allow_short = constraints_cfg.get("allow_short", False)
    weight_sum = constraints_cfg.get("weight_sum", True)
    leverage = constraints_cfg.get("leverage", None)

    # Per-asset bounds
    bounds = get_weight_bounds(tickers, bounds_cfg)

    # Long-only or allow short
    if not allow_short:
        # Ensure lower bounds are at least 0
        bounds = [(max(0.0, b[0]), b[1]) for b in bounds]

    # Constraints summary
    constraints = {
        "tickers": tickers,
        "bounds": bounds,
        "allow_short": allow_short,
        "weight_sum": weight_sum,
        "leverage": leverage
    }

    logger.info(f"Constraints loaded: {constraints}")
    return constraints

def print_constraints(constraints):
    print("Portfolio Constraints:")
    print(f"Tickers: {constraints['tickers']}")
    print(f"Per-asset bounds: {constraints['bounds']}")
    print(f"Allow short: {constraints['allow_short']}")
    print(f"Enforce weights sum to 1: {constraints['weight_sum']}")
    if constraints['leverage'] is not None:
        print(f"Leverage constraint: {constraints['leverage']}")

if __name__ == "__main__":
    yaml_path = os.path.join(os.path.dirname(__file__), "portfolio.yaml")
    constraints = get_constraints_from_yaml(yaml_path)
    print_constraints(constraints)