import os
import subprocess

# Set these paths as needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REGIME_DIR = os.path.join(BASE_DIR, "regime_detection")
FORECAST_DIR = os.path.join(BASE_DIR, "forcasting_ml")
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolio_optimisation")
STRAT_DIR = os.path.join(BASE_DIR, "strategy_test")

def run_python(script_path, args=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR  # Ensures all submodules are found
    cmd = ["python", script_path]
    if args:
        cmd += args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def main(run_model_training=False, run_visualisation=False, run_live_feed=False, run_inference=False, run_portfolio=False, run_strat=False, run_ml=False):
    # If no flags, run full pipeline 
    if not (run_model_training or run_visualisation or run_live_feed or run_inference or run_portfolio or run_strat or run_ml):
        run_python(os.path.join(DATA_DIR, "data_ingestion.py"))
        run_python(os.path.join(DATA_DIR, "macro_data.py"))
        run_python(os.path.join(REGIME_DIR, "hidden_markov.py"))
        run_python(os.path.join(REGIME_DIR, "changepoint_detection.py"))
    # Optional steps
    if run_model_training:  
        run_python(os.path.join(FORECAST_DIR, "feature_engineering.py"))
        run_python(os.path.join(FORECAST_DIR, "model_training.py"))
    if run_visualisation:
        run_python(os.path.join(REGIME_DIR, "visualise_regimes.py"))
    if run_live_feed:
        run_python(os.path.join(DATA_DIR, "live_feed_interface.py"))
    if run_inference:
        run_python(os.path.join(FORECAST_DIR, "model_inference.py"))
    if run_portfolio:
        run_python(os.path.join(PORTFOLIO_DIR, "risk_metrics.py"))
        run_python(os.path.join(PORTFOLIO_DIR, "constraints.py"))
        run_python(os.path.join(PORTFOLIO_DIR, "optimizer.py"))
    if run_strat:
        run_python(os.path.join(STRAT_DIR, "backtester.py"))
        run_python(os.path.join(STRAT_DIR, "performance_evaluator.py"))
    if run_ml:
        run_python(os.path.join(FORECAST_DIR, "ml_strategy.py"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="Run model training step")
    parser.add_argument("--visualise", action="store_true", help="Run regime visualisation step")
    parser.add_argument("--live", action="store_true", help="Run live feed step")
    parser.add_argument("--inference", action="store_true", help="Run model inference step")
    parser.add_argument("--portfolio", action="store_true", help="Run portfolio risk, constraints, and optimizer")
    parser.add_argument("--strat", action="store_true", help="Run strategy backtester and performance evaluator")
    parser.add_argument("--ml_strat", action="store_true", help="Run ML strategy backtest")
    args = parser.parse_args()
    main(
        run_model_training=args.model,
        run_visualisation=args.visualise,
        run_live_feed=args.live,
        run_inference=args.inference,
        run_portfolio=args.portfolio,
        run_strat=args.strat,
        run_ml=args.ml_strat
    )