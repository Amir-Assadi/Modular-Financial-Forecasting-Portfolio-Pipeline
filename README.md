# Mini HF: Modular Financial Forecasting & Portfolio Pipeline

Mini HF is a modular, end-to-end pipeline for financial forecasting, regime detection, ML-driven strategy backtesting, and portfolio optimization.  

---

## Features

- **Data Ingestion:** Download and clean historical price data for tickers specified in YAML config
- **Macro & Economic Data:** Fetch VIX index, inflation rates, and interest rates from FRED
- **Google Trends:** Collect search trend data for tickers and economic terms
- **Live Data Feed:** Real-time data fetching with automatic backfill capabilities
- **Feature Engineering:** Generate technical, macro, regime, and composite features dynamically from YAML config
- **Regime Detection:** Hidden Markov Model (HMM) and changepoint detection for market regime analysis
- **Model Training:** Flexible ML model training (classification/regression) with config-driven targets and features
- **Inference:** Live and batch inference using trained models and engineered features
- **Strategy Backtesting:** ML-driven and rule-based strategy backtesting with performance evaluation
- **Portfolio Optimization:** Risk metrics, constraints, and optimizer for multi-asset portfolios
- **Visualization:** Regime, performance, and allocation plots for analysis

---

## Data System

The pipeline includes three main data acquisition components:

### **Historical Data Ingestion** (`data_ingestion.py`)
- Downloads historical OHLCV data from Yahoo Finance
- Configurable date ranges and intervals (1d, etc.)
- Automated data cleaning and validation

### **Macro & Economic Data** (`macro_data.py`)
- **FRED Economic Indicators:**
  - Interest Rates (`FEDFUNDS`)
  - Inflation (`CPIAUCSL`) 
  - VIX Index (`VIXCLS`)
- **Google Trends Data:**
  - Search trends for economic terms and tickers
- All data sources configurable via YAML

### **Live Data Interface** (`live_feed_interface.py`)
- Real-time data fetching with smart backfill
- Continuous updates at specified intervals
- Handles missing data gaps automatically

---

## Regime Detection System

The regime detection system uses technical features and data from the data collection system to identify market regimes through Hidden Markov Models (HMM) and changepoint detection, capturing both gradual and sudden regime changes.

### **Hidden Markov Model** (`hidden_markov.py`)
- Uses technical features (returns, volatility, momentum, SMA, MACD, RSI) and macro data
- Identifies gradual regime transitions through probabilistic modeling
- Configurable number of states, covariance type, and iterations
- Outputs regime classifications and transition probabilities

### **Changepoint Detection** (`changepoint_detection.py`)
- Detects sudden structural breaks in market behavior using ruptures library
- Uses composite features from price, macro, and technical indicators
- Configurable penalty parameters and detection methods (PELT, BinSeg, Window)
- Identifies specific dates when market regimes shift abruptly

### **Regime Visualization** (`visualise_regimes.py`)
- Multi-panel plots showing returns with regime overlays
- Regime probability evolution over time
- Changepoint markers and uncertainty analysis
- Color-coded regime backgrounds for easy interpretation

### **Key Features**
- **Dual Approach**: Combines HMM (gradual changes) with changepoint detection (sudden changes)
- **Technical Integration**: Uses price-based features, volatility, momentum indicators
- **Macro Integration**: Incorporates economic indicators for regime context
- **Uncertainty Analysis**: Highlights periods of regime uncertainty
- **Robust Handling**: Manages missing data and edge cases gracefully

---

## Forecasting ML System

The forecasting ML system performs feature engineering, model training, inference, and ML-driven strategy backtesting using dynamically configurable targets and time horizons.

### **Feature Engineering** (`feature_engineering.py`)
- **Data Integration**: Combines price data with regime detection outputs, macro indicators, and trend data
- **Technical Features**: Generates SMA, MACD, RSI, momentum, volatility, drawdown, and Sharpe ratios
- **Lagged Features**: Creates time-series lags for returns, momentum, VIX, and regime probabilities
- **Composite Features**: Z-score normalization, rolling statistics, and regime confidence metrics
- **Target Generation**: Dynamically creates regression (`future_return_1d`) and classification (`future_return_1d_class`) targets
- **Data Cleaning**: Removes zero-variance, all-NaN, and constant columns with median imputation
- **Feature Scaling**: StandardScaler normalization for consistent model input

### **Model Training** (`model_training.py`)
- **Multi-Model Support**: Ridge, Random Forest, Gradient Boosting, SVM, XGBoost for both regression and classification
- **Dynamic Configuration**: Time horizons, target variables, and model parameters controlled via YAML
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit cross-validation
- **Feature Selection**: Automatic removal of non-numeric, constant, and leakage-prone features
- **Model Persistence**: Saves trained models, scalers, and feature columns as bundled pickle files
- **Performance Metrics**: Comprehensive evaluation (accuracy, precision, recall, F1, ROC-AUC for classification; RMSE, MAE, R² for regression)

### **Live Data Feature Engineering** (`live_data_feature_engineering.py`)
- **Real-time Processing**: Engineers features for live inference using the same pipeline as training
- **Consistent Scaling**: Uses fitted scalers from training for feature normalization
- **Missing Data Handling**: Fills missing regime/macro features with NaN for robust inference
- **Feature Alignment**: Ensures live features match training feature columns and order

### **Model Inference** (`model_inference.py`)
- **Dynamic Model Loading**: Automatically selects classification vs regression models based on target variable names
- **Live Prediction**: Processes latest live data through feature engineering and trained models
- **Multi-Ticker Support**: Handles inference for multiple assets simultaneously
- **Error Resilience**: Continues processing other tickers if individual predictions fail

### **ML Strategy Backtesting** (`ml_strategy.py`)
- **Walk-Forward Testing**: Trains models on expanding windows and tests on out-of-sample periods
- **Automatic Retraining**: Retrains models at configurable intervals (yearly by default)
- **All-In/All-Out Logic**: Binary position sizing based on classification predictions (1=buy, 0=sell)
- **Performance Tracking**: Records portfolio value, benchmark comparison, and allocation history
- **Rebalancing**: Daily rebalancing with configurable frequency
- **Comprehensive Metrics**: CAGR, Sharpe ratio, maximum drawdown, Sortino ratio, hit rate, tracking error

### **Key Features**
- **Config-Driven**: All parameters (features, models, horizons, rebalancing) controlled via YAML
- **Time Horizon Flexibility**: Currently 1-day predictions, easily configurable for longer horizons
- **Robust Pipeline**: Handles missing data, failed predictions, and edge cases gracefully
- **Production Ready**: Live inference capabilities with consistent feature engineering
- **Performance Analysis**: Detailed backtesting with allocation tracking and visualization

---

## Portfolio Optimization System

The portfolio optimization system calculates comprehensive risk metrics, applies flexible constraints, and optimizes portfolio allocations using multiple objective functions and covariance estimation methods.

### **Risk Metrics** (`risk_metrics.py`)
- **Core Risk Measures**: Volatility, Sharpe ratio, Sortino ratio, maximum drawdown, beta, and alpha
- **Rolling Metrics**: Time-varying risk analysis with configurable windows (default 63 days)
- **Covariance Estimation**: Multiple methods including sample, Ledoit-Wolf shrinkage, and EWMA
- **Benchmark Analysis**: Beta and alpha calculations against configurable benchmark (default SPY)
- **Data Integration**: Fetches price data from Yahoo Finance with configurable date ranges
- **Output Generation**: Saves risk metrics, covariance matrices, and price data for optimizer

### **Portfolio Constraints** (`constraints.py`)
- **Weight Bounds**: Per-asset minimum and maximum allocation limits
- **Long/Short Control**: Configurable long-only or long-short constraints
- **Weight Sum**: Enforces fully invested constraint (weights sum to 1)
- **Leverage Limits**: Optional leverage constraints for enhanced risk control
- **Default Bounds**: Fallback constraints when asset-specific bounds not specified
- **YAML Integration**: All constraints dynamically loaded from configuration

### **Portfolio Optimizer** (`optimizer.py`)
- **Multiple Objectives**: 
  - Maximum Sharpe ratio optimization
  - Minimum volatility (risk parity available)
  - Risk parity portfolio construction
- **Expected Returns Methods**:
  - Manual percentage returns
  - Manual target prices (calculates expected returns)
  - Historical mean returns (annualized)
- **Advanced Optimization**: Uses scipy.optimize with SLSQP method and constraint handling
- **Portfolio Analytics**: Complete performance metrics including Sortino, beta, alpha
- **Comprehensive Visualization**:
  - Portfolio weight pie charts
  - Efficient frontier plots
  - Risk-return scatter plots
  - Cumulative returns backtesting
  - Portfolio allocation over time
  - Correlation heatmaps
  - Constraint visualization

### **Key Features**
- **Multi-Currency Support**: Configurable capital amounts and currency display
- **Flexible Expected Returns**: Three methods for return estimation (manual, target price, historical)
- **Robust Covariance**: Multiple covariance estimation techniques for better risk modeling
- **Complete Backtesting**: Historical portfolio performance with benchmark comparison
- **Visual Analytics**: Comprehensive plotting suite for portfolio analysis
- **Configuration Driven**: All parameters controlled via YAML for easy customization

---

## Strategy Testing System

The strategy testing system provides comprehensive backtesting capabilities for multiple trading strategies with detailed performance evaluation, transaction costs, slippage modeling, and extensive visualization.

### **Strategy Definitions** (`definitions.py`)
- **Sharpe Ratio Rebalance Strategy**: 
  - Dynamic portfolio optimization using rolling mean returns
  - Maximum Sharpe ratio objective with configurable rebalancing frequency
  - Dynamic universe selection based on data availability
  - Weight constraints and lookback period configuration
- **Mean Reversion Strategy**:
  - Log-space trend analysis with standard deviation channels
  - Multi-level buying/selling based on distance from trend line
  - Cash management with gradual position building
  - Configurable channel boundaries and fill steps
- **Region Scaled Mean Reversion**:
  - Advanced mean reversion with region-specific multipliers
  - Percentage-based buying and profit-taking logic
  - Log-space price analysis with trend detrending
  - Sophisticated allocation scaling based on price regions

### **Backtester** (`backtester.py`)
- **Multi-Strategy Support**: Runs multiple strategies simultaneously with individual configurations
- **Transaction Cost Modeling**: Per-asset, per-day transaction costs with configurable rates
- **Slippage Simulation**: Random slippage within specified ranges for realistic execution costs
- **Data Cleaning Integration**: Uses shared data cleaning pipeline for consistent price data
- **Benchmark Handling**: Automatic benchmark data download and alignment
- **Synthetic Cash**: Adds synthetic cash positions for strategies requiring cash allocation
- **Results Export**: Saves portfolio values, weights, asset returns, and cost breakdowns

### **Performance Evaluator** (`performance_evaluator.py`)
- **Comprehensive Metrics**:
  - CAGR (Compound Annual Growth Rate)
  - Volatility and Sharpe Ratio
  - Sortino Ratio and Maximum Drawdown
  - Calmar Ratio and Hit Rate
  - Beta, Alpha, and Tracking Error vs benchmark
- **Cost Analysis**: Separates gross vs net performance to show impact of transaction costs
- **Advanced Visualizations**:
  - Portfolio value vs benchmark (log scale)
  - Final portfolio allocation pie charts
  - Drawdown curves over time
  - Rolling risk metrics (63-day windows)
  - Daily returns distribution histograms
  - Asset allocation evolution over time
  - Multi-strategy comparison plots

### **Key Features**
- **Realistic Cost Modeling**: Transaction costs and slippage applied per asset per day
- **Flexible Configuration**: All strategy parameters controlled via YAML
- **Multi-Currency Support**: Configurable capital amounts and currencies (GBP, USD, etc.)
- **Risk-Free Rate Integration**: Proper cash position handling with risk-free returns
- **Dynamic Asset Universe**: Strategies adapt to available assets at each rebalancing period
- **Comprehensive Logging**: Detailed progress tracking and error handling
- **Performance Attribution**: Gross vs net returns analysis showing cost impact

---

## Folder Structure

```
Mini HF/
├── data/
│   ├── config_data.yaml           # Data configuration (tickers, dates, intervals)
│   ├── data_ingestion.py          # Historical price data from Yahoo Finance
│   ├── macro_data.py              # FRED economic data + Google Trends
│   ├── live_feed_interface.py     # Real-time data fetching with backfill
│   ├── data_cleaning.py           # Shared data cleaning utilities
│   ├── historical_data/           # Historical CSV files
│   └── live_data/                 # Real-time data files
├── regime_detection/              # Market regime analysis
│   ├── hidden_markov.py           # HMM regime detection (gradual changes)
│   ├── changepoint_detection.py   # Structural break detection (sudden changes)
│   ├── visualise_regimes.py       # Regime and changepoint visualization
│   ├── markov_output/             # HMM regime classifications
│   ├── markov_prob_output/        # Regime transition probabilities
│   ├── changepoint_output/        # Detected changepoints and segments
│   └── hmm_models/                # Trained HMM models and scalers
├── forcasting_ml/                 # ML forecasting and strategy system
│   ├── feature_engineering.py     # Feature creation and data preprocessing
│   ├── live_data_feature_engineering.py  # Real-time feature engineering
│   ├── model_training.py          # ML model training with hyperparameter tuning
│   ├── model_inference.py         # Live prediction and batch inference
│   ├── ml_strategy.py             # Walk-forward strategy backtesting
│   ├── model_utils.py             # Model utilities and configuration helpers
│   ├── train_config.yaml          # ML training configuration
│   ├── ml_strategy_config.yaml    # Strategy backtesting parameters
│   ├── feature_engineering_output/ # Processed feature datasets
│   ├── trained_model/             # Model bundles with scalers and metadata
│   └── ml_strategy_data/          # Strategy backtest results and performance
├── portfolio_optimisation/        # Portfolio optimization and risk management
│   ├── risk_metrics.py            # Risk calculation and covariance estimation
│   ├── constraints.py             # Portfolio constraint management
│   ├── optimizer.py               # Multi-objective portfolio optimization
│   ├── portfolio.yaml             # Portfolio configuration and constraints
│   ├── risk_metrics_output/       # Risk metrics, covariance, and price data
│   ├── optimized_portfolio.csv    # Optimal portfolio weights and allocations
│   └── risk_metrics_rolling.csv   # Time-varying risk metrics
├── strategy_test/                 # Rule-based strategy backtesting and evaluation
│   ├── backtester.py              # Multi-strategy backtesting engine
│   ├── definitions.py             # Strategy implementation definitions
│   ├── performance_evaluator.py   # Performance metrics and visualization
│   ├── test_config.yaml           # Strategy parameters and configurations
│   └── results/                   # Backtest outputs and performance files
└── utils/                         # Logging and shared utilities
```

---

## Required Python Packages

- pandas>=1.5.0
- numpy>=1.21.0
- pyyaml>=6.0
- scikit-learn>=1.1.0
- ruptures>=1.1.0
- seaborn>=0.11.0
- matplotlib>=3.5.0
- scipy>=1.9.0
- xgboost>=1.6.0
- yfinance>=0.2.0
- pandas-datareader>=0.10.0
- pytrends>=4.9.0
- joblib>=1.1.0
- hmmlearn>=0.2.8

Install all dependencies with:
```
pip install -r requirements.txt
```
Or manually install each package with versions:
```
pip install pandas>=1.5.0 numpy>=1.21.0 pyyaml>=6.0 scikit-learn>=1.1.0 xgboost>=1.6.0 yfinance>=0.2.0 pandas-datareader>=0.10.0 pytrends>=4.9.0 matplotlib>=3.5.0 seaborn>=0.11.0 scipy>=1.9.0 hmmlearn>=0.2.8 ruptures>=1.1.0 joblib>=1.1.0
```

---

## Usage

Run the main pipeline:
```
python -m main
```

Optional flags:
- `--model` : Run model training step
- `--visualise` : Run regime visualisation step
- `--live` : Run live feed step
- `--inference` : Run model inference step
- `--portfolio` : Run portfolio risk, constraints, and optimizer
- `--strat` : Run strategy backtester and performance evaluator
- `--ml_strat` : Run ML strategy backtest

Example:
```
python -m main --model --inference --portfolio
```

---

## Configuration

All pipeline steps are controlled via YAML config files:
- `forcasting_ml/train_config.yaml` : Features, targets, model, CV, etc.
- `forcasting_ml/ml_strategy_config.yaml` : ML strategy backtest settings.
- `data/config_data.yaml` : Data sources, tickers, date range, etc.
- `portfolio_optimisation/portfolio.yaml` : Portfolio constraints and optimization settings.

---

## Logging

All modules use a central logger (`utils/logger.py`).  
Logs are printed to console for debugging and audit.

---

## Notes

- The pipeline is modular: you can run any step independently or chain them via the main script.
- All features, targets, and models are dynamically controlled via YAML config—no code changes needed for most workflow tweaks.
- If you encounter missing files or warnings, check the logs for details; the pipeline will fill missing features with NaN and continue where possible.

---

## License & Disclaimer

**MIT License** - Copyright (c) 2025 Amir Assadi

⚠️ **IMPORTANT**: This software is for educational purposes only. Not financial advice. Trading involves risk of loss. Use at your own risk. Always consult a qualified financial advisor.

---

## Authors

Developed by Amir Assadi.  
For questions or contributions, open an issue or pull request.

---

## Quick Start

1. **Clone the repository**
   ```
   git clone <repository-url>
   cd Mini-HF
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Configure your data sources**
   ```
   # Edit data/config_data.yaml to set your tickers and date ranges
   # Edit forcasting_ml/train_config.yaml for ML parameters
   ```

4. **Run the full pipeline**
   ```
   python -m main
   ```

5. **Or run specific components**
   ```
   python -m main --model --inference --portfolio
   ```

---

## Troubleshooting

### **Installation Issues**

**Package Installation Fails**
```
ERROR: Could not find a version that satisfies the requirement
```
- **Solution**: Upgrade pip first: `pip install --upgrade pip`
- Try installing packages individually: `pip install pandas numpy`
- For Windows: Use `pip install --user package_name`

**hmmlearn Installation Error**
```
Microsoft Visual C++ 14.0 is required
```
- **Solution**: Install Microsoft Visual C++ Build Tools
- Or use conda: `conda install -c conda-forge hmmlearn`

**pytrends Import Error**
```
ImportError: No module named 'pytrends'
```
- **Solution**: `pip install --upgrade pytrends`
- If still fails: `pip uninstall pytrends && pip install pytrends`

---

### **Data Issues**

**Yahoo Finance Data Fails**
```
No data found, symbol may be delisted
```
- **Solution**: Check ticker symbols are valid (use Yahoo Finance website)
- Try different date ranges (some tickers have limited history)
- Add delays between requests: modify `data_ingestion.py`

**FRED API Rate Limit**
```
HTTPError: 429 Too Many Requests
```
- **Solution**: Add time delays in `macro_data.py`
- Reduce number of macro indicators
- Run macro data collection separately first

**Google Trends API Error**
```
Retry.__init__() got an unexpected keyword argument 'method_whitelist'
```
- **Solution**: `pip install --upgrade pytrends requests`
- Pipeline continues with available data if trends fail
- Set `google_trends: false` in config to skip

**Missing Historical Data Files**
```
FileNotFoundError: data/historical_data/NVDA.csv not found
```
- **Solution**: Run data ingestion first: `python -m main` (no flags)
- Check `data/config_data.yaml` has correct tickers
- Ensure internet connection for data download

---

### **Regime Detection Issues**

**HMM Training Fails**
```
ValueError: 'covars' must be symmetric, positive-definite
```
- **Solution**: Reduce number of HMM states (try 2-3 states)
- Increase historical data period (use 3+ years)
- Check for sufficient price variation in data

**Changepoint Detection Error**
```
IndexError: list index out of range
```
- **Solution**: Ensure sufficient data points (minimum 100 days)
- Adjust penalty parameter in changepoint config
- Check for missing or NaN values in price data

**Empty Regime Files**
```
regime_detection/markov_output/ contains empty files
```
- **Solution**: Check logs for HMM training errors
- Verify feature engineering completed successfully
- Reduce complexity (fewer features, simpler model)

---

### **Machine Learning Issues**

**Feature Engineering Fails**
```
KeyError: 'regime' column not found
```
- **Solution**: Run regime detection first: `python -m main --visualise`
- Check regime files exist in `regime_detection/markov_output/`
- Pipeline creates placeholder files if missing

**Model Training Memory Error**
```
MemoryError: Unable to allocate array
```
- **Solution**: Reduce date range in config files
- Use fewer tickers for training
- Reduce feature lookback periods
- Close other applications to free RAM

**XGBoost GPU Error**
```
XGBoostError: GPU not available
```
- **Solution**: Install CPU version: `pip install xgboost`
- Or set `tree_method: 'hist'` in model config
- Use different model: change to `"random_forest"`

**Model Loading Error**
```
FileNotFoundError: Model bundle not found
```
- **Solution**: Run model training: `python -m main --model`
- Check `forcasting_ml/trained_model/` directory exists
- Verify model training completed without errors

---

### **Live Data & Inference Issues**

**Live Data Feed Fails**
```
ConnectionError: Failed to fetch live data
```
- **Solution**: Check internet connection
- Verify Yahoo Finance is accessible
- Reduce request frequency in live feed config

**Inference Prediction Fails**
```
ValueError: Feature shape mismatch
```
- **Solution**: Ensure live data has same features as training
- Run feature engineering on live data first
- Check for missing regime or macro data

**Real-time Updates Stop**
```
Live feed stopped updating
```
- **Solution**: Check logs for specific errors
- Restart live feed: `python -m main --live`
- Verify system time and market hours

---

### **Portfolio & Strategy Issues**

**Optimization Fails**
```
OptimizationError: Failed to converge
```
- **Solution**: Relax constraints in `portfolio.yaml`
- Use different covariance method (`sample` instead of `ledoit-wolf`)
- Reduce number of assets in portfolio

**Strategy Backtest Memory Issues**
```
Memory usage too high during backtesting
```
- **Solution**: Reduce backtest period
- Use fewer strategies simultaneously
- Increase rebalancing frequency (less frequent trades)

**Performance Metrics NaN**
```
Warning: Sharpe ratio is NaN
```
- **Solution**: Check for zero volatility periods
- Ensure sufficient non-zero returns
- Verify benchmark data is available

---

### **Configuration Issues**

**YAML Parsing Error**
```
YAMLError: could not determine a constructor
```
- **Solution**: Check YAML syntax (indentation, colons, quotes)
- Use YAML validator online
- Ensure no tabs (use spaces only)

**Config File Not Found**
```
FileNotFoundError: config file not found
```
- **Solution**: Ensure you're in the correct directory
- Check file paths are relative to project root
- Copy default config files if missing

---

### **Performance Optimization**

**Pipeline Runs Too Slowly**
- **Solution**: 
  - Reduce date ranges for testing
  - Use fewer tickers during development
  - Decrease feature lookback windows
  - Skip heavy computations (regime detection) during testing

**Memory Usage Too High**
- **Solution**:
  - Process tickers one at a time
  - Clear DataFrames after use: `del df`
  - Use pandas chunking for large datasets
  - Reduce model complexity

---

### **Common Workflow Issues**

**"Permission Denied" Errors**
- **Solution**: Close Excel/other programs using CSV files
- Run command prompt as administrator (Windows)
- Check file permissions

**Logs Show Warnings But Continue**
- **Expected Behavior**: Pipeline designed to continue with warnings
- **Action**: Review logs to understand what data is missing
- Non-critical warnings (missing trends, regime files) are normal

**Results Look Unrealistic**
- **Solution**: Check for data leakage in features
- Verify transaction costs are applied
- Review strategy logic for bugs
- Compare with benchmark performance


---
