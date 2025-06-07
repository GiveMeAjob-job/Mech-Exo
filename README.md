# Mech-Exo: Mechanical Exoskeleton Trading System

A systematic trading system designed to provide consistent, risk-managed returns through factor-based idea generation, position sizing, and automated execution.

## 🎯 Project Vision

| Goal              | KPI (12-mo rolling)                 |
| ----------------- | ----------------------------------- |
| Preserve capital  | Max draw-down ≤ 10% NAV            |
| Compound steadily | Sharpe > 0.6, CAGR > 15%           |
| Minimize friction | Fees + slippage ≤ 25% of gross P/L |
| Zero blow-ups     | No single trade loss > 2% NAV      |

## 🏗️ Architecture Overview

```text
                   ┌────────────────────┐
                   │   Data Pipeline    │  ✅ COMPLETED
                   └─────────┬──────────┘
                             ▼
┌─────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ Idea Scorer │─► │ Position Sizer     │─► │ Risk & Guard-Rails   │
└─────────────┘   └────────────────────┘   └────────────┬─────────┘
      ✅                    ✅                          ✅
                             ▲                         │
                             │                         │
                   ┌─────────┴──────────┐   ┌──────────┴──────────┐
                   │ Execution Engine   │   │  Reporting & UI     │
                   └────────────────────┘   └─────────────────────┘
                             🚧                         🚧
```

## 📁 Project Structure

```
mech_exo/
├── config/                 # Configuration files
│   ├── api_keys.yml       # API keys template
│   ├── risk_limits.yml    # Risk management config
│   └── factors.yml        # Factor scoring weights
├── data/                  # Data storage
│   ├── raw/              # Raw data downloads
│   └── processed/        # Processed data
├── mech_exo/             # Main package
│   ├── datasource/       # ✅ Data fetching & storage
│   ├── scoring/          # ✅ Factor-based idea scoring
│   ├── sizing/           # ✅ Position sizing
│   ├── risk/             # ✅ Risk management
│   ├── execution/        # 🚧 Trade execution
│   ├── reporting/        # 🚧 Dashboards & reports
│   ├── backtest/         # 🚧 Backtesting engine
│   └── utils/            # ✅ Utilities
├── dags/                 # ✅ Prefect pipelines
├── tests/                # ✅ Test suite
├── notebooks/            # ✅ Jupyter notebooks
└── scripts/              # ✅ Test scripts
```

## ✅ Phase P1: Data Pipeline (COMPLETED)

### Features Implemented:
- **OHLCDownloader**: Fetches price data from Yahoo Finance with retry logic and validation
- **FundamentalFetcher**: Retrieves fundamental data from yfinance and Finnhub APIs
- **NewsScraper**: Collects news and calculates sentiment scores
- **DataStorage**: DuckDB-based storage with comprehensive schema
- **Prefect Pipeline**: Automated nightly data collection with quality checks

### Key Components:
```python
from mech_exo.datasource import OHLCDownloader, DataStorage

# Initialize components
storage = DataStorage()
downloader = OHLCDownloader(config)

# Fetch and store data
data = downloader.fetch(['SPY', 'QQQ'], period="1mo")
storage.store_ohlc_data(data)
```

### Database Schema:
- `ohlc_data`: Price, volume, returns, volatility, ATR
- `fundamental_data`: P/E, ROE, revenue growth, debt ratios, etc.
- `news_data`: Headlines, sentiment scores, sources
- `universe`: Trading universe/watchlist
- `data_quality`: Data quality metrics and monitoring

## ✅ Phase P2: Idea Scorer (COMPLETED)

### Features Implemented:
- **IdeaScorer**: Main scoring engine with configurable factor weights
- **FactorFactory**: Modular factor calculation system
- **Multi-Factor Model**: Fundamental, technical, sentiment, and quality factors
- **Sector Adjustments**: Sector-specific scoring multipliers
- **Ranking System**: Composite scoring with percentile rankings

### Available Factors:
```yaml
fundamental:
  pe_ratio: {weight: 15, direction: lower_better}
  return_on_equity: {weight: 18, direction: higher_better}
  revenue_growth: {weight: 15, direction: higher_better}
  earnings_growth: {weight: 20, direction: higher_better}

technical:
  rsi_14: {weight: 8, direction: mean_revert}
  momentum_12_1: {weight: 12, direction: higher_better}
  volatility_ratio: {weight: 6, direction: lower_better}
```

### Usage Example:
```python
from mech_exo.scoring import IdeaScorer

# Initialize scorer
scorer = IdeaScorer()

# Score universe
ranking = scorer.rank_universe()

# Get top ideas
top_10 = scorer.get_top_ideas(n=10)
```

## ✅ Phase P3: Position Sizing & Risk Management (COMPLETED)

### Features Implemented:
- **PositionSizer**: Multi-method sizing (ATR, volatility, fixed %, Kelly criterion)
- **StopEngine**: Comprehensive stop-loss management (hard, trailing, time, volatility stops)
- **RiskChecker**: Real-time portfolio risk monitoring and violation detection
- **CLI Integration**: `exo risk status` command for risk checking

### Key Components:
```python
from mech_exo.sizing import PositionSizer, SizingMethod
from mech_exo.risk import RiskChecker, Portfolio, Position, StopEngine

# Initialize position sizer
sizer = PositionSizer(nav=100000)

# Calculate position size using ATR method
shares = sizer.calculate_size("AAPL", 150.0, method=SizingMethod.ATR_BASED, atr=2.0)

# Generate stop levels
stop_engine = StopEngine()
stops = stop_engine.generate_stops(entry_price=150.0, position_type="long", atr=2.0)

# Check portfolio risk
portfolio = Portfolio(100000)
portfolio.add_position(Position("AAPL", shares, 150.0, 155.0, datetime.now()))

checker = RiskChecker(portfolio)
risk_status = checker.get_risk_status_summary()
```

### Position Sizing Methods:
- **Fixed Percent**: Risk fixed percentage of NAV per trade
- **ATR-Based**: Size based on Average True Range for volatility adjustment  
- **Volatility-Based**: Inverse volatility targeting for consistent risk
- **Kelly Criterion**: Optimal sizing based on win rate and average win/loss

### Stop Loss Types:
- **Hard Stop**: Fixed percentage stop loss (15% default)
- **Trailing Stop**: Dynamic stop that trails price movements (25% default)
- **Profit Target**: Take profit level (30% default)
- **Time Stop**: Exit after specified time period (60 days default)
- **Volatility Stop**: ATR-based stop distance

### Risk Monitoring:
- **Position Limits**: Max single position size (10% NAV)
- **Portfolio Limits**: Gross/net exposure limits (150%/100%)
- **Sector Limits**: Max sector concentration (20%)
- **Drawdown Limits**: Maximum portfolio drawdown (10%)
- **Leverage Monitoring**: Real-time leverage calculation

## 🚧 Next Phases (Upcoming)

### Phase P4: Execution Engine
- **OrderRouter**: IB Gateway integration
- **BrokerInterface**: Paper and live trading modes
- **ExecutionTracker**: Fill tracking and slippage monitoring

### Phase P5: Reporting & Dashboard
- **DailySnapshot**: Performance reporting
- **Dash App**: Interactive dashboard
- **Alerting**: Slack/Telegram notifications

### Phase P6: Backtesting
- **EventBacktester**: Historical strategy testing
- **WalkForwardEvaluator**: Out-of-sample validation
- **Performance Analytics**: Risk-adjusted returns analysis

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd Mech-Exo

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy template and add your keys
cp config/api_keys.yml config/api_keys_local.yml
# Edit config/api_keys_local.yml with your actual API keys
```

### 3. Test Data Pipeline
```bash
python scripts/test_pipeline.py
```

### 4. Test Scoring System
```bash
python scripts/test_scoring.py
```

### 5. Test Position Sizing & Risk Management
```bash
python scripts/test_p3_flow.py
```

### 6. Check Risk Status
```bash
python -m mech_exo.cli risk status --nav 100000
```

### 7. Explore with Jupyter
```bash
jupyter notebook notebooks/scoring_exploration.ipynb
```

## 🧪 Testing

The project includes comprehensive tests for all implemented modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_datasource.py -v
python -m pytest tests/test_scoring.py -v
```

## 📊 Example Output

### Scoring Results:
```
🏆 Top Investment Ideas:
rank  symbol  composite_score  percentile  current_price  pe_ratio
1     AAPL    0.842           0.95        150.23         25.4
2     MSFT    0.731           0.85        280.45         28.1
3     GOOGL   0.695           0.75        125.67         22.8
```

### Data Pipeline Status:
```
✅ OHLC records: 2,500
✅ Fundamental records: 50
✅ News articles: 150
📊 Data quality: 95% complete
```

## 🔧 Configuration

### Risk Limits (`config/risk_limits.yml`):
```yaml
position_sizing:
  max_single_trade_risk: 0.02  # 2% of NAV per trade
  max_sector_exposure: 0.20    # 20% of NAV per sector

portfolio:
  max_gross_exposure: 1.5      # 150% of NAV (1.5x leverage)
  max_drawdown: 0.10           # 10% maximum drawdown
```

### Factor Weights (`config/factors.yml`):
Fully configurable factor weights and directions for scoring customization.

## 🛡️ Risk Management

- **Position-level**: Max 2% NAV risk per trade
- **Portfolio-level**: Max 10% drawdown
- **Sector limits**: Max 20% exposure per sector
- **Leverage limits**: Max 1.5x gross exposure
- **Stop losses**: 25% trailing stops

## 📈 Performance Targets

- **CAGR**: >15% annually
- **Sharpe Ratio**: >0.6
- **Max Drawdown**: <10%
- **Win Rate**: >55%
- **Cost Control**: <25% of gross P/L

## 🔮 Future Enhancements

- Machine learning factor discovery
- Alternative data integration
- Options strategy implementation
- Multi-asset support (crypto, forex)
- Real-time execution optimization

---

**Status**: Phase P1, P2 & P3 Complete ✅  
**Next**: Execution Engine (P4) 🚧  
**Timeline**: ~2 weeks per phase at current pace

Built with Python 3.11+ | DuckDB | Prefect | Interactive Brokers API