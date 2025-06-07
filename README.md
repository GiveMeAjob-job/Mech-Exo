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
                             ✅                         🚧
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
│   ├── execution/        # ✅ Trade execution
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

## ✅ Phase P4: Execution Engine (COMPLETED)

### Features Implemented:
- **OrderRouter**: Advanced order routing with pre-trade risk checks and retry logic
- **BrokerAdapter**: Pluggable broker interface supporting Interactive Brokers and StubBroker
- **SafetyValve**: Live-mode safety controls with CLI confirmation and sentinel orders
- **FillStore**: DuckDB-based execution database with timezone-aware storage
- **Structured Logging**: JSON-formatted execution monitoring with performance metrics
- **Integration Tests**: Comprehensive testing with 5/5 test scenarios passing

### Trading Modes:
The execution engine supports three distinct modes controlled by the `EXO_MODE` environment variable:

#### **Stub Mode** (Default - Offline Testing)
```bash
export EXO_MODE=stub  # or leave unset
python scripts/test_execution_integration.py
```
- **Purpose**: CI testing, development, and strategy verification
- **Broker**: Enhanced StubBroker with realistic market simulation
- **Features**: Instant fills, configurable slippage, commission simulation
- **Database**: Local DuckDB file for fill persistence
- **Safety**: No real money risk

#### **Paper Mode** (Live Market, Simulated Orders)
```bash
export EXO_MODE=paper
python -m mech_exo.execution.daily_executor
```
- **Purpose**: Live market testing with real prices but simulated execution
- **Broker**: Interactive Brokers Paper Trading account
- **Features**: Real market data, simulated fills, IB Gateway integration
- **Database**: Production DuckDB with full audit trail
- **Safety**: Paper trading account, no real money

#### **Live Mode** (Production Trading)
```bash
export EXO_MODE=live
python -m mech_exo.execution.daily_executor
```
- **Purpose**: Production trading with real money
- **Broker**: Interactive Brokers Live account
- **Features**: Real execution, full audit trail, safety valve protection
- **Database**: Production DuckDB with comprehensive logging
- **Safety**: CLI confirmation + sentinel orders + emergency abort

### Key Components:

#### **OrderRouter with Risk Integration**:
```python
from mech_exo.execution import OrderRouter
from mech_exo.execution.models import create_market_order
from mech_exo.risk import RiskChecker, Portfolio
from tests.stubs.broker_stub import EnhancedStubBroker

# Initialize components
broker = EnhancedStubBroker({'simulate_fills': True})
await broker.connect()

portfolio = Portfolio(100000)
risk_checker = RiskChecker(portfolio)

router = OrderRouter(broker, risk_checker, {
    'max_retries': 3,
    'safety': {
        'safety_mode': 'full_safety',  # CLI + sentinel orders
        'max_daily_value': 50000.0
    }
})

# Route order with full safety checks
order = create_market_order("AAPL", 100, strategy="momentum")
result = await router.route_order(order)

if result.decision.value == 'APPROVE':
    print(f"Order approved: {result.routing_notes}")
else:
    print(f"Order rejected: {result.rejection_reason}")
```

#### **SafetyValve for Live Trading**:
```python
from mech_exo.execution import SafetyValve

# Initialize safety valve
safety_valve = SafetyValve(broker, {
    'safety_mode': 'full_safety',
    'sentinel': {
        'symbol': 'CAD',
        'quantity': 100,
        'max_price': 1.50
    },
    'max_daily_value': 100000.0
})

# Authorize live trading (CLI confirmation + sentinel order)
authorized = await safety_valve.authorize_live_trading("Daily trading session")
if authorized:
    print("✅ Live trading authorized")
else:
    print("❌ Live trading blocked by safety valve")
```

#### **FillStore for Execution Tracking**:
```python
from mech_exo.execution import FillStore
from datetime import date

# Initialize fill store
fill_store = FillStore()  # Uses default path: data/execution.db

# Get today's fills
today_fills = fill_store.get_fills(date_filter=date.today())

# Get daily execution metrics
daily_metrics = fill_store.get_daily_metrics(date.today())
print(f"Today: {daily_metrics['fills']['total_fills']} fills")
print(f"Volume: ${daily_metrics['fills']['total_value']:,.0f}")
print(f"Avg slippage: {daily_metrics['fills']['avg_slippage_bps']:.1f} bps")

# Get last fill timestamp for session recovery
last_fill = fill_store.last_fill_ts()
```

### Environment Variables:

| Variable | Values | Description |
|----------|---------|-------------|
| `EXO_MODE` | `stub`, `paper`, `live` | Trading mode (default: `stub`) |
| `IB_GATEWAY_HOST` | IP address | IB Gateway host (default: `127.0.0.1`) |
| `IB_GATEWAY_PORT` | Port number | IB Gateway port (paper: `7497`, live: `7496`) |
| `EXECUTION_DB_PATH` | File path | DuckDB execution database path |
| `PREFECT_LOGGING_LEVEL` | `DEBUG`, `INFO`, `WARNING` | Prefect log level |

### Execution Database Schema:
```sql
-- Core fills table (timezone-aware)
CREATE TABLE fills (
    fill_id VARCHAR PRIMARY KEY,
    order_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    fill_time TIMESTAMPTZ NOT NULL,  -- UTC timezone-aware
    commission DECIMAL(8,4) DEFAULT 0,
    exchange VARCHAR,
    strategy VARCHAR,
    -- Performance metrics
    slippage_bps DECIMAL(8,4),
    execution_venue VARCHAR,
    liquidity_flag VARCHAR
);

-- Orders table for audit trail
CREATE TABLE orders (
    order_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    quantity INTEGER NOT NULL,
    order_type VARCHAR CHECK (order_type IN ('MKT', 'LMT', 'STP')),
    limit_price DECIMAL(10,4),
    stop_price DECIMAL(10,4),
    status VARCHAR CHECK (status IN ('PENDING', 'SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED')),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    strategy VARCHAR,
    broker_order_id VARCHAR
);
```

### Structured Logging Output:
```json
{
  "timestamp": "2025-06-07T21:54:35.488428+00:00",
  "level": "INFO",
  "logger": "mech_exo.execution.order_router",
  "message": "Fill received: AAPL 100 @ $175.06",
  "extra": {
    "event_type": "execution.fill",
    "session_id": "session_20250607_175435_e39a93bf",
    "trading_mode": "stub",
    "component": "order_router",
    "fill_id": "9455b669-bff8-4bc9-822e-c1e9dfc918cc",
    "order_id": "5acb3e65-6dea-4de4-a879-268d45181c81",
    "symbol": "AAPL",
    "quantity": 100,
    "price": 175.06,
    "commission": 1.5,
    "exchange": "STUB_EXCHANGE",
    "strategy": "test_logging"
  }
}
```

### Performance Metrics:
The execution engine tracks comprehensive performance metrics:
- **Order routing latency**: Time from signal to broker submission
- **Fill latency**: Time from order to fill
- **Slippage tracking**: Basis points of price impact
- **Commission costs**: Per-trade and daily totals
- **Execution quality**: Fill rate, rejection rate, retry statistics

### Error Handling & Recovery:
- **Retry Logic**: Configurable retry on gateway errors and timeouts
- **Circuit Breaker**: Emergency abort stops all trading immediately
- **Session Recovery**: Resume trading from last known state
- **Audit Trail**: Complete order and fill history for reconciliation

### Integration Testing Results:
```
📊 Integration Test Summary: 5/5 tests passed
✅ Verified Components:
  - EnhancedStubBroker with realistic simulation
  - OrderRouter with pre-trade risk checks  
  - FillStore persistence and daily metrics
  - Order rejection handling
  - End-to-end trading session flow
🚀 Ready for CI and production testing!
```

## 🚧 Next Phases (Upcoming)

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

### 7. Test Execution Engine
```bash
# Test with StubBroker (safe, offline)
export EXO_MODE=stub
python scripts/test_execution_integration.py

# Test OrderRouter + SafetyValve integration
python scripts/test_orderrouter_safety.py

# Test structured logging
python scripts/test_structured_logging.py
```

### 8. Paper Trading (30-Minute Setup)
```bash
# 1. Install IB Gateway (from Interactive Brokers)
# 2. Configure paper trading account
# 3. Start paper trading mode
export EXO_MODE=paper
export IB_GATEWAY_PORT=7497  # Paper trading port
python -m mech_exo.execution.daily_executor
```

### 9. Explore with Jupyter
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

# Test execution engine integration
python scripts/test_execution_integration.py
python scripts/test_orderrouter_safety.py
python scripts/test_structured_logging.py

# Run execution-specific pytest tests
python -m pytest tests/integration/test_execution.py -v -m execution
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

**Status**: Phase P1, P2, P3 & P4 Complete ✅  
**Next**: Reporting & Dashboard (P5) 🚧  
**Timeline**: ~2 weeks per phase at current pace

Built with Python 3.11+ | DuckDB | Prefect | Interactive Brokers API