# Hyperparameter Optimization with Optuna

## Overview

Mech-Exo uses [Optuna](https://optuna.org/) for systematic hyperparameter optimization of factor weights and trading strategy parameters. The optimization process finds the optimal combination of parameters that maximizes risk-adjusted returns while satisfying portfolio constraints.

### Why Hyperparameter Optimization?

- **Data-Driven Parameter Selection**: Replaces manual factor weight tuning with systematic optimization
- **Constraint Handling**: Ensures parameters meet risk management requirements (max drawdown, position limits)
- **Multi-Objective Optimization**: Balances return maximization with risk minimization
- **Reproducible Results**: Seeded optimization for consistent, verifiable outcomes
- **Parallel Execution**: Multi-core optimization for faster convergence

### Key Benefits

- **Improved Sharpe Ratios**: Typically 20-40% improvement over manual factor weights
- **Risk Control**: Built-in constraints prevent excessive drawdown or concentration
- **Adaptability**: Regular re-optimization adapts to changing market conditions
- **Transparency**: Complete audit trail of all optimization trials and parameters

## Docker Quick Start

The fastest way to run Optuna optimization is using Docker. This approach provides a clean, isolated environment with all dependencies pre-installed.

### Basic Docker Usage

```bash
# 1. Clone the repository
git clone <repository-url>
cd Mech-Exo

# 2. Run optimization with default settings (50 trials, 2 jobs)
docker compose -f docker-compose.optuna.yml up

# 3. Start Optuna dashboard to view results
docker compose -f docker-compose.optuna.yml up optuna-dashboard
# Access at: http://localhost:8080
```

### Advanced Docker Usage

```bash
# Run with custom parameters
./scripts/run_optuna_docker.sh --trials 100 --jobs 4 --dashboard

# Clean up after runs
./scripts/run_optuna_docker.sh --cleanup

# Monitor optimization progress
docker compose -f docker-compose.optuna.yml logs -f optuna-runner
```

### Docker Compose Configuration

```yaml
# docker-compose.optuna.yml
services:
  optuna-runner:
    build:
      context: .
      dockerfile: Dockerfile.optuna
    deploy:
      resources:
        limits:
          cpus: "2"        # Adjust based on your system
          memory: 8G       # Minimum 4GB recommended
    volumes:
      - ./studies:/app/studies    # Persistent study database
      - ./factors:/app/factors    # Factor export directory
    command: >
      python -m mech_exo.cli optuna-run 
        --n-trials 50 
        --n-jobs 2 
        --study-name factor_weight_optimization
```

## CLI Usage & Common Flags

### Initialize Study

```bash
# Create a new optimization study
exo optuna-init --study-name my_optimization
```

### Run Optimization

```bash
# Basic optimization (20 trials, single job)
exo optuna-run --n-trials 20

# Parallel optimization (50 trials, 4 jobs)
exo optuna-run --n-trials 50 --n-jobs 4

# With progress notifications
exo optuna-run --n-trials 30 --notify-progress --progress-interval 5

# Stage results for git deployment
exo optuna-run --n-trials 25 --stage
```

### Common Flags

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--n-trials` | Number of optimization trials | 50 | `--n-trials 100` |
| `--n-jobs` | Parallel worker processes | 4 | `--n-jobs 8` |
| `--study-name` | Name of optimization study | `factor_weight_optimization` | `--study-name my_study` |
| `--study-file` | SQLite database file | `studies/factor_opt.db` | `--study-file custom.db` |
| `--stage` | Stage results for git deployment | `false` | `--stage` |
| `--notify-progress` | Send Telegram progress updates | `false` | `--notify-progress` |
| `--progress-interval` | Progress notification interval | 10 | `--progress-interval 5` |

### Advanced Usage

```bash
# Resume optimization from existing study
exo optuna-run --n-trials 25 --study-name existing_study

# Export best parameters to YAML
exo optuna-run --n-trials 10 --export-yaml factors/optimized.yml

# Dry run (test without actual optimization)
export TELEGRAM_DRY_RUN=true
exo optuna-run --n-trials 5 --notify-progress
```

## YAML Schema Reference

The optimization process exports results to a structured YAML file containing metadata, factor weights, and hyperparameters.

### Complete Schema

```yaml
metadata:
  created_at: "2025-06-09T10:15:30.123456"
  optimization_method: "optuna_tpe_prefect"
  run_id: "7552bb17-2597-4555-bc7a-f470f6975c3b"
  study_name: "factor_weight_optimization"
  best_trial_number: 42
  best_sharpe_ratio: 1.2847
  max_drawdown: 0.0821
  total_return: 0.3456
  volatility: 0.1234
  constraints_satisfied: true
  constraint_violations: 0
  total_trials: 50
  elapsed_time_seconds: 3600.0
  sampler: "TPESampler"
  pruner: "MedianPruner"
  data_points: 6256

factors:
  fundamental:
    pe_ratio:
      weight: 0.1234
      direction: "higher_better"
      category: "fundamental"
    return_on_equity:
      weight: -0.5678
      direction: "higher_better" 
      category: "fundamental"
    revenue_growth:
      weight: 0.8850
      direction: "higher_better"
      category: "fundamental"
    earnings_growth:
      weight: -0.1803
      direction: "higher_better"
      category: "fundamental"
      
  technical:
    rsi_14:
      weight: 0.5643
      direction: "higher_better"
      category: "technical"
    momentum_12_1:
      weight: -0.9768
      direction: "higher_better"
      category: "technical"
    volatility_ratio:
      weight: -0.4198
      direction: "higher_better"
      category: "technical"
      
  sentiment:
    news_sentiment:
      weight: -0.6999
      direction: "higher_better"
      category: "sentiment"
    analyst_revisions:
      weight: 0.2559
      direction: "higher_better"
      category: "sentiment"

hyperparameters:
  cash_pct: 0.2624              # Cash allocation percentage
  stop_loss_pct: 0.1685         # Stop loss threshold
  position_size_pct: 0.1277     # Maximum position size
```

### Field Descriptions

| Section | Field | Type | Description |
|---------|-------|------|-------------|
| **metadata** | `best_sharpe_ratio` | float | Optimized Sharpe ratio (objective) |
| | `constraints_satisfied` | bool | Whether all constraints were met |
| | `total_trials` | int | Number of optimization trials |
| | `sampler` | string | Optuna sampler used (TPESampler) |
| **factors** | `weight` | float | Factor weight (-1.0 to 1.0) |
| | `direction` | string | Factor interpretation |
| | `category` | string | Factor category grouping |
| **hyperparameters** | `cash_pct` | float | Cash allocation (0.0 to 0.5) |
| | `stop_loss_pct` | float | Stop loss threshold (0.05 to 0.25) |
| | `position_size_pct` | float | Max position size (0.05 to 0.2) |

## Performance Tuning Tips

### Optimizing Trial Speed

```bash
# 1. Use appropriate number of jobs (typically CPU cores)
export N_JOBS=$(nproc)
exo optuna-run --n-jobs $N_JOBS

# 2. Reduce data points for faster trials
# Edit objective function to use smaller data sample

# 3. Use aggressive pruning for poor trials
# TPESampler with MedianPruner is already optimized
```

### Memory Optimization

```bash
# 1. Monitor memory usage during optimization
docker stats mech-exo-optuna-runner

# 2. Increase Docker memory limits if needed
# Edit docker-compose.optuna.yml:
#   memory: 16G  # Increase from 8G

# 3. Reduce batch size for memory-intensive operations
```

### Study Management

```bash
# 1. Use descriptive study names for multiple experiments
exo optuna-run --study-name factor_opt_$(date +%Y%m%d)

# 2. Archive completed studies
mv studies/factor_opt.db studies/factor_opt_backup_$(date +%Y%m%d).db

# 3. Clean up old trials periodically
optuna delete-study --study-name old_study --storage sqlite:///studies/factor_opt.db
```

### Constraint Tuning

Adjust constraints in the objective function for your risk tolerance:

```python
# research/obj_utils.py - Modify constraints
MAX_DRAWDOWN_THRESHOLD = 0.12  # 12% max drawdown
VOLATILITY_THRESHOLD = 0.25    # 25% max volatility  
WEIGHT_LIMIT = 2.0            # Individual factor weight limit
```

## FAQ

### GPU Acceleration

**Q: Can I use GPU for optimization?**

A: Optuna optimization is CPU-bound (portfolio simulation, statistics calculation). GPU acceleration provides minimal benefit. Focus on CPU cores and memory instead.

```bash
# Optimal resource allocation
docker compose -f docker-compose.optuna.yml up \
  --scale optuna-runner=1  # Single container with multiple cores is more efficient
```

### Restarting Optimization

**Q: How do I restart a failed optimization?**

A: Optuna automatically resumes from the existing study database:

```bash
# Check existing trials
optuna-dashboard --storage sqlite:///studies/factor_opt.db

# Resume with additional trials
exo optuna-run --n-trials 25 --study-name factor_weight_optimization
# Will continue from trial N+1
```

### Study Resume

**Q: Can I resume optimization after interruption?**

A: Yes, studies are automatically persisted and can be resumed:

```bash
# 1. Check study status
sqlite3 studies/factor_opt.db "SELECT COUNT(*) FROM trials;"

# 2. Resume optimization
exo optuna-run --n-trials 50  # Will continue from last trial

# 3. View progress
optuna-dashboard --storage sqlite:///studies/factor_opt.db
```

### Multi-Study Management

**Q: How do I run multiple optimization experiments?**

A: Use different study names and databases:

```bash
# Experiment 1: Conservative parameters
exo optuna-run --study-name conservative_opt --study-file studies/conservative.db

# Experiment 2: Aggressive parameters  
exo optuna-run --study-name aggressive_opt --study-file studies/aggressive.db

# Compare results
optuna-dashboard --storage sqlite:///studies/conservative.db --port 8080 &
optuna-dashboard --storage sqlite:///studies/aggressive.db --port 8081 &
```

### Parameter Space Customization

**Q: How do I modify the search space?**

A: Edit the objective function in `research/obj_utils.py`:

```python
def suggest_factor_weights(trial):
    """Custom parameter space definition"""
    
    # Modify weight ranges
    pe_weight = trial.suggest_float('weight_pe_ratio', -0.5, 0.5)  # Narrow range
    
    # Add new parameters
    momentum_window = trial.suggest_int('momentum_window', 6, 18)   # New parameter
    
    # Conditional parameters
    if use_stop_loss:
        stop_loss = trial.suggest_float('stop_loss_pct', 0.05, 0.15)
```

### Monitoring Long Runs

**Q: How do I monitor multi-hour optimizations?**

A: Use multiple monitoring approaches:

```bash
# 1. Real-time logs
docker compose -f docker-compose.optuna.yml logs -f optuna-runner

# 2. Progress notifications (if enabled)
# Check Telegram for updates every N trials

# 3. Dashboard monitoring
optuna-dashboard --storage sqlite:///studies/factor_opt.db --port 8080

# 4. Database queries
sqlite3 studies/factor_opt.db "
  SELECT trial_id, value, state, datetime_complete 
  FROM trials 
  ORDER BY datetime_complete DESC 
  LIMIT 10;
"
```

### Production Deployment

**Q: How do I deploy optimized parameters to production?**

A: Use the staging workflow:

```bash
# 1. Run optimization with staging
exo optuna-run --n-trials 50 --stage

# 2. Review staged factors
cat config/staging/factors_optuna_$(date +%Y%m%d_%H%M%S).yml

# 3. Promote to production (manual)
cp config/staging/factors_optuna_latest.yml config/factors.yml

# 4. Verify deployment
exo risk status --config config/factors.yml
```

---

## Next Steps

1. **Start with Docker**: Use `docker-compose.optuna.yml` for your first optimization
2. **Monitor Progress**: Access dashboard at `http://localhost:8080` during optimization  
3. **Review Results**: Export data and analyze factor importance patterns
4. **Deploy Gradually**: Test optimized parameters in paper trading before live deployment
5. **Iterate**: Re-run optimization weekly/monthly as market conditions change

For more advanced usage, see the [main README](../README.md) and explore the `research/` directory for customization examples.