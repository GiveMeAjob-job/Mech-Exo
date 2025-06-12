# ML Factor Training Pipeline

## Overview

The ML Factor training pipeline uses LightGBM and XGBoost to create predictive alpha signals based on engineered features from price, fundamental, and sentiment data.

## Training Pipeline

### Quick Start

```bash
# Install ML dependencies
pip install lightgbm scikit-learn scipy

# Build features (prerequisite)
exo ml-features --start 2022-01-01 --end 2025-01-01

# Train model with default settings
exo ml-train --algo lightgbm --lookback 3y --cv 5

# Train with custom parameters
exo ml-train --algo xgboost --lookback 1y --cv 3 --n-iter 20 --seed 42
```

### Command Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--algo` | `lightgbm`, `xgboost` | `lightgbm` | Algorithm to use |
| `--lookback` | `3y`, `1y`, `180d`, `90d` | `3y` | Training data lookback period |
| `--cv` | Integer | `5` | Number of cross-validation folds |
| `--n-iter` | Integer | `30` | Hyperparameter search iterations |
| `--seed` | Integer | `42` | Random seed for reproducibility |
| `--features-dir` | Path | `data/features` | Directory containing feature files |
| `--models-dir` | Path | `models` | Directory to save models and metrics |
| `--verbose` | Flag | `false` | Enable verbose logging |

### Output Files

The training pipeline produces three main outputs:

#### 1. Model Files
- **LightGBM**: `models/lgbm_YYYYMMDD_HHMMSS.txt`
- **XGBoost**: `models/xgb_YYYYMMDD_HHMMSS.json`

#### 2. Metrics File
`models/metrics_ALGO_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "2025-06-09T16:08:07.123456",
  "algorithm": "lightgbm",
  "best_auc": 0.7234,
  "best_params": {
    "num_leaves": 63,
    "learning_rate": 0.1,
    "max_depth": 7
  },
  "metrics": {
    "mean_auc": 0.7123,
    "std_auc": 0.0234,
    "mean_ic": 0.0456,
    "std_ic": 0.0123,
    "mean_accuracy": 0.6789,
    "std_accuracy": 0.0345
  },
  "training_samples": 1250,
  "features": 22,
  "cv_folds": 5,
  "n_iter": 30
}
```

#### 3. Success Criteria
- **AUC ≥ 0.60**: Model meets minimum predictive performance
- **IC > 0.0**: Positive information coefficient indicates signal quality
- **No overfitting**: Consistent performance across CV folds

### Algorithm Comparison

| Feature | LightGBM | XGBoost |
|---------|----------|---------|
| **Speed** | Faster training | Moderate speed |
| **Memory** | Lower memory usage | Higher memory usage |
| **Accuracy** | Excellent | Excellent |
| **Categorical Features** | Native support | Requires encoding |
| **GPU Support** | Yes | Yes |
| **Model Size** | Smaller files | Larger files |

### Hyperparameter Search Space

#### LightGBM Parameters
- `num_leaves`: [31, 63, 127, 255]
- `max_depth`: [3, 5, 7, 10, -1]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `n_estimators`: [100, 200, 500, 1000]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]
- `reg_alpha`: [0, 0.1, 0.5, 1.0]
- `reg_lambda`: [0, 0.1, 0.5, 1.0]

#### XGBoost Parameters
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `n_estimators`: [100, 200, 500, 1000]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]
- `gamma`: [0, 0.1, 0.5, 1.0]
- `reg_alpha`: [0, 0.1, 0.5, 1.0]
- `reg_lambda`: [1, 1.5, 2.0]

### Performance Metrics

#### Primary Metrics
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between positive and negative returns
- **IC (Information Coefficient)**: Spearman correlation between predictions and forward returns
- **Accuracy**: Percentage of correct binary predictions

#### Cross-Validation
- **Time Series Split**: Respects temporal order of data
- **Forward-Looking Labels**: 10-day forward returns to avoid look-ahead bias
- **Fold Statistics**: Mean ± standard deviation across CV folds

### Troubleshooting

#### Common Issues

**1. Insufficient Training Data**
```
Error: Insufficient training data: 50 samples. Need at least 100.
```
- **Solution**: Extend `--lookback` period or build more feature files

**2. Low AUC Performance**
```
⚠️  Model AUC below threshold (<0.60)
```
- **Solution**: 
  - Increase `--n-iter` for better hyperparameter search
  - Add more features via feature engineering
  - Check data quality and label distribution

**3. Memory Issues**
```
MemoryError: Unable to allocate array
```
- **Solution**:
  - Use smaller `--lookback` period
  - Reduce `--cv` folds
  - Switch from XGBoost to LightGBM

#### Installation Issues

**LightGBM Installation**
```bash
# Standard installation
pip install lightgbm

# If wheel not available (compile from source)
pip install lightgbm --no-binary lightgbm

# For Docker (install dependencies)
apt-get install -y libomp-dev
```

**XGBoost Installation**
```bash
# Standard installation
pip install xgboost

# For GPU support
pip install xgboost[gpu]
```

### Example Output

```bash
$ exo ml-train --algo lightgbm --lookback 90d --cv 2 --n-iter 5 --verbose

🚀 Starting ML model training...
   Algorithm: lightgbm
   Lookbook: 90d
   CV folds: 2
   Hyperparameter iterations: 5
   Features dir: data/features
   Models dir: models
   Random seed: 42

INFO: 🚀 Starting ML training pipeline...
INFO: Building features from 2024-12-11 to 2025-06-09
INFO: Loading data sources...
INFO: Loaded 325 feature files
INFO: Total feature records: 1,625
INFO: Unique symbols: 5
INFO: Date range: 2024-12-11 to 2025-03-29

INFO: Preparing training data with 10-day forward returns
INFO: Training data prepared: 1,250 samples, 13 features
INFO: Label distribution: {0: 625, 1: 625}

INFO: Starting lightgbm training with 2-fold CV
INFO: Fold 1/2: train=625, val=625
INFO: Fold 2/2: train=625, val=625

INFO: Starting hyperparameter search...
Fitting 2 folds for each of 5 candidates, totalling 10 fits
INFO: Best CV AUC: 0.7234

✅ ML training completed successfully!
   Best AUC: 0.7234
   Training samples: 1,250
   Features: 13
   Model saved: models/lgbm_20250609_160807.txt
   Metrics saved: models/metrics_lightgbm_20250609_160807.json

📊 Performance Metrics:
   Mean AUC: 0.7123 ± 0.0234
   Mean IC: 0.0456 ± 0.0123
   Mean Accuracy: 0.6789 ± 0.0345

🎯 Model meets AUC threshold (≥0.60)!
```

## Daily Inference

### Quick Start

```bash
# Generate ML predictions for today
exo ml-predict --model models/lgbm_20250609_160807.txt \
               --date 2025-06-11 \
               --symbols SPY QQQ IWM \
               --outfile ml_scores.csv

# Integrate ML scores with idea ranking
exo score --use-ml --ml-scores ml_scores.csv --output idea_scores_with_ml.csv

# Or let IdeaScorer load ML scores from database automatically
exo score --use-ml --output final_rankings.csv
```

### ML Prediction Command

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | File path | Required | Path to trained model file |
| `--date` | `YYYY-MM-DD`, `today` | `today` | Target date for prediction |
| `--symbols` | Comma-separated | All available | Symbols to predict |
| `--features-dir` | Directory path | `data/features` | Feature files directory |
| `--outfile` | File path | `ml_scores.csv` | Output CSV file |
| `--normalize` | Flag | `True` | Min-max normalize scores 0-1 |
| `--verbose` | Flag | `False` | Enable verbose logging |

### IdeaScorer Integration

The IdeaScorer now supports ML integration with the `--use-ml` flag:

```bash
# Traditional scoring only
exo score --symbols AAPL MSFT GOOGL --output traditional_scores.csv

# ML-enhanced scoring (30% ML weight by default)
exo score --use-ml --symbols AAPL MSFT GOOGL --output ml_enhanced_scores.csv

# Custom ML scores file
exo score --use-ml --ml-scores custom_ml_scores.csv --output final_scores.csv
```

#### ML Weight Configuration

The ML integration weight is controlled by the `ml_weight` parameter in `config/factors.yml`:

```yaml
# ML integration settings
ml_weight: 0.3  # 30% ML, 70% traditional factors

# Factor weights (traditional)
fundamental:
  pe_ratio:
    weight: 25
    direction: lower_better
  # ... other factors
```

#### Score Blending Formula

```python
final_score = (1 - ml_weight) * traditional_rank + ml_weight * ml_rank
```

### Prefect Flow Integration

The daily ML inference runs automatically via Prefect at **09:40 UTC**:

#### Flow: `ml-daily-inference`

1. **`load_latest_model()`**: Finds most recent trained model
2. **`build_features_today()`**: Generates features for current date
3. **`infer_scores()`**: Generates ML predictions, saves to CSV
4. **`store_ml_scores()`**: Stores predictions in DuckDB `ml_scores` table
5. **`update_idea_scores()`**: Runs IdeaScorer with ML integration

```python
# Manual flow execution
from dags.ml_inference_flow import ml_daily_inference_flow

result = ml_daily_inference_flow(
    symbols=['SPY', 'QQQ', 'IWM'],  # Optional symbol filter
    models_dir="models",
    features_dir="data/features"
)
```

### Output Files and Database Storage

#### CSV Outputs
- **`ml_scores_daily.csv`**: Raw ML predictions with normalization
- **`idea_scores_with_ml.csv`**: Final rankings with ML integration

#### Database Tables

**`ml_scores` table**:
```sql
CREATE TABLE ml_scores (
    symbol VARCHAR,
    ml_score FLOAT,
    prediction_date DATE,
    model_path VARCHAR,
    algorithm VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (symbol, prediction_date)
);
```

**Enhanced `idea_scores` with ML columns**:
```csv
rank,symbol,composite_score,ml_rank,final_score,ml_weight_used,uses_ml
1,AAPL,85.2,2,83.8,0.3,True
2,MSFT,82.1,1,84.2,0.3,True
3,GOOGL,78.9,3,76.1,0.3,True
```

### Performance and Optimization

#### Batch Processing
- **1k+ symbols**: Automatically batched in chunks of 1,000
- **Memory optimization**: Features loaded incrementally
- **Speed**: <5ms inference for 20-symbol test matrix

#### Model Wrapper Benefits
- **Unified API**: Same interface for LightGBM and XGBoost
- **Auto-detection**: Algorithm detected from file extension
- **Missing value handling**: Forward-fill and zero-fill fallbacks
- **Feature alignment**: Automatic column reordering for model compatibility

### Error Handling and Troubleshooting

#### Common Issues

**1. Missing Model File**
```
❌ File not found: models/lgbm_model.txt
```
- **Solution**: Ensure model exists and path is correct
- **Check**: `ls models/` for available models

**2. Feature Mismatch**
```
⚠️ Missing features: ['fundamental_pe_ratio', 'sentiment_score']
```
- **Solution**: Features automatically added as zeros
- **Impact**: May reduce prediction quality

**3. No ML Scores in Database**
```
⚠️ ML scores not available - using traditional scores only
```
- **Solution**: Run `exo ml-predict` first or provide `--ml-scores` file
- **Fallback**: IdeaScorer continues with traditional factors only

#### Performance Monitoring

```bash
# Check recent ML predictions
exo ml-predict --model models/latest_model.txt --verbose

# Verify database storage
sqlite3 data/mech_exo.duckdb "SELECT * FROM ml_scores ORDER BY prediction_date DESC LIMIT 5"

# Compare traditional vs ML-enhanced scores
exo score --output traditional.csv
exo score --use-ml --output ml_enhanced.csv
```

### Integration Examples

#### Daily Workflow
```bash
# 1. Generate features (automated via Prefect)
exo ml-features --start 2025-06-11 --end 2025-06-11

# 2. Generate ML predictions
exo ml-predict --model models/lgbm_latest.txt --date 2025-06-11

# 3. Generate enhanced rankings
exo score --use-ml --top 20 --output top_ideas.csv

# 4. View results
cat top_ideas.csv | head -10
```

#### Custom ML Weight Testing
```bash
# Test different ML weights by updating config/factors.yml
echo "ml_weight: 0.5" >> config/factors.yml  # 50% ML weight
exo score --use-ml --output ml_50pct.csv

echo "ml_weight: 0.1" >> config/factors.yml  # 10% ML weight  
exo score --use-ml --output ml_10pct.csv

# Compare impact on rankings
diff ml_50pct.csv ml_10pct.csv
```

### Next Steps

1. **Dashboard Integration**: Add ML signals tab (Day 5)
2. **Performance Monitoring**: Track model decay and retrain triggers
3. **Feature Engineering**: Expand feature set based on SHAP insights
4. **Ensemble Models**: Combine multiple ML algorithms

---

For advanced usage and integration examples, see the main [README](../README.md).