# Dockerfile.ml - ML-enhanced image with LightGBM, XGBoost, and SHAP
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ML-specific dependencies
RUN pip install --no-cache-dir \
    lightgbm==4.3.0 \
    xgboost==2.0.3 \
    scikit-learn==1.4.0 \
    scipy==1.12.0 \
    shap==0.44.1 \
    optuna==3.5.0 \
    plotly==5.18.0 \
    kaleido==0.2.1

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/features data/processed data/raw models reports/shap

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for dashboard
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8050/healthz')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "mech_exo.cli", "dashboard"]