FROM python:3.11-slim AS runtime

# Build args
ARG ENABLE_OPTUNA_DASHBOARD=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies including dash extras and gunicorn
RUN pip install --no-cache-dir -e ".[dash]" gunicorn

# Conditionally install Optuna dashboard if enabled
RUN if [ "$ENABLE_OPTUNA_DASHBOARD" = "true" ]; then \
        pip install --no-cache-dir optuna optuna-dashboard; \
    fi

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8050 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8050/healthz || exit 1

# Run gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8050", "mech_exo.reporting.dash_app:create_app()"]