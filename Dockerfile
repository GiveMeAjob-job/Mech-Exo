FROM python:3.11-slim AS runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies including dash extras and gunicorn
RUN pip install --no-cache-dir -e ".[dash]" gunicorn

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8050/healthz || exit 1

# Run gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8050", "mech_exo.reporting.dash_app:create_app()"]