# Mech-Exo Trading System Makefile

.PHONY: help install test dash dash-build dash-run dash-stop docker-up docker-down clean

help:  ## Show this help message
	@echo "Mech-Exo Trading System"
	@echo "======================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

install-dash:  ## Install with dashboard dependencies
	pip install -e ".[dash]"

test:  ## Run tests
	pytest tests/

dash:  ## Run dashboard locally (development)
	python -m mech_exo.cli dash --port 8050 --debug

dash-build:  ## Build dashboard Docker image
	docker build -t mech-exo-dash .

dash-run:  ## Run dashboard in Docker container
	docker run -d --name mech-exo-dash -p 8050:8050 --env-file .env mech-exo-dash

dash-stop:  ## Stop dashboard container
	docker stop mech-exo-dash || true
	docker rm mech-exo-dash || true

docker-up:  ## Start all services with Docker Compose
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## View dashboard logs
	docker compose logs -f dash

docker-build:  ## Build services with Docker Compose
	docker compose build

clean:  ## Clean up containers and images
	docker compose down --volumes --remove-orphans
	docker system prune -f

# Development helpers
dev-setup:  ## Set up development environment
	cp .env.example .env
	mkdir -p data/processed data/raw
	make install-dash

lint:  ## Run code linting
	ruff check mech_exo/
	black --check mech_exo/

format:  ## Format code
	black mech_exo/
	ruff check --fix mech_exo/