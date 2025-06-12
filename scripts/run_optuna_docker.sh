#!/bin/bash
# 
# Mech-Exo Optuna Docker Runner
# Convenient script for running hyperparameter optimization in Docker
#

set -e

# Default values
TRIALS=50
JOBS=2
STUDY_NAME="factor_weight_optimization"
COMPOSE_FILE="docker-compose.optuna.yml"
PROFILE=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo "Mech-Exo Optuna Docker Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --trials TRIALS     Number of optimization trials (default: 50)"
    echo "  -j, --jobs JOBS         Number of parallel jobs (default: 2)"
    echo "  -s, --study STUDY       Study name (default: factor_weight_optimization)"
    echo "  -d, --dashboard         Also start Optuna dashboard on port 8080"
    echo "  -c, --cleanup           Clean up containers and volumes after run"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults (50 trials, 2 jobs)"
    echo "  $0 -t 100 -j 4                       # Run 100 trials with 4 parallel jobs"
    echo "  $0 -t 20 -d                          # Run 20 trials and start dashboard"
    echo "  $0 -c                                 # Clean up previous runs"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--trials)
            TRIALS="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -s|--study)
            STUDY_NAME="$2"
            shift 2
            ;;
        -d|--dashboard)
            PROFILE="--profile dashboard"
            shift
            ;;
        -c|--cleanup)
            echo -e "${YELLOW}🧹 Cleaning up Docker containers and volumes...${NC}"
            docker compose -f $COMPOSE_FILE down --volumes --remove-orphans
            docker system prune -f
            echo -e "${GREEN}✅ Cleanup completed${NC}"
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Pre-flight checks
echo -e "${BLUE}🔍 Pre-flight checks...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Docker compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

# Create directories if they don't exist
mkdir -p studies factors data config

# Display configuration
echo -e "${GREEN}🚀 Starting Optuna optimization with Docker...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Trials: ${YELLOW}$TRIALS${NC}"
echo -e "  Parallel Jobs: ${YELLOW}$JOBS${NC}"
echo -e "  Study Name: ${YELLOW}$STUDY_NAME${NC}"
echo -e "  Compose File: ${YELLOW}$COMPOSE_FILE${NC}"

if [ -n "$PROFILE" ]; then
    echo -e "  Dashboard: ${GREEN}Enabled${NC} (http://localhost:8080)"
else
    echo -e "  Dashboard: ${YELLOW}Disabled${NC}"
fi

echo ""

# Build and run the optimization
echo -e "${BLUE}🔨 Building Docker image...${NC}"
docker compose -f $COMPOSE_FILE build optuna-runner

echo -e "${BLUE}🏃 Running optimization...${NC}"

# Override command with user parameters
export OPTUNA_COMMAND="python -m mech_exo.cli optuna-run --n-trials $TRIALS --n-jobs $JOBS --study-name $STUDY_NAME --study-file studies/factor_opt.db --no-stage --notify-progress --progress-interval 5"

# Run the services
if [ -n "$PROFILE" ]; then
    # Start both runner and dashboard
    docker compose -f $COMPOSE_FILE up --build
else
    # Start only the runner
    docker compose -f $COMPOSE_FILE up --build optuna-runner
fi

# Check results
echo ""
echo -e "${GREEN}🎯 Optimization completed!${NC}"

# Display results if available
if [ -f "studies/factor_opt.db" ]; then
    echo -e "${BLUE}📊 Study database created: studies/factor_opt.db${NC}"
    
    # Show study info using sqlite3
    echo -e "${BLUE}📈 Study summary:${NC}"
    sqlite3 studies/factor_opt.db "
        SELECT 
            'Study Name: ' || study_name,
            'Total Trials: ' || COUNT(*),
            'Best Value: ' || ROUND(MAX(value), 4)
        FROM trials 
        WHERE state = 2
        GROUP BY study_name;
    " 2>/dev/null || echo "  (Study summary not available)"
fi

# Display factor files
FACTOR_FILES=$(find factors/ -name "factors_opt_*.yml" -type f 2>/dev/null | head -5)
if [ -n "$FACTOR_FILES" ]; then
    echo -e "${BLUE}📁 Generated factor files:${NC}"
    echo "$FACTOR_FILES" | while read -r file; do
        echo "  $file"
    done
fi

echo ""
echo -e "${GREEN}✅ Docker optimization run completed successfully!${NC}"

# Next steps
echo -e "${YELLOW}🔍 Next steps:${NC}"
echo "  1. View results: exo dash --port 8050 (🔬 Optuna Monitor tab)"
echo "  2. Start dashboard: docker compose -f $COMPOSE_FILE up optuna-dashboard"
echo "  3. Export data: exo export --table optuna_trials --range last7d --fmt csv"

if [ -n "$PROFILE" ]; then
    echo ""
    echo -e "${GREEN}📊 Optuna Dashboard is running at: http://localhost:8080${NC}"
fi