#!/bin/bash
#
# Download Optuna study database from object storage
# Useful for retrieving archived optimization studies
#

set -e

# Configuration
STUDY_NAME=""
BACKUP_URL=""
OUTPUT_DIR="studies"
FORCE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo "Download Optuna Study Database"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --study STUDY       Study name to download"
    echo "  -u, --url URL           Backup URL (S3/GCS/HTTP endpoint)"
    echo "  -o, --output DIR        Output directory (default: studies)"
    echo "  -f, --force             Overwrite existing files"
    echo "  -l, --list              List available studies"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -s factor_opt_20250101 -u s3://bucket/studies/"
    echo "  $0 -l                                             # List available studies"
    echo "  $0 -s factor_opt_latest -f                       # Download latest, overwrite"
    echo ""
    echo "Environment Variables:"
    echo "  STUDY_BACKUP_URL        Default backup URL"
    echo "  AWS_PROFILE             AWS profile for S3 access"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--study)
            STUDY_NAME="$2"
            shift 2
            ;;
        -u|--url)
            BACKUP_URL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -l|--list)
            echo -e "${BLUE}📋 Listing available studies...${NC}"
            # This would be implemented based on your object storage
            echo "Available studies:"
            echo "  factor_opt_20250608.db (Latest)"
            echo "  factor_opt_20250601.db"
            echo "  factor_opt_20250525.db"
            echo ""
            echo "Use --study STUDY_NAME to download a specific study"
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

# Use environment variable as fallback
if [ -z "$BACKUP_URL" ] && [ -n "$STUDY_BACKUP_URL" ]; then
    BACKUP_URL="$STUDY_BACKUP_URL"
fi

# Validation
if [ -z "$STUDY_NAME" ]; then
    echo -e "${RED}❌ Study name is required. Use -s/--study option.${NC}"
    show_help
    exit 1
fi

if [ -z "$BACKUP_URL" ]; then
    echo -e "${RED}❌ Backup URL is required. Use -u/--url option or set STUDY_BACKUP_URL.${NC}"
    show_help
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine file names
STUDY_FILE="${STUDY_NAME}.db"
if [[ ! "$STUDY_FILE" == *.db ]]; then
    STUDY_FILE="${STUDY_NAME}.db"
fi

LOCAL_PATH="${OUTPUT_DIR}/${STUDY_FILE}"
REMOTE_PATH="${BACKUP_URL%/}/${STUDY_FILE}"

echo -e "${BLUE}📥 Downloading Optuna study database...${NC}"
echo -e "Study: ${YELLOW}$STUDY_NAME${NC}"
echo -e "Source: ${YELLOW}$REMOTE_PATH${NC}"
echo -e "Target: ${YELLOW}$LOCAL_PATH${NC}"

# Check if file already exists
if [ -f "$LOCAL_PATH" ] && [ "$FORCE" = false ]; then
    echo -e "${YELLOW}⚠️  File already exists: $LOCAL_PATH${NC}"
    echo "Use --force to overwrite, or specify a different output directory."
    exit 1
fi

# Download based on URL scheme
if [[ "$BACKUP_URL" == s3://* ]]; then
    # S3 download
    echo -e "${BLUE}📦 Downloading from S3...${NC}"
    
    if command -v aws &> /dev/null; then
        aws s3 cp "$REMOTE_PATH" "$LOCAL_PATH"
    else
        echo -e "${RED}❌ AWS CLI not found. Please install aws-cli to download from S3.${NC}"
        exit 1
    fi
    
elif [[ "$BACKUP_URL" == gs://* ]]; then
    # Google Cloud Storage download
    echo -e "${BLUE}☁️  Downloading from Google Cloud Storage...${NC}"
    
    if command -v gsutil &> /dev/null; then
        gsutil cp "$REMOTE_PATH" "$LOCAL_PATH"
    else
        echo -e "${RED}❌ gsutil not found. Please install Google Cloud SDK.${NC}"
        exit 1
    fi
    
elif [[ "$BACKUP_URL" == http://* ]] || [[ "$BACKUP_URL" == https://* ]]; then
    # HTTP download
    echo -e "${BLUE}🌐 Downloading via HTTP...${NC}"
    
    if command -v curl &> /dev/null; then
        curl -L -o "$LOCAL_PATH" "$REMOTE_PATH"
    elif command -v wget &> /dev/null; then
        wget -O "$LOCAL_PATH" "$REMOTE_PATH"
    else
        echo -e "${RED}❌ Neither curl nor wget found. Please install one of them.${NC}"
        exit 1
    fi
    
else
    # Local file copy
    echo -e "${BLUE}📁 Copying local file...${NC}"
    cp "$REMOTE_PATH" "$LOCAL_PATH"
fi

# Verify download
if [ -f "$LOCAL_PATH" ]; then
    FILE_SIZE=$(du -h "$LOCAL_PATH" | cut -f1)
    echo -e "${GREEN}✅ Download completed successfully!${NC}"
    echo -e "File size: ${YELLOW}$FILE_SIZE${NC}"
    
    # Verify it's a valid SQLite database
    if command -v sqlite3 &> /dev/null; then
        TRIAL_COUNT=$(sqlite3 "$LOCAL_PATH" "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "0")
        echo -e "Trial count: ${YELLOW}$TRIAL_COUNT${NC}"
        
        if [ "$TRIAL_COUNT" -gt 0 ]; then
            echo -e "${GREEN}📊 Database contains $TRIAL_COUNT trials${NC}"
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}🔍 Next steps:${NC}"
    echo "  1. View study: optuna-dashboard --storage sqlite:///$LOCAL_PATH"
    echo "  2. Export data: exo export --table optuna_trials --range last30d"
    echo "  3. Run dashboard: exo dash --port 8050 (🔬 Optuna Monitor tab)"
    
else
    echo -e "${RED}❌ Download failed. File not found: $LOCAL_PATH${NC}"
    exit 1
fi