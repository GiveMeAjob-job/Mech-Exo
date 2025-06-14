#!/bin/bash
# GitHub Actions Secrets Setup Script
# Run this script to configure required secrets for Risk Control system

set -e

echo "🔐 Setting up GitHub Secrets for Mech-Exo Risk Control"
echo "=================================================="

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Please install it first:"
    echo "   brew install gh  # macOS"
    echo "   or visit: https://cli.github.com/"
    exit 1
fi

# Check if logged in
if ! gh auth status &> /dev/null; then
    echo "❌ Not logged in to GitHub CLI. Please run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI ready"

# Function to set secret
set_secret() {
    local name=$1
    local description=$2
    local required=$3
    
    echo ""
    echo "🔑 Setting up: $name"
    echo "   Description: $description"
    
    if [ "$required" = "true" ]; then
        echo "   ⚠️  REQUIRED for system operation"
    else
        echo "   ℹ️  Optional (enhanced functionality)"
    fi
    
    read -sp "   Enter value for $name (or press Enter to skip): " value
    echo ""
    
    if [ -n "$value" ]; then
        echo "$value" | gh secret set "$name"
        echo "   ✅ $name configured"
    else
        if [ "$required" = "true" ]; then
            echo "   ⚠️  WARNING: $name is required but not set!"
        else
            echo "   ℹ️  $name skipped (optional)"
        fi
    fi
}

echo ""
echo "📋 Required Secrets for Risk Control System:"
echo ""

# Critical secrets
set_secret "TELEGRAM_BOT_TOKEN" "Telegram bot token for risk alerts" "true"
set_secret "TELEGRAM_CHAT_ID" "Telegram chat ID for alert delivery" "true"

echo ""
echo "📋 Optional Secrets (Enhanced Features):"
echo ""

# Optional secrets
set_secret "AWS_ACCESS_KEY_ID" "AWS access key for S3 audit storage" "false"
set_secret "AWS_SECRET_ACCESS_KEY" "AWS secret key for S3 audit storage" "false"
set_secret "PREFECT_API_KEY" "Prefect Cloud API key for workflow orchestration" "false"

echo ""
echo "📋 Environment Variables (Set in workflow files):"
echo ""
echo "NODE_OPTIONS=\"--max-old-space-size=4096\"  # Prevents JS heap OOM"
echo "AWS_DEFAULT_REGION=\"us-east-1\"           # Default AWS region"
echo "DATABASE_URL=\"sqlite:///data/trading.db\"  # Database connection"
echo "DASH_HOST=\"0.0.0.0\"                      # Dashboard host"
echo "DASH_PORT=\"8050\"                         # Dashboard port"

echo ""
echo "🎯 Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Verify secrets: gh secret list"
echo "2. Test workflow: git push (triggers risk_master.yml)"
echo "3. Check CI output for secret validation"
echo ""
echo "For secret rotation, see: README-OPS.md"