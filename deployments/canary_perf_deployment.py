"""
Prefect deployment for Daily Canary Performance Tracker

Schedules the canary_performance_flow to run daily at 23:30 UTC (after market close).
"""

import sys
from pathlib import Path
from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dags.canary_perf_flow import canary_performance_flow


def create_canary_deployment():
    """Create and return the canary performance deployment"""
    
    # Schedule for 23:30 UTC daily (after US market close)
    schedule = CronSchedule(
        cron="30 23 * * *",  # 23:30 UTC every day
        timezone="UTC"
    )
    
    # Create deployment
    deployment = canary_performance_flow.to_deployment(
        name="daily-canary-performance",
        description="Daily canary vs base performance tracking, scheduled at 23:30 UTC",
        version="1.0.0",
        schedule=schedule,
        parameters={
            "target_date": None,  # Will use today's date
            "window_days": 30     # 30-day rolling Sharpe
        },
        tags=["production", "canary", "performance", "daily"],
        work_pool_name="default"  # Use default work pool
    )
    
    return deployment


if __name__ == "__main__":
    # Create and serve the deployment
    print("🚀 Creating canary performance deployment...")
    
    deployment = create_canary_deployment()
    
    print(f"📅 Deployment created: {deployment.name}")
    print(f"⏰ Schedule: Daily at 23:30 UTC")
    print(f"🏷️  Tags: {deployment.tags}")
    
    # Serve the deployment
    print("\n🔄 Serving deployment (press Ctrl+C to stop)...")
    serve(deployment)