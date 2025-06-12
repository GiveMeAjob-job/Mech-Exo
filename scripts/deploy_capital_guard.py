#!/usr/bin/env python3
"""
Capital Guard Flow Deployment Script

Creates and schedules the capital guard flow to run daily at 08:45 UTC.
This flow monitors capital utilization and updates health status.
"""

import sys
from pathlib import Path
from datetime import time
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_capital_guard_deployment():
    """Create and register the capital guard deployment"""
    
    try:
        from dags.capital_guard_flow import capital_guard_flow
        
        # Create deployment with daily schedule at 08:45 UTC
        deployment = Deployment.build_from_flow(
            flow=capital_guard_flow,
            name="capital-guard-daily",
            description="Daily capital limit monitoring and health checks",
            tags=["capital", "monitoring", "daily", "production"],
            
            # Schedule: Daily at 08:45 UTC (before market open)
            schedule=CronSchedule(
                cron="45 8 * * *",  # 08:45 UTC every day
                timezone="UTC"
            ),
            
            # Parameters
            parameters={
                "stub_mode": False  # Production mode by default
            },
            
            # Work pool and infrastructure
            work_pool_name="default",
            
            # Version and metadata
            version="1.0.0",
            
            # Storage and execution settings
            path="./",
            entrypoint="dags/capital_guard_flow.py:capital_guard_flow"
        )
        
        # Apply deployment
        deployment_id = deployment.apply()
        
        logger.info(f"✅ Capital Guard deployment created successfully")
        logger.info(f"📋 Deployment ID: {deployment_id}")
        logger.info(f"⏰ Schedule: Daily at 08:45 UTC")
        logger.info(f"🏷️ Name: capital-guard-daily")
        
        return deployment_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create deployment: {e}")
        raise


def create_test_deployment():
    """Create a test deployment that runs immediately"""
    
    try:
        from dags.capital_guard_flow import capital_guard_flow
        
        # Create test deployment without schedule
        deployment = Deployment.build_from_flow(
            flow=capital_guard_flow,
            name="capital-guard-test",
            description="Test deployment for capital guard flow",
            tags=["capital", "monitoring", "test"],
            
            # No schedule - manual execution only
            schedule=None,
            
            # Parameters for testing
            parameters={
                "stub_mode": True  # Use stub mode for testing
            },
            
            # Work pool
            work_pool_name="default",
            
            # Version
            version="1.0.0-test",
            
            # Storage and execution settings
            path="./",
            entrypoint="dags/capital_guard_flow.py:capital_guard_flow"
        )
        
        # Apply deployment
        deployment_id = deployment.apply()
        
        logger.info(f"✅ Capital Guard test deployment created")
        logger.info(f"📋 Deployment ID: {deployment_id}")
        logger.info(f"🧪 Mode: Test (stub mode enabled)")
        logger.info(f"🏷️ Name: capital-guard-test")
        
        return deployment_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create test deployment: {e}")
        raise


def main():
    """Main deployment script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Capital Guard Flow")
    parser.add_argument("--test", action="store_true", 
                       help="Create test deployment instead of production")
    parser.add_argument("--run-now", action="store_true",
                       help="Run the flow immediately after deployment")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            deployment_id = create_test_deployment()
        else:
            deployment_id = create_capital_guard_deployment()
        
        if args.run_now:
            logger.info("🚀 Triggering immediate flow run...")
            
            # Import Prefect client to trigger run
            from prefect.client.orchestration import get_client
            import asyncio
            
            async def trigger_run():
                async with get_client() as client:
                    # Get deployment
                    deployment = await client.read_deployment_by_name(
                        "capital-guard-test" if args.test else "capital-guard-daily"
                    )
                    
                    # Create flow run
                    flow_run = await client.create_flow_run_from_deployment(
                        deployment.id,
                        parameters={"stub_mode": True} if args.test else {}
                    )
                    
                    logger.info(f"✅ Flow run created: {flow_run.id}")
                    return flow_run.id
            
            run_id = asyncio.run(trigger_run())
            logger.info(f"📊 Monitor run at: http://localhost:4200/flow-runs/{run_id}")
        
        logger.info("\n🎉 Capital Guard deployment completed successfully!")
        
        if not args.test:
            logger.info("\n📅 Next scheduled run: Tomorrow at 08:45 UTC")
            logger.info("🔍 Monitor deployments at: http://localhost:4200/deployments")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()