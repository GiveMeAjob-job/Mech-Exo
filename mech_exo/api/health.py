"""
Health and Risk API Endpoints - Day 5 Module 2

Provides health monitoring and risk summary endpoints for dashboard consumption.
Includes the compact /riskz endpoint for unified risk control panel.
"""

import logging
import time
from datetime import datetime, date
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health status
    
    Returns:
        Dictionary with detailed health metrics
    """
    try:
        start_time = time.time()
        
        # Import here to avoid circular imports
        from ..reporting.query import get_health_data
        
        # Get health data
        health_data = get_health_data()
        
        # Add response metadata
        health_data.update({
            'api_version': '1.0',
            'response_time_ms': round((time.time() - start_time) * 1000, 2),
            'timestamp': datetime.now().isoformat()
        })
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/riskz")
async def get_risk_summary() -> Dict[str, Any]:
    """
    Get compact risk summary for dashboard consumption
    
    Returns:
        Compact JSON with 6 key risk metrics:
        {
            "today_pnl": -0.35,     // Today's P&L percentage
            "month_pnl": -1.2,      // Month-to-date P&L percentage  
            "kill": false,          // Kill-switch status (true = disabled)
            "canary": "on",         // Canary status ("on"/"off"/"unknown")
            "drill_days": 23,       // Days since last drill
            "capital_pct": 81       // Capital utilization percentage
        }
    """
    try:
        start_time = time.time()
        
        # Get risk data components
        risk_data = await _fetch_risk_data_async()
        
        # Build compact response
        response = {
            'today_pnl': risk_data.get('today_pnl', 0.0),
            'month_pnl': risk_data.get('month_pnl', 0.0),
            'kill': risk_data.get('kill_switch_enabled', False),
            'canary': risk_data.get('canary_status', 'unknown'),
            'drill_days': risk_data.get('drill_days', 999),
            'capital_pct': risk_data.get('capital_utilization_pct', 0)
        }
        
        # Add performance metadata
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Set custom headers for performance monitoring
        headers = {
            'X-Response-Time-Ms': str(response_time),
            'X-Data-Timestamp': datetime.now().isoformat(),
            'Cache-Control': 'max-age=30'  # Cache for 30 seconds
        }
        
        return JSONResponse(content=response, headers=headers)
        
    except Exception as e:
        logger.error(f"Risk summary failed: {e}")
        
        # Return safe defaults on error
        error_response = {
            'today_pnl': 0.0,
            'month_pnl': 0.0,
            'kill': True,  # Safe default - assume trading disabled
            'canary': 'unknown',
            'drill_days': 999,
            'capital_pct': 0
        }
        
        return JSONResponse(
            content=error_response, 
            status_code=200,  # Still return 200 with safe defaults
            headers={'X-Error': str(e)}
        )


async def _fetch_risk_data_async() -> Dict[str, Any]:
    """
    Asynchronously fetch all risk data components
    
    Returns:
        Dictionary with risk data from various sources
    """
    try:
        # Run data fetching tasks concurrently
        tasks = [
            _get_pnl_data(),
            _get_killswitch_data(), 
            _get_canary_data(),
            _get_drill_data(),
            _get_capital_data()
        ]
        
        # Execute all tasks concurrently with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results, handling any exceptions
        combined_data = {}
        for result in results:
            if isinstance(result, dict):
                combined_data.update(result)
            elif isinstance(result, Exception):
                logger.warning(f"Risk data component failed: {result}")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Async risk data fetch failed: {e}")
        return {}


async def _get_pnl_data() -> Dict[str, Any]:
    """Get P&L data (today and month-to-date)"""
    try:
        from ..reporting.query import get_health_data
        from ..utils.monthly_loss_guard import get_mtd_summary
        
        # Get daily P&L
        health_data = get_health_data()
        today_pnl = health_data.get('day_loss_pct', 0.0)
        
        # Get monthly P&L
        try:
            mtd_summary = get_mtd_summary()
            month_pnl = mtd_summary.get('mtd_pct', 0.0)
        except Exception as e:
            logger.warning(f"Monthly P&L fetch failed: {e}")
            month_pnl = 0.0
        
        return {
            'today_pnl': today_pnl,
            'month_pnl': month_pnl
        }
        
    except Exception as e:
        logger.error(f"P&L data fetch failed: {e}")
        return {'today_pnl': 0.0, 'month_pnl': 0.0}


async def _get_killswitch_data() -> Dict[str, Any]:
    """Get kill-switch status"""
    try:
        from ..cli.killswitch import KillSwitchManager
        
        manager = KillSwitchManager()
        status = manager.get_status()
        
        # Return inverted logic: kill=true means trading disabled
        return {
            'kill_switch_enabled': not status.get('trading_enabled', True)
        }
        
    except Exception as e:
        logger.error(f"Kill-switch data fetch failed: {e}")
        return {'kill_switch_enabled': True}  # Safe default


async def _get_canary_data() -> Dict[str, Any]:
    """Get canary/A-B testing status"""
    try:
        # This would integrate with A/B testing system
        # For now, check config or use default
        from ..utils.config import ConfigManager
        
        config_manager = ConfigManager()
        try:
            config = config_manager.load_config('canary')
            canary_enabled = config.get('enabled', False)
            canary_status = 'on' if canary_enabled else 'off'
        except:
            # Fallback to environment or default
            import os
            canary_status = 'on' if os.getenv('CANARY_ENABLED', 'false').lower() == 'true' else 'off'
        
        return {
            'canary_status': canary_status
        }
        
    except Exception as e:
        logger.error(f"Canary data fetch failed: {e}")
        return {'canary_status': 'unknown'}


async def _get_drill_data() -> Dict[str, Any]:
    """Get drill status (days since last drill)"""
    try:
        from ..dags.drill_flow import get_last_drill_info
        
        drill_info = get_last_drill_info()
        if drill_info and drill_info.get('drill_date'):
            drill_days = drill_info.get('days_ago', 999)
        else:
            drill_days = 999  # No drill found
        
        return {
            'drill_days': drill_days
        }
        
    except Exception as e:
        logger.error(f"Drill data fetch failed: {e}")
        return {'drill_days': 999}


async def _get_capital_data() -> Dict[str, Any]:
    """Get capital utilization percentage"""
    try:
        from ..cli.capital import CapitalManager
        
        manager = CapitalManager()
        config = manager.config
        
        # Get utilization data
        utilization_data = config.get('utilization', {})
        accounts_data = utilization_data.get('accounts', {})
        
        if accounts_data:
            # Calculate average utilization across accounts
            total_utilization = sum(
                account.get('utilization_pct', 0) 
                for account in accounts_data.values()
            )
            avg_utilization = total_utilization / len(accounts_data)
        else:
            avg_utilization = 0
        
        return {
            'capital_utilization_pct': round(avg_utilization, 1)
        }
        
    except Exception as e:
        logger.error(f"Capital data fetch failed: {e}")
        return {'capital_utilization_pct': 0}


@router.get("/riskz/test")
async def test_risk_endpoint() -> Dict[str, Any]:
    """
    Test endpoint that returns sample risk data for development
    
    Returns:
        Sample risk data for testing dashboard integration
    """
    import random
    
    # Generate realistic test data
    test_data = {
        'today_pnl': round(random.uniform(-1.5, 0.5), 2),
        'month_pnl': round(random.uniform(-4.0, 2.0), 2),
        'kill': random.choice([True, False]),
        'canary': random.choice(['on', 'off', 'unknown']),
        'drill_days': random.randint(1, 150),
        'capital_pct': round(random.uniform(30, 95), 1)
    }
    
    # Add test metadata
    headers = {
        'X-Test-Mode': 'true',
        'X-Response-Time-Ms': '5.2',
        'X-Data-Timestamp': datetime.now().isoformat()
    }
    
    return JSONResponse(content=test_data, headers=headers)


@router.get("/status")
async def get_api_status() -> Dict[str, Any]:
    """
    Get API status and performance metrics
    
    Returns:
        API health and performance information
    """
    try:
        start_time = time.time()
        
        # Test database connectivity
        try:
            from ..datasource.storage import DataStorage
            storage = DataStorage()
            storage.conn.execute("SELECT 1").fetchone()
            storage.close()
            db_status = "connected"
        except Exception as e:
            logger.warning(f"Database test failed: {e}")
            db_status = "error"
        
        # Test risk data fetch
        try:
            risk_data = await _fetch_risk_data_async()
            risk_status = "ok" if risk_data else "partial"
        except Exception as e:
            logger.warning(f"Risk data test failed: {e}")
            risk_status = "error"
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            'api_status': 'healthy',
            'database_status': db_status,
            'risk_data_status': risk_status,
            'response_time_ms': response_time,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# Performance monitoring middleware
@router.middleware("http")
async def monitor_performance(request: Request, call_next):
    """Monitor API performance and log slow requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log slow requests
    if process_time > 0.1:  # 100ms threshold
        logger.warning(f"Slow API request: {request.url.path} took {process_time:.3f}s")
    
    # Add performance header
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    
    return response