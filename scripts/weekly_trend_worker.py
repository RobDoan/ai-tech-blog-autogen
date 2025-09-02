#!/usr/bin/env python3
"""
Weekly Trend Worker Script

Standalone executable script for running the weekly trend discovery process.
Designed to be scheduled via cron or other task schedulers.

Usage:
    python scripts/weekly_trend_worker.py [options]

Example cron entry (run every Sunday at 2 AM):
    0 2 * * 0 cd /path/to/project && python scripts/weekly_trend_worker.py

Exit codes:
    0: Success
    1: General error
    2: Configuration error
    3: S3 error
    4: Concurrent execution detected
"""

import sys
import os
import asyncio
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.topic_discovery.weekly_trend_worker import WeeklyTrendWorker


class WorkerScriptConfig:
    """Configuration class for the worker script"""
    
    def __init__(self):
        self.log_level = logging.INFO
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.structured_logging = False
        self.dry_run = False
        
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        config.log_level = getattr(logging, args.log_level.upper())
        config.structured_logging = args.structured_logging
        config.dry_run = args.dry_run
        return config


def setup_logging(config: WorkerScriptConfig):
    """Setup logging configuration"""
    import json
    
    if config.structured_logging:
        # JSON structured logging for monitoring systems
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if they exist
                if hasattr(record, 'source'):
                    log_entry['source'] = record.source
                if hasattr(record, 'article_count'):
                    log_entry['article_count'] = record.article_count
                if hasattr(record, 'error'):
                    log_entry['error'] = record.error
                    
                return json.dumps(log_entry)
        
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
    else:
        # Standard logging format
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config.log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Reduce noise from external libraries
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Weekly Trend Worker - Automated trend discovery and CSV export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default settings
  %(prog)s --log-level debug         # Enable debug logging
  %(prog)s --structured-logging      # Use JSON structured logging
  %(prog)s --dry-run                 # Test run without S3 upload
  %(prog)s --config custom.json     # Use custom configuration file

Exit Codes:
  0  Success
  1  General error
  2  Configuration error  
  3  S3 error
  4  Concurrent execution detected
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error'],
        default='info',
        help='Set logging level (default: info)'
    )
    
    parser.add_argument(
        '--structured-logging',
        action='store_true',
        help='Enable JSON structured logging for monitoring systems'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test run without uploading to S3'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to custom configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show last execution status and exit'
    )
    
    return parser.parse_args()


def load_custom_config(config_file: Path) -> dict:
    """Load custom configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded custom configuration from {config_file}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file {config_file}: {e}")
        sys.exit(2)  # Configuration error


async def run_worker(worker_config: dict, dry_run: bool = False) -> int:
    """
    Run the weekly trend worker
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Weekly Trend Worker execution")
    
    try:
        # Initialize worker
        if dry_run:
            logger.info("DRY RUN MODE - S3 upload will be skipped")
            # Modify config for dry run
            worker_config = worker_config.copy()
            worker_config['s3_bucket'] = None  # Disable S3 upload
        
        worker = WeeklyTrendWorker(config=worker_config)
        
        # Check last status
        last_status = worker.get_last_status()
        logger.info(f"Last execution: {last_status.get('last_run_status', 'never_run')}")
        
        # Run the discovery process
        results = await worker.run_weekly_discovery()
        
        # Log results
        status = results['status']
        trends_count = results['trends_discovered']
        
        if status == 'completed':
            logger.info(f"✅ Weekly trend discovery completed successfully")
            logger.info(f"📊 Discovered {trends_count} trending topics")
            
            if results.get('csv_file_uploaded'):
                logger.info(f"☁️ Uploaded to S3: {results['csv_file_uploaded']}")
            elif dry_run:
                logger.info(f"💾 Dry run - Local file: {results.get('local_backup_file')}")
            else:
                logger.warning(f"⚠️ S3 upload failed - Local backup: {results.get('local_backup_file')}")
            
            return 0  # Success
            
        elif 'concurrent' in results.get('error_message', '').lower():
            logger.error(f"🚫 Concurrent execution detected")
            return 4  # Concurrent execution
            
        else:
            logger.error(f"❌ Weekly trend discovery failed: {results.get('error_message')}")
            return 1  # General error
            
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}", exc_info=True)
        return 1  # General error


def show_status(worker_config: dict):
    """Show last execution status"""
    try:
        worker = WeeklyTrendWorker(config=worker_config)
        status = worker.get_last_status()
        
        print("📋 Weekly Trend Worker Status")
        print("=" * 40)
        print(f"Worker Type: {status.get('worker_type', 'unknown')}")
        print(f"Last Status: {status.get('last_run_status', 'never_run')}")
        print(f"Last Started: {status.get('last_started_at', 'N/A')}")
        print(f"Last Completed: {status.get('last_completed_at', 'N/A')}")
        print(f"Trends Discovered: {status.get('trends_discovered', 0)}")
        print(f"CSV File: {status.get('csv_file_uploaded', 'N/A')}")
        print(f"Local Backup: {status.get('local_backup_file', 'N/A')}")
        
        if status.get('error_message'):
            print(f"Last Error: {status['error_message']}")
            
    except Exception as e:
        print(f"Error retrieving status: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    start_time = datetime.now(timezone.utc)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup configuration
    script_config = WorkerScriptConfig.from_args(args)
    setup_logging(script_config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Weekly Trend Worker starting at {start_time.isoformat()}")
    
    # Load worker configuration
    worker_config = None
    if args.config:
        worker_config = load_custom_config(args.config)
    
    # Handle status request
    if args.status:
        show_status(worker_config or {})
        return 0
    
    try:
        # Run the worker
        exit_code = asyncio.run(run_worker(worker_config or {}, dry_run=args.dry_run))
        
        # Final logging
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        if exit_code == 0:
            logger.info(f"✅ Worker completed successfully in {duration:.1f} seconds")
        else:
            logger.error(f"❌ Worker failed with exit code {exit_code} after {duration:.1f} seconds")
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Worker interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)