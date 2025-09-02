#!/usr/bin/env python3
"""
Enhanced Weekly Trend Worker Script

Standalone script for running the AI-powered blog title discovery system.
This script orchestrates the complete enhanced pipeline with command-line options
for configuration, debugging, and dry-run mode.
"""

import asyncio
import argparse
import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.topic_discovery.enhanced_weekly_trend_worker import EnhancedWeeklyTrendWorker
from src.services.topic_discovery.ai_fallback_handler import AIFallbackHandler, RetryConfig, FallbackMetrics
from src.services.topic_discovery.config import validate_configuration
from src.py_env import OPENAI_API_KEY


def setup_logging(level: str = "INFO", log_file: str = None, detailed: bool = False) -> logging.Logger:
    """
    Set up logging configuration for the worker script
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        detailed: Whether to include detailed formatting
        
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging format
    if detailed:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    else:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce verbosity of some third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    logger = logging.getLogger("enhanced_worker")
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Weekly Trend Worker - AI-powered blog title discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run with AI analysis
  python scripts/enhanced_weekly_trend_worker.py
  
  # Dry run without S3 upload
  python scripts/enhanced_weekly_trend_worker.py --dry-run
  
  # Debug mode with detailed logging
  python scripts/enhanced_weekly_trend_worker.py --log-level DEBUG --detailed-logs
  
  # Disable AI analysis (fallback mode only)
  python scripts/enhanced_weekly_trend_worker.py --disable-ai --disable-patterns
  
  # Custom configuration
  python scripts/enhanced_weekly_trend_worker.py --max-titles 50 --min-specificity 0.6
        """
    )
    
    # Core execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without uploading to S3 (for testing)"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to JSON configuration file"
    )
    
    # AI configuration
    parser.add_argument(
        "--disable-ai",
        action="store_true",
        help="Disable AI analysis (use fallback methods only)"
    )
    
    parser.add_argument(
        "--disable-patterns",
        action="store_true",
        help="Disable pattern detection and theme identification"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (overrides environment variable)"
    )
    
    # Processing limits
    parser.add_argument(
        "--max-titles",
        type=int,
        default=30,
        help="Maximum number of blog titles to generate (default: 30)"
    )
    
    parser.add_argument(
        "--min-specificity",
        type=float,
        default=0.4,
        help="Minimum specificity score for titles (0.0-1.0, default: 0.4)"
    )
    
    parser.add_argument(
        "--content-timeout",
        type=int,
        default=300,
        help="Timeout for content extraction in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--ai-timeout",
        type=int,
        default=600,
        help="Timeout for AI analysis in seconds (default: 600)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: console only)"
    )
    
    parser.add_argument(
        "--detailed-logs",
        action="store_true",
        help="Enable detailed logging with file names and line numbers"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/enhanced",
        help="Output directory for CSV files (default: ./data/enhanced)"
    )
    
    parser.add_argument(
        "--analytics-file",
        type=str,
        help="Path to save detailed analytics JSON file"
    )
    
    # Validation and testing
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--test-ai",
        action="store_true",
        help="Test AI connectivity and exit"
    )
    
    return parser.parse_args()


def load_config_file(config_file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")


def build_worker_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build worker configuration from arguments"""
    
    # Start with the base worker configuration from config module
    from src.services.topic_discovery.config import get_worker_config
    config = get_worker_config()
    
    # Load from config file if provided
    if args.config_file:
        config.update(load_config_file(args.config_file))
    
    # Override with command line arguments
    config.update({
        'max_blog_titles_per_run': args.max_titles,
        'min_specificity_threshold': args.min_specificity,
        'enable_ai_analysis': not args.disable_ai,
        'enable_pattern_detection': not args.disable_patterns,
        'content_extraction_timeout': args.content_timeout,
        'ai_analysis_timeout': args.ai_timeout,
        'local_backup_dir': args.output_dir,
    })
    
    return config


async def test_ai_connectivity(api_key: str) -> bool:
    """Test AI connectivity and API access"""
    
    logger = logging.getLogger("enhanced_worker")
    logger.info("Testing AI connectivity...")
    
    try:
        from src.services.topic_discovery.ai_semantic_analyzer import AISemanticAnalyzer
        
        # Create minimal test
        analyzer = AISemanticAnalyzer(api_key=api_key)
        
        # Simple test with minimal token usage
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        
        await analyzer.close()
        await client.close()
        
        logger.info("✅ AI connectivity test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI connectivity test failed: {str(e)}")
        return False


def save_analytics(analytics: Dict[str, Any], file_path: str):
    """Save detailed analytics to JSON file"""
    
    logger = logging.getLogger("enhanced_worker")
    
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        analytics_with_metadata = {
            "generated_at": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "analytics": analytics
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analytics_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analytics saved to: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save analytics: {str(e)}")


def print_summary(results: Dict[str, Any]):
    """Print execution summary to console"""
    
    print("\n" + "="*60)
    print("ENHANCED WEEKLY TREND WORKER - EXECUTION SUMMARY")
    print("="*60)
    
    # Execution details
    print(f"Status: {results['status'].upper()}")
    print(f"Execution Duration: {results.get('execution_duration', 0):.1f} seconds")
    print(f"Blog Titles Discovered: {results['blog_titles_discovered']}")
    
    if results.get('csv_file_uploaded'):
        print(f"✅ Data uploaded to: {results['csv_file_uploaded']}")
    elif results.get('local_backup_file'):
        print(f"📁 Local file saved: {results['local_backup_file']}")
    
    # Analytics summary
    if 'analytics' in results:
        analytics = results['analytics']
        
        print(f"\n📊 ANALYTICS SUMMARY")
        print(f"Articles Extracted: {analytics.get('content_extraction', {}).get('total_articles_extracted', 0)}")
        print(f"AI Insights Generated: {analytics.get('ai_analysis', {}).get('semantic_insights_generated', 0)}")
        print(f"Title Candidates: {analytics.get('title_generation', {}).get('total_title_candidates', 0)}")
        print(f"Titles Ranked: {analytics.get('title_scoring', {}).get('titles_ranked', 0)}")
        
        if 'pattern_detection' in analytics:
            pd = analytics['pattern_detection']
            print(f"Emerging Themes: {pd.get('emerging_themes_identified', 0)}")
            print(f"Content Series: {pd.get('content_series_suggested', 0)}")
    
    # Error information
    if results.get('error_message'):
        print(f"\n❌ ERROR: {results['error_message']}")
    
    print("="*60)


async def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        detailed=args.detailed_logs
    )
    
    try:
        logger.info("Enhanced Weekly Trend Worker starting...")
        
        # Validate configuration
        if args.validate_config:
            logger.info("Validating configuration...")
            issues = validate_configuration()
            if issues:
                logger.warning("Configuration issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                return 1
            else:
                logger.info("✅ Configuration validation passed")
                return 0
        
        # Determine API key
        api_key = args.openai_api_key or OPENAI_API_KEY
        
        # Test AI connectivity if requested
        if args.test_ai:
            if not api_key:
                logger.error("OpenAI API key not provided for AI test")
                return 1
            
            success = await test_ai_connectivity(api_key)
            return 0 if success else 1
        
        # Build configuration
        worker_config = build_worker_config(args)
        
        # Log configuration summary
        logger.info("Worker Configuration:")
        logger.info(f"  - AI Analysis: {'Enabled' if worker_config['enable_ai_analysis'] else 'Disabled'}")
        logger.info(f"  - Pattern Detection: {'Enabled' if worker_config['enable_pattern_detection'] else 'Disabled'}")
        logger.info(f"  - Max Titles: {worker_config['max_blog_titles_per_run']}")
        logger.info(f"  - Min Specificity: {worker_config['min_specificity_threshold']}")
        logger.info(f"  - Output Directory: {worker_config['local_backup_dir']}")
        
        # Create and configure worker
        enhanced_worker = EnhancedWeeklyTrendWorker(
            config=worker_config,
            openai_api_key=api_key if worker_config['enable_ai_analysis'] else None
        )
        
        # Run discovery process
        if args.dry_run:
            logger.info("Running in DRY-RUN mode (no S3 upload)")
            results = await enhanced_worker.run_dry_run()
        else:
            logger.info("Running enhanced blog title discovery")
            results = await enhanced_worker.run_enhanced_discovery()
        
        # Save analytics if requested
        if args.analytics_file and 'analytics' in results:
            save_analytics(results['analytics'], args.analytics_file)
        
        # Print summary
        print_summary(results)
        
        # Return appropriate exit code
        return 0 if results['status'] == 'completed' else 1
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)