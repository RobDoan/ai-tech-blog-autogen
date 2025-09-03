"""
Centralized configuration module that loads all environment variables from .env file.
This module provides a single source of truth for all configuration values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================================
# API Keys - Required for core functionality
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")

# =============================================================================
# AI Model Configuration - OpenAI Settings
# =============================================================================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# =============================================================================
# Multi-Agent Blog Writer Configuration
# =============================================================================
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
ENABLE_CODE_AGENT = os.getenv("ENABLE_CODE_AGENT", "true").lower() == "true"
ENABLE_SEO_AGENT = os.getenv("ENABLE_SEO_AGENT", "true").lower() == "true"
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "7.0"))
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "markdown")
SAVE_CONVERSATION_LOG = os.getenv("SAVE_CONVERSATION_LOG", "true").lower() == "true"

# =============================================================================
# AWS Configuration - For S3 uploads and cloud storage
# =============================================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "trending-topics-data")
S3_KEY_PREFIX = os.getenv("S3_KEY_PREFIX", "trends/")
S3_FILE_NAMING_PATTERN = os.getenv(
    "S3_FILE_NAMING_PATTERN", "weekly-trends-{timestamp}.csv"
)

# =============================================================================
# File Storage and Data Configuration
# =============================================================================
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
LOCAL_BACKUP_DIR = Path(os.getenv("LOCAL_BACKUP_DIR", "./data/backups"))
STATUS_FILE_PATH = Path(os.getenv("STATUS_FILE_PATH", "./data/worker_status.json"))
LOCK_FILE_PATH = Path(os.getenv("LOCK_FILE_PATH", "./data/worker.lock"))

# =============================================================================
# Trend Discovery and Worker Configuration
# =============================================================================
MAX_TRENDS_PER_RUN = int(os.getenv("MAX_TRENDS_PER_RUN", "50"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.1"))
CONCURRENT_EXECUTION_CHECK = (
    os.getenv("CONCURRENT_EXECUTION_CHECK", "true").lower() == "true"
)
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STRUCTURED_LOGGING = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# =============================================================================
# Application Configuration
# =============================================================================
APP_NAME = os.getenv("APP_NAME", "Automated Blog")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# =============================================================================
# Security Configuration
# =============================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")

# =============================================================================
# Database configuration (for future use)
# =============================================================================
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# =============================================================================
# Backward compatibility - Legacy variable names
# =============================================================================
# Keep these for existing code that might use the old lowercase names
openai_api_key = OPENAI_API_KEY
serpapi_api_key = SERPAPI_API_KEY
newsapi_api_key = NEWSAPI_API_KEY
apify_api_token = APIFY_API_TOKEN
github_api_token = GITHUB_API_TOKEN
aws_access_key_id = AWS_ACCESS_KEY_ID
aws_secret_access_key = AWS_SECRET_ACCESS_KEY
aws_region = AWS_REGION
s3_bucket_name = S3_BUCKET_NAME
database_url = DATABASE_URL
redis_url = REDIS_URL


def get_required_env_var(var_name: str, error_msg: str | None = None) -> str:
    """
    Get a required environment variable, raising an error if not found.

    Args:
        var_name: Name of the environment variable
        error_msg: Custom error message to display

    Returns:
        The environment variable value

    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        msg = error_msg or f"{var_name} environment variable is required but not set"
        raise ValueError(msg)
    return value


def validate_config() -> dict[str, bool]:
    """
    Validate the current configuration and return status of different components.

    Returns:
        Dict mapping component names to their configuration validity
    """
    status = {
        "openai_configured": bool(OPENAI_API_KEY),
        "aws_configured": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
        "external_apis_configured": bool(SERPAPI_API_KEY or NEWSAPI_API_KEY),
        "github_configured": bool(GITHUB_API_TOKEN),
        "data_paths_exist": DATA_PATH.exists() or DATA_PATH.parent.exists(),
    }

    return status
