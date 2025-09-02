import os
from dotenv import load_dotenv

load_dotenv() # Loads variables from.env file

# API Keys for external services
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
newsapi_api_key = os.getenv("NEWSAPI_API_KEY")
apify_api_token = os.getenv("APIFY_API_TOKEN")
github_api_token = os.getenv("GITHUB_API_TOKEN")

# AWS S3 configuration for weekly trend worker
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')
s3_bucket_name = os.getenv('S3_BUCKET_NAME', 'trending-topics-data')

# Database configuration
database_url = os.getenv('DATABASE_URL')
redis_url = os.getenv('REDIS_URL')