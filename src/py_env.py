import os
from dotenv import load_dotenv

load_dotenv() # Loads variables from.env file

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
newsapi_api_key = os.getenv("NEWSAPI_API_KEY")
apify_api_token = os.getenv("APIFY_API_TOKEN")
github_api_token = os.getenv("GITHUB_API_TOKEN")