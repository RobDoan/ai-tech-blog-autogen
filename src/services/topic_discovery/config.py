# src/services/topic_discovery/config.py
"""
Configuration management for the Weekly Trend Worker system.
Centralizes all configuration constants and environment variable handling.
"""

import os
from typing import Dict, List
from dataclasses import dataclass
from src.py_env import (
    aws_access_key_id, aws_secret_access_key, aws_region, s3_bucket_name,
    serpapi_api_key, newsapi_api_key, apify_api_token
)


@dataclass
class RSSSourceConfig:
    """Configuration for a single RSS source"""
    name: str
    rss_url: str
    weight: float
    timeout: int = 30


# RSS feed sources configuration as specified in requirements
RSS_SOURCES = [
    RSSSourceConfig(
        name="Netflix Tech Blog",
        rss_url="https://netflixtechblog.com/feed",
        weight=0.9,
        timeout=30
    ),
    RSSSourceConfig(
        name="GDB Blog", 
        rss_url="https://feeds.feedburner.com/GDBcode",
        weight=0.8,
        timeout=30
    ),
    RSSSourceConfig(
        name="Facebook Engineering",
        rss_url="https://engineering.fb.com/feed/",
        weight=0.9,
        timeout=30
    ),
    RSSSourceConfig(
        name="AWS Blog",
        rss_url="https://aws.amazon.com/blogs/aws/feed/",
        weight=0.9,
        timeout=30
    ),
    RSSSourceConfig(
        name="Stripe Engineering",
        rss_url="https://stripe.com/blog/engineering/feed.xml", 
        weight=0.8,
        timeout=30
    ),
    RSSSourceConfig(
        name="Hacker News",
        rss_url="https://news.ycombinator.com/rss",
        weight=0.7,
        timeout=30
    )
]


# Worker configuration
WORKER_CONFIG = {
    'max_trends_per_run': int(os.getenv('MAX_TRENDS_PER_RUN', 50)),
    'score_threshold': float(os.getenv('SCORE_THRESHOLD', 0.1)),
    'concurrent_execution_check': os.getenv('CONCURRENT_EXECUTION_CHECK', 'true').lower() == 'true',
    'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', 3)),
    'retry_delay': int(os.getenv('RETRY_DELAY', 5)),
    's3_bucket': s3_bucket_name or os.getenv('S3_BUCKET_NAME', 'trending-topics-data'),
    's3_key_prefix': os.getenv('S3_KEY_PREFIX', 'trends/'),
    'local_backup_dir': os.getenv('LOCAL_BACKUP_DIR', './data/backups'),
    'status_file_path': os.getenv('STATUS_FILE_PATH', './data/worker_status.json'),
    'lock_file_path': os.getenv('LOCK_FILE_PATH', './data/worker.lock')
}


# S3 configuration
S3_CONFIG = {
    'bucket_name': s3_bucket_name or os.getenv('S3_BUCKET_NAME', 'trending-topics-data'),
    'region': aws_region or os.getenv('AWS_REGION', 'us-east-1'),
    'key_prefix': os.getenv('S3_KEY_PREFIX', 'trends/'),
    'file_naming_pattern': os.getenv('S3_FILE_NAMING_PATTERN', 'weekly-trends-{timestamp}.csv'),
    'metadata_tags': {
        'source': 'weekly-trend-worker',
        'data_type': 'trending_topics',
        'project': 'ai-tech-blog-autogen'
    }
}


# Enhanced tech keywords for trend detection
TECH_KEYWORDS = [
    # AI/ML Keywords
    'AI', 'Artificial Intelligence', 'Machine Learning', 'ML', 'Deep Learning', 
    'Neural Networks', 'GPT', 'OpenAI', 'ChatGPT', 'Claude', 'Gemini',
    'LLM', 'Large Language Models', 'Computer Vision', 'NLP', 'Natural Language Processing',
    'Transformer', 'BERT', 'Vision Transformers', 'Stable Diffusion', 'Midjourney',
    
    # Programming Languages
    'Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'Golang', 'Rust', 
    'C++', 'C#', 'Swift', 'Kotlin', 'PHP', 'Ruby', 'Scala', 'Clojure',
    'WebAssembly', 'WASM', 'Dart', 'Elixir', 'Haskell',
    
    # Web Frameworks & Libraries
    'React', 'Vue', 'Vue.js', 'Angular', 'Svelte', 'Next.js', 'Nuxt.js',
    'Node.js', 'Express', 'Fastify', 'Django', 'Flask', 'FastAPI', 
    'Spring', 'Spring Boot', 'Ruby on Rails', 'Laravel', 'Symfony',
    
    # AI/ML Frameworks
    'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy',
    'Hugging Face', 'LangChain', 'OpenCV', 'JAX', 'MLflow', 'Weights & Biases',
    
    # Cloud & Infrastructure
    'AWS', 'Amazon Web Services', 'Azure', 'Microsoft Azure', 'GCP', 'Google Cloud',
    'Vercel', 'Netlify', 'Cloudflare', 'DigitalOcean', 'Heroku', 'Railway',
    'Docker', 'Kubernetes', 'K8s', 'Helm', 'Istio', 'Terraform', 'Pulumi',
    
    # DevOps & CI/CD
    'DevOps', 'CI/CD', 'GitHub Actions', 'GitLab CI', 'Jenkins', 'CircleCI',
    'Docker', 'Containerization', 'Microservices', 'Serverless', 'Lambda',
    'API Gateway', 'Load Balancer', 'CDN', 'Edge Computing',
    
    # Blockchain & Web3
    'Blockchain', 'Web3', 'NFT', 'Non-Fungible Token', 'Cryptocurrency', 
    'Bitcoin', 'Ethereum', 'Solana', 'Polygon', 'DeFi', 'Decentralized Finance',
    'Smart Contracts', 'DAO', 'Decentralized Autonomous Organization', 
    'Metaverse', 'Layer 2', 'zkSync', 'Optimism', 'Arbitrum',
    
    # Mobile Development
    'Mobile', 'iOS', 'Android', 'Swift', 'Kotlin', 'Flutter', 'React Native',
    'Xamarin', 'Ionic', 'Progressive Web App', 'PWA', 'Mobile First',
    
    # Data & Analytics
    'Data Science', 'Big Data', 'Analytics', 'Business Intelligence', 'BI',
    'ETL', 'ELT', 'Data Pipeline', 'Data Engineering', 'Data Warehouse',
    'Apache Spark', 'Hadoop', 'Kafka', 'Apache Airflow', 'dbt', 'Snowflake',
    'Databricks', 'Data Lake', 'Data Mesh', 'Real-time Analytics',
    
    # Databases
    'Database', 'SQL', 'NoSQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis',
    'Elasticsearch', 'Cassandra', 'DynamoDB', 'CouchDB', 'Neo4j',
    'Vector Database', 'Pinecone', 'Weaviate', 'Chroma', 'SQLite',
    
    # APIs & Architecture
    'API', 'REST', 'RESTful', 'GraphQL', 'gRPC', 'WebSocket', 'Server-Sent Events',
    'Event Driven Architecture', 'Message Queue', 'Pub/Sub', 'MQTT',
    'API Gateway', 'OpenAPI', 'Swagger', 'JSON-RPC', 'tRPC',
    
    # Security & Privacy
    'Cybersecurity', 'Security', 'Privacy', 'GDPR', 'CCPA', 'Authentication',
    'Authorization', 'OAuth', 'JWT', 'Zero Trust', 'VPN', 'End-to-End Encryption',
    'Penetration Testing', 'Vulnerability', 'Bug Bounty', 'OWASP',
    'Multi-Factor Authentication', 'MFA', 'Biometric Authentication',
    
    # Development Tools & Practices
    'GitHub', 'Git', 'GitLab', 'Version Control', 'Code Review', 'Pull Request',
    'Agile', 'Scrum', 'Kanban', 'Test Driven Development', 'TDD', 'BDD',
    'Unit Testing', 'Integration Testing', 'End-to-End Testing', 'Playwright',
    'Cypress', 'Jest', 'Pytest', 'Selenium', 'Code Quality', 'ESLint', 'Prettier',
    
    # Emerging Technologies
    'Quantum Computing', 'Quantum', 'Edge AI', '5G', '6G', 'IoT', 'Internet of Things',
    'Augmented Reality', 'AR', 'Virtual Reality', 'VR', 'Mixed Reality', 'MR',
    'Digital Twin', 'Autonomous Vehicles', 'Self-Driving Cars', 'Robotics',
    'Brain-Computer Interface', 'BCI', 'Neuralink', 'Wearable Technology',
    
    # Industry Trends
    'Remote Work', 'Work From Home', 'Hybrid Work', 'Digital Transformation',
    'Low-Code', 'No-Code', 'Citizen Developer', 'SaaS', 'PaaS', 'IaaS',
    'Subscription Model', 'API Economy', 'Platform Economy', 'Creator Economy',
    'Green Technology', 'Sustainable Tech', 'Carbon Footprint', 'ESG'
]


# API configuration
API_CONFIG = {
    'serpapi': {
        'api_key': serpapi_api_key,
        'timeout': 30,
        'max_retries': 3
    },
    'newsapi': {
        'api_key': newsapi_api_key,
        'timeout': 30,
        'max_retries': 3,
        'sources': [
            'techcrunch.com',
            'venturebeat.com', 
            'wired.com',
            'arstechnica.com',
            'theverge.com',
            'engadget.com',
            'mashable.com',
            'recode.net'
        ]
    },
    'apify': {
        'api_token': apify_api_token,
        'timeout': 60,
        'max_retries': 2
    }
}


# CSV export configuration
CSV_CONFIG = {
    'columns': [
        'topic', 'source', 'score', 'news_score', 'external_score',
        'sources', 'article_count', 'discovery_method', 'confidence_level',
        'discovered_at', 'duplicate_flag'
    ],
    'encoding': 'utf-8',
    'delimiter': ',',
    'quotechar': '"',
    'quoting': 1  # csv.QUOTE_ALL
}


# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO').upper(),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'structured': os.getenv('STRUCTURED_LOGGING', 'false').lower() == 'true',
    'file_path': os.getenv('LOG_FILE_PATH'),  # None for console only
    'max_bytes': int(os.getenv('LOG_MAX_BYTES', 10485760)),  # 10MB
    'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5))
}


def get_worker_config() -> Dict:
    """Get worker configuration with environment variable overrides"""
    return WORKER_CONFIG.copy()


def get_s3_config() -> Dict:
    """Get S3 configuration with environment variable overrides"""
    return S3_CONFIG.copy()


def get_rss_sources() -> List[Dict]:
    """Get RSS sources as dictionaries for backward compatibility"""
    return [
        {
            'name': source.name,
            'rss_url': source.rss_url,
            'weight': source.weight,
            'timeout': source.timeout
        }
        for source in RSS_SOURCES
    ]


def validate_configuration() -> List[str]:
    """
    Validate configuration and return list of issues
    
    Returns:
        List of configuration issues (empty if all good)
    """
    issues = []
    
    # Check required environment variables for S3
    if not aws_access_key_id and not aws_secret_access_key:
        issues.append("AWS credentials not configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    
    # Check S3 bucket name
    if not s3_bucket_name:
        issues.append("S3 bucket name not configured (S3_BUCKET_NAME)")
    
    # Check API keys (warn but don't fail)
    if not serpapi_api_key:
        issues.append("Warning: SERPAPI_API_KEY not configured (will use mock data)")
    
    if not newsapi_api_key:
        issues.append("Warning: NEWSAPI_API_KEY not configured (will use mock data)")
    
    if not apify_api_token:
        issues.append("Warning: APIFY_API_TOKEN not configured (will use mock data)")
    
    # Validate numeric configurations
    try:
        max_trends = int(os.getenv('MAX_TRENDS_PER_RUN', 50))
        if max_trends <= 0:
            issues.append("MAX_TRENDS_PER_RUN must be positive")
    except ValueError:
        issues.append("MAX_TRENDS_PER_RUN must be a valid integer")
    
    try:
        threshold = float(os.getenv('SCORE_THRESHOLD', 0.1))
        if threshold < 0 or threshold > 1:
            issues.append("SCORE_THRESHOLD must be between 0 and 1")
    except ValueError:
        issues.append("SCORE_THRESHOLD must be a valid float")
    
    return issues