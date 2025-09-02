# Weekly Trend Worker

Automated weekly trend discovery system that combines RSS feeds from major tech companies with external APIs to identify trending topics, then exports results to CSV and uploads to S3.

## Overview

The Weekly Trend Worker replaces database dependencies with a robust CSV export and S3 upload system while maintaining comprehensive trend discovery capabilities from high-quality sources.

## Features

- **RSS Feed Processing**: 6 authoritative tech sources (Netflix, AWS, Facebook, Stripe, GDB, Hacker News)
- **Enhanced Topic Extraction**: 100+ tech keywords with confidence scoring and source attribution
- **Trend Aggregation**: Combines RSS and external API data with intelligent scoring algorithms
- **CSV Export**: Standardized format with timestamped naming and metadata
- **S3 Upload**: Robust cloud storage with retry logic and exponential backoff
- **Local Backup**: Automatic local file storage when S3 is unavailable
- **Concurrent Execution Prevention**: Resource-safe file-based locking using context managers to prevent overlapping runs
- **Comprehensive Logging**: Structured logging with monitoring integration
- **Error Recovery**: Graceful degradation and retry mechanisms

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Verify installation
python scripts/weekly_trend_worker.py --status
```

### Basic Usage

```bash
# Run with default settings
python scripts/weekly_trend_worker.py

# Run with debug logging
python scripts/weekly_trend_worker.py --log-level debug

# Dry run (no S3 upload)
python scripts/weekly_trend_worker.py --dry-run

# Use custom configuration
python scripts/weekly_trend_worker.py --config example_config.json
```

### Environment Setup

Create a `.env` file with required configuration:

```bash
# AWS S3 Configuration (required)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key  
AWS_REGION=us-east-1
S3_BUCKET_NAME=trending-topics-data

# External API Keys (optional - will use mock data if missing)
SERPAPI_API_KEY=your_serpapi_key
NEWSAPI_API_KEY=your_newsapi_key
APIFY_API_TOKEN=your_apify_token

# Worker Configuration (optional)
MAX_TRENDS_PER_RUN=50
SCORE_THRESHOLD=0.1
CONCURRENT_EXECUTION_CHECK=true
```

## Scheduling

### Cron Job Setup

```bash
# Edit crontab
crontab -e

# Add entry to run every Sunday at 2 AM
0 2 * * 0 cd /path/to/project && python scripts/weekly_trend_worker.py >> /var/log/trend-worker.log 2>&1
```

### Docker Setup

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -e .

# Run weekly
CMD ["python", "scripts/weekly_trend_worker.py"]
```

```bash
# Build and run
docker build -t weekly-trend-worker .
docker run --env-file .env weekly-trend-worker
```

## Configuration

### Worker Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_trends_per_run` | 50 | Maximum trends to include in output |
| `score_threshold` | 0.1 | Minimum score for trend inclusion |
| `concurrent_execution_check` | true | Enable concurrent execution prevention |
| `retry_attempts` | 3 | Number of retry attempts for S3 upload |
| `retry_delay` | 5 | Delay between retries (seconds) |
| `s3_bucket` | trending-topics-data | S3 bucket name |
| `s3_key_prefix` | trends/ | S3 key prefix for uploads |

### RSS Sources

The system uses these high-quality RSS feeds as specified in requirements:

- Netflix Tech Blog: `https://netflixtechblog.com/feed`
- GDB Blog: `https://feeds.feedburner.com/GDBcode`
- Facebook Engineering: `https://engineering.fb.com/feed/`
- AWS Blog: `https://aws.amazon.com/blogs/aws/feed/`
- Stripe Engineering: `https://stripe.com/blog/engineering/feed.xml`
- Hacker News: `https://news.ycombinator.com/rss`

## CSV Output Format

The system generates CSV files with this standardized structure:

```csv
topic,source,score,news_score,external_score,sources,article_count,discovery_method,confidence_level,discovered_at,duplicate_flag
"Machine Learning",weekly_worker,2.85,2.1,0.75,"Netflix Tech Blog,AWS Blog",12,combined,high,"2024-01-15T10:00:00Z",false
```

### Column Descriptions

- **topic**: The trending topic name
- **source**: Always "weekly_worker" for this system  
- **score**: Combined confidence score (0-1+)
- **news_score**: RSS feed analysis score
- **external_score**: External API trend score
- **sources**: Comma-separated list of contributing RSS sources
- **article_count**: Number of articles mentioning this topic
- **discovery_method**: "rss_analysis", "api_analysis", or "combined"
- **confidence_level**: "low", "medium", or "high"
- **discovered_at**: ISO timestamp of discovery
- **duplicate_flag**: Boolean indicating potential duplicates

## S3 Integration

### File Naming

Files are uploaded to S3 with timestamped naming:

```
s3://your-bucket/trends/weekly-trends-2024-01-15.csv
```

### Metadata

Each uploaded file includes metadata tags:

```json
{
  "source": "weekly-trend-worker",
  "data_type": "trending_topics", 
  "created_at": "2024-01-15T10:00:00Z"
}
```

## Monitoring and Logging

### Status File

The worker maintains a local status file (`./data/worker_status.json`) with execution history:

```json
{
  "worker_type": "weekly_trend_worker",
  "last_run_status": "completed",
  "last_started_at": "2024-01-15T10:00:00Z",
  "last_completed_at": "2024-01-15T10:15:00Z", 
  "trends_discovered": 23,
  "csv_file_uploaded": "s3://bucket/trends/weekly-trends-2024-01-15.csv",
  "error_message": null
}
```

### Structured Logging

Enable JSON structured logging for monitoring systems:

```bash
python scripts/weekly_trend_worker.py --structured-logging
```

Sample structured log entry:

```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "level": "INFO", 
  "logger": "WeeklyTrendWorker",
  "message": "Successfully scanned Netflix Tech Blog: 15 articles",
  "source": "Netflix Tech Blog",
  "article_count": 15
}
```

## Error Handling

The system includes comprehensive error handling:

- **RSS Feed Failures**: Individual feed failures don't stop the process
- **S3 Upload Failures**: Automatic retry with exponential backoff
- **API Timeouts**: Graceful fallbacks to available data sources
- **Concurrent Execution**: Resource-safe file-based locking with automatic cleanup prevents overlapping runs
- **Network Issues**: Robust timeout and retry mechanisms

## Exit Codes

The worker script returns specific exit codes for monitoring:

- **0**: Success
- **1**: General error
- **2**: Configuration error
- **3**: S3 error
- **4**: Concurrent execution detected

## Testing

```bash
# Run unit tests
python -m pytest tests/test_news_scanner.py -v
python -m pytest tests/test_weekly_trend_worker.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

## Development

### Adding New RSS Sources

Update the `RSS_SOURCES` configuration in `src/services/topic_discovery/config.py`:

```python
RSS_SOURCES.append(RSSSourceConfig(
    name="New Tech Blog",
    rss_url="https://example.com/feed",
    weight=0.8,
    timeout=30
))
```

### Adding New Keywords

Extend the `TECH_KEYWORDS` list in the configuration file:

```python
TECH_KEYWORDS.extend([
    'New Technology',
    'Emerging Framework'
])
```

### Custom Scoring Algorithms  

Override the `_aggregate_and_score_trends` method in `WeeklyTrendWorker`:

```python
async def _aggregate_and_score_trends(self, news_trends, external_trend):
    # Custom scoring logic here
    pass
```

## Troubleshooting

### Common Issues

1. **"Unable to locate credentials"**
   - Set AWS credentials in environment variables or IAM role
   
2. **"S3 bucket access test failed"**
   - Verify bucket exists and permissions are correct
   - Check AWS region configuration

3. **"Concurrent execution detected"**  
   - Another worker instance is running
   - Lock files are automatically cleaned up by context managers
   - Stale lock files should be rare due to improved resource management

4. **"No trends discovered"**
   - RSS feeds may be temporarily unavailable
   - Check network connectivity and feed URLs

### Debug Mode

Run with debug logging to troubleshoot issues:

```bash
python scripts/weekly_trend_worker.py --log-level debug --dry-run
```

This will show detailed information about:
- RSS feed processing
- Topic extraction
- Scoring algorithms  
- S3 operations
- Error conditions

## Architecture

### Components

- **TechNewsScanner**: RSS feed processing and topic extraction
- **TrendSpotter**: External API integration for trend validation
- **WeeklyTrendWorker**: Main orchestration class with resource-safe file locking
- **FileLockManager**: Context manager for safe file-based locking without resource leaks
- **Configuration**: Centralized settings and environment management
- **Worker Script**: Standalone executable for scheduling

### Workflow

1. **RSS Collection**: Scan 6 configured RSS feeds concurrently
2. **Topic Extraction**: Extract trending topics with keyword matching
3. **External Validation**: Query external APIs for trend confirmation  
4. **Aggregation**: Combine and score trends from all sources
5. **CSV Export**: Generate standardized CSV output
6. **S3 Upload**: Upload to cloud storage with retry logic
7. **Status Update**: Update local status file with results

### Data Flow

```
RSS Feeds → Topic Extraction → Trend Aggregation → CSV Export → S3 Upload
     ↓              ↓               ↓              ↓         ↓
External APIs → Validation → Combined Scoring → Local Backup → Status File
```

### Technical Improvements

#### Resource Management
- **Context Managers**: All file operations use proper context managers to prevent resource leaks
- **Automatic Cleanup**: File handles and locks are automatically released even during exceptions
- **Exception Safety**: Robust error handling ensures system stability under all conditions

#### Concurrent Execution Safety
- **FileLockManager**: Custom context manager for safe file-based locking
- **Automatic Lock Release**: Locks are automatically released when the context exits
- **Process Isolation**: Each worker instance properly isolates its resources

This completes the Weekly Trend Worker implementation with comprehensive documentation, resource-safe operation, and examples for production deployment.