# tests/test_weekly_trend_worker.py
"""
Unit tests for the WeeklyTrendWorker orchestration class.
Tests the complete workflow including trend aggregation, CSV export, and S3 upload.
"""

import pytest
import asyncio
import json
import csv
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
from pathlib import Path

from src.services.topic_discovery.weekly_trend_worker import WeeklyTrendWorker


class TestWeeklyTrendWorker:
    """Test suite for WeeklyTrendWorker"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Test configuration"""
        return {
            'max_trends_per_run': 10,
            'score_threshold': 0.2,
            'concurrent_execution_check': False,  # Disable for tests
            'retry_attempts': 2,
            'retry_delay': 1,
            's3_bucket': 'test-bucket',
            's3_key_prefix': 'test-trends/',
            'local_backup_dir': str(temp_dir / 'backups'),
            'status_file_path': str(temp_dir / 'status.json'),
            'lock_file_path': str(temp_dir / 'worker.lock')
        }
    
    @pytest.fixture
    def mock_news_trends(self):
        """Mock news trends data"""
        return [
            {
                'topic': 'Machine Learning',
                'score': 2.5,
                'news_score': 2.5,
                'external_score': 0.0,
                'sources': 'Netflix Tech Blog,AWS Blog',
                'article_count': 5,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'high',
                'discovered_at': datetime.now(timezone.utc)
            },
            {
                'topic': 'Docker',
                'score': 1.8,
                'news_score': 1.8,
                'external_score': 0.0,
                'sources': 'Facebook Engineering',
                'article_count': 3,
                'discovery_method': 'rss_analysis', 
                'confidence_level': 'medium',
                'discovered_at': datetime.now(timezone.utc)
            },
            {
                'topic': 'React',
                'score': 1.2,
                'news_score': 1.2,
                'external_score': 0.0,
                'sources': 'Stripe Engineering',
                'article_count': 2,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'medium',
                'discovered_at': datetime.now(timezone.utc)
            }
        ]
    
    @pytest.fixture
    def mock_external_trend(self):
        """Mock external trend data"""
        return {
            'topic': 'Artificial Intelligence',
            'score': 0.85,
            'metrics': {
                'search_rank': 1,
                'article_count': 25,
                'tweet_volume': 1500
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @pytest.fixture
    def worker(self, test_config):
        """Create WeeklyTrendWorker instance with test config"""
        with patch('src.services.topic_discovery.weekly_trend_worker.boto3'):
            worker = WeeklyTrendWorker(config=test_config)
            worker.s3_client = Mock()  # Mock S3 client
            return worker
    
    def test_worker_initialization(self, worker, test_config):
        """Test worker initialization"""
        assert worker.config == test_config
        assert worker.news_scanner is not None
        assert worker.trend_spotter is not None
        
        # Check directories are created
        assert Path(test_config['local_backup_dir']).exists()
        assert Path(test_config['status_file_path']).parent.exists()
    
    def test_default_config(self):
        """Test default configuration"""
        with patch('src.services.topic_discovery.weekly_trend_worker.boto3'):
            worker = WeeklyTrendWorker()
            
        config = worker.config
        assert config['max_trends_per_run'] == 50
        assert config['score_threshold'] == 0.1
        assert config['concurrent_execution_check'] is True
        assert config['retry_attempts'] == 3
    
    @pytest.mark.asyncio
    async def test_collect_news_trends(self, worker, mock_news_trends):
        """Test news trend collection"""
        # Mock the news scanner
        worker.news_scanner.scan_tech_news = AsyncMock(return_value=mock_news_trends)
        
        trends = await worker._collect_news_trends()
        
        assert trends == mock_news_trends
        worker.news_scanner.scan_tech_news.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_news_trends_error(self, worker):
        """Test news trend collection with error"""
        # Mock error in news scanner
        worker.news_scanner.scan_tech_news = AsyncMock(side_effect=Exception("RSS Error"))
        
        trends = await worker._collect_news_trends()
        
        assert trends == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_collect_external_trends(self, worker, mock_external_trend):
        """Test external trend collection"""
        # Mock trend spotter
        worker.trend_spotter.get_weekly_trend = Mock(return_value=mock_external_trend)
        
        trends = await worker._collect_external_trends()
        
        assert trends == mock_external_trend
        worker.trend_spotter.get_weekly_trend.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_external_trends_error(self, worker):
        """Test external trend collection with error"""
        # Mock error in trend spotter
        worker.trend_spotter.get_weekly_trend = Mock(side_effect=Exception("API Error"))
        
        trends = await worker._collect_external_trends()
        
        assert trends == {}  # Should return empty dict on error
    
    @pytest.mark.asyncio
    async def test_aggregate_and_score_trends(self, worker, mock_news_trends, mock_external_trend):
        """Test trend aggregation and scoring"""
        aggregated = await worker._aggregate_and_score_trends(mock_news_trends, mock_external_trend)
        
        assert isinstance(aggregated, list)
        assert len(aggregated) > 0
        
        # Check that external trend was added
        topic_names = [t['topic'] for t in aggregated]
        assert 'Artificial Intelligence' in topic_names
        
        # Check combined scoring
        ai_trend = next(t for t in aggregated if t['topic'] == 'Artificial Intelligence')
        assert ai_trend['external_score'] > 0
        assert ai_trend['discovery_method'] == 'api_analysis'
    
    @pytest.mark.asyncio
    async def test_aggregate_trends_with_duplicates(self, worker):
        """Test trend aggregation with duplicate topics"""
        news_trends = [
            {
                'topic': 'Machine Learning',
                'score': 2.0,
                'sources': 'Source A',
                'article_count': 3,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'medium',
                'discovered_at': datetime.now(timezone.utc)
            },
            {
                'topic': 'machine learning',  # Duplicate (different case)
                'score': 1.5,
                'sources': 'Source B',
                'article_count': 2,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'low',
                'discovered_at': datetime.now(timezone.utc)
            }
        ]
        
        external_trend = {}
        
        aggregated = await worker._aggregate_and_score_trends(news_trends, external_trend)
        
        # Should merge duplicates
        ml_trends = [t for t in aggregated if 'machine learning' in t['topic'].lower()]
        assert len(ml_trends) == 1
        
        merged_trend = ml_trends[0]
        assert merged_trend['duplicate_flag'] is True
        assert merged_trend['article_count'] == 5  # 3 + 2
    
    @pytest.mark.asyncio
    async def test_export_trends_to_csv(self, worker, mock_news_trends):
        """Test CSV export functionality"""
        csv_path = await worker._export_trends_to_csv(mock_news_trends)
        
        assert csv_path.exists()
        assert csv_path.suffix == '.csv'
        
        # Read and verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == len(mock_news_trends)
        
        # Check CSV structure
        expected_columns = [
            'topic', 'source', 'score', 'news_score', 'external_score',
            'sources', 'article_count', 'discovery_method', 'confidence_level',
            'discovered_at', 'duplicate_flag'
        ]
        
        assert list(rows[0].keys()) == expected_columns
        
        # Check data
        first_row = rows[0]
        assert first_row['topic'] == 'Machine Learning'
        assert first_row['score'] == '2.5'
        assert first_row['sources'] == 'Netflix Tech Blog,AWS Blog'
    
    @pytest.mark.asyncio
    async def test_upload_to_s3_success(self, worker, temp_dir):
        """Test successful S3 upload"""
        # Create test CSV file
        test_csv = temp_dir / 'test.csv'
        test_csv.write_text('topic,score\nTest Topic,1.0\n')
        
        # Mock successful S3 upload
        worker.s3_client.upload_file = Mock()
        
        s3_url = await worker._upload_to_s3(test_csv)
        
        assert s3_url is not None
        assert s3_url.startswith('s3://')
        assert worker.config['s3_bucket'] in s3_url
        worker.s3_client.upload_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_to_s3_failure(self, worker, temp_dir):
        """Test S3 upload failure with retries"""
        # Create test CSV file
        test_csv = temp_dir / 'test.csv'
        test_csv.write_text('topic,score\nTest Topic,1.0\n')
        
        # Mock S3 upload failure
        worker.s3_client.upload_file = Mock(side_effect=Exception("S3 Error"))
        
        s3_url = await worker._upload_to_s3(test_csv)
        
        assert s3_url is None
        # Should retry based on config
        assert worker.s3_client.upload_file.call_count == worker.config['retry_attempts']
    
    @pytest.mark.asyncio
    async def test_upload_to_s3_no_client(self, worker, temp_dir):
        """Test S3 upload when no client is available"""
        # Create test CSV file
        test_csv = temp_dir / 'test.csv'
        test_csv.write_text('topic,score\nTest Topic,1.0\n')
        
        # Remove S3 client
        worker.s3_client = None
        
        s3_url = await worker._upload_to_s3(test_csv)
        
        assert s3_url is None
    
    def test_update_status_file(self, worker, temp_dir):
        """Test status file update"""
        results = {
            'status': 'completed',
            'started_at': '2024-01-15T10:00:00Z',
            'completed_at': '2024-01-15T10:15:00Z',
            'trends_discovered': 25,
            'csv_file_uploaded': 's3://bucket/trends.csv',
            'local_backup_file': '/path/to/backup.csv',
            'error_message': None
        }
        
        worker._update_status_file(results)
        
        # Check status file was created
        status_path = Path(worker.config['status_file_path'])
        assert status_path.exists()
        
        # Check content
        with open(status_path, 'r') as f:
            status_data = json.load(f)
        
        assert status_data['worker_type'] == 'weekly_trend_worker'
        assert status_data['last_run_status'] == 'completed'
        assert status_data['trends_discovered'] == 25
        assert status_data['csv_file_uploaded'] == 's3://bucket/trends.csv'
    
    def test_get_last_status(self, worker, temp_dir):
        """Test getting last status"""
        # No status file exists
        status = worker.get_last_status()
        assert status['status'] == 'never_run'
        
        # Create status file
        status_data = {
            'worker_type': 'weekly_trend_worker',
            'last_run_status': 'completed',
            'trends_discovered': 42
        }
        
        status_path = Path(worker.config['status_file_path'])
        with open(status_path, 'w') as f:
            json.dump(status_data, f)
        
        # Read status
        status = worker.get_last_status()
        assert status['last_run_status'] == 'completed'
        assert status['trends_discovered'] == 42
    
    def test_acquire_and_release_lock(self, worker):
        """Test file locking mechanism"""
        # Test acquiring lock
        acquired = worker._acquire_lock()
        assert acquired is True
        
        # Test lock file exists
        lock_path = Path(worker.config['lock_file_path'])
        assert lock_path.exists()
        
        # Test releasing lock
        worker._release_lock()
        assert not lock_path.exists()
    
    def test_acquire_lock_already_locked(self, worker):
        """Test acquiring lock when already locked"""
        # First acquisition should succeed
        assert worker._acquire_lock() is True
        
        # Create another worker with same config
        worker2 = WeeklyTrendWorker(config=worker.config)
        
        # Second acquisition should fail
        assert worker2._acquire_lock() is False
        
        # Cleanup
        worker._release_lock()
    
    @pytest.mark.asyncio
    async def test_run_weekly_discovery_success(self, worker, mock_news_trends, mock_external_trend):
        """Test complete successful workflow"""
        # Mock all components
        worker.news_scanner.scan_tech_news = AsyncMock(return_value=mock_news_trends)
        worker.trend_spotter.get_weekly_trend = Mock(return_value=mock_external_trend)
        worker.s3_client.upload_file = Mock()
        
        results = await worker.run_weekly_discovery()
        
        assert results['status'] == 'completed'
        assert results['trends_discovered'] > 0
        assert results['csv_file_uploaded'] is not None
        assert results['error_message'] is None
        assert results['completed_at'] is not None
    
    @pytest.mark.asyncio
    async def test_run_weekly_discovery_with_error(self, worker):
        """Test workflow with error"""
        # Mock error in news scanner
        worker.news_scanner.scan_tech_news = AsyncMock(side_effect=Exception("Test Error"))
        
        results = await worker.run_weekly_discovery()
        
        assert results['status'] == 'failed'
        assert results['error_message'] is not None
        assert 'Test Error' in results['error_message']
    
    @pytest.mark.asyncio
    async def test_s3_client_initialization_with_credentials(self):
        """Test S3 client initialization with credentials"""
        test_config = {'s3_bucket': 'test-bucket'}
        
        with patch('src.services.topic_discovery.weekly_trend_worker.aws_access_key_id', 'test_key'):
            with patch('src.services.topic_discovery.weekly_trend_worker.aws_secret_access_key', 'test_secret'):
                with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client') as mock_boto3:
                    mock_client = Mock()
                    mock_boto3.return_value = mock_client
                    mock_client.head_bucket = Mock()  # Successful bucket test
                    
                    worker = WeeklyTrendWorker(config=test_config)
                    
                    assert worker.s3_client == mock_client
                    mock_boto3.assert_called_with(
                        's3',
                        aws_access_key_id='test_key',
                        aws_secret_access_key='test_secret',
                        region_name='us-east-1'
                    )
    
    @pytest.mark.asyncio
    async def test_s3_client_initialization_no_credentials(self):
        """Test S3 client initialization without credentials"""
        test_config = {'s3_bucket': 'test-bucket'}
        
        with patch('src.services.topic_discovery.weekly_trend_worker.aws_access_key_id', None):
            with patch('src.services.topic_discovery.weekly_trend_worker.aws_secret_access_key', None):
                with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client') as mock_boto3:
                    mock_client = Mock()
                    mock_boto3.return_value = mock_client
                    mock_client.head_bucket = Mock()  # Successful bucket test
                    
                    worker = WeeklyTrendWorker(config=test_config)
                    
                    assert worker.s3_client == mock_client
                    mock_boto3.assert_called_with('s3', region_name='us-east-1')
    
    @pytest.mark.asyncio
    async def test_score_threshold_filtering(self, worker):
        """Test that trends below threshold are filtered out"""
        # Set high threshold
        worker.config['score_threshold'] = 2.0
        
        low_score_trends = [
            {
                'topic': 'Low Score Topic',
                'score': 0.5,  # Below threshold
                'sources': 'Test Source',
                'article_count': 1,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'low',
                'discovered_at': datetime.now(timezone.utc)
            }
        ]
        
        aggregated = await worker._aggregate_and_score_trends(low_score_trends, {})
        
        # Should be filtered out
        assert len(aggregated) == 0
    
    @pytest.mark.asyncio 
    async def test_max_trends_limiting(self, worker):
        """Test that results are limited to max_trends_per_run"""
        worker.config['max_trends_per_run'] = 2
        
        # Create more trends than the limit
        many_trends = []
        for i in range(5):
            many_trends.append({
                'topic': f'Topic {i}',
                'score': 3.0 - (i * 0.1),  # Descending scores
                'sources': 'Test Source',
                'article_count': 1,
                'discovery_method': 'rss_analysis',
                'confidence_level': 'medium',
                'discovered_at': datetime.now(timezone.utc)
            })
        
        aggregated = await worker._aggregate_and_score_trends(many_trends, {})
        
        # Should be limited to max_trends_per_run
        assert len(aggregated) == 2
        
        # Should be highest scoring trends
        assert aggregated[0]['topic'] == 'Topic 0'  # Highest score
        assert aggregated[1]['topic'] == 'Topic 1'  # Second highest