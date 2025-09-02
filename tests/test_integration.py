# tests/test_integration.py
"""
Integration tests for the complete Weekly Trend Worker workflow.
Tests end-to-end functionality with real components but mocked external services.
"""

import pytest
import asyncio
import tempfile
import os
import json
import csv
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from datetime import datetime, timezone

from src.services.topic_discovery.weekly_trend_worker import WeeklyTrendWorker
from src.services.topic_discovery.news_scanner import TechNewsScanner
from src.services.topic_discovery.trend_spotter import TrendSpotter


class TestIntegrationWorkflow:
    """Integration tests for complete workflow"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create directory structure
            (workspace / 'data' / 'backups').mkdir(parents=True)
            (workspace / 'logs').mkdir(parents=True)
            
            yield workspace
    
    @pytest.fixture
    def integration_config(self, temp_workspace):
        """Integration test configuration"""
        return {
            'max_trends_per_run': 20,
            'score_threshold': 0.1,
            'concurrent_execution_check': True,
            'retry_attempts': 2,
            'retry_delay': 1,
            's3_bucket': 'test-integration-bucket',
            's3_key_prefix': 'integration-tests/',
            'local_backup_dir': str(temp_workspace / 'data' / 'backups'),
            'status_file_path': str(temp_workspace / 'data' / 'worker_status.json'),
            'lock_file_path': str(temp_workspace / 'data' / 'worker.lock')
        }
    
    @pytest.fixture
    def mock_rss_responses(self):
        """Mock RSS feed responses for multiple sources"""
        return {
            'Netflix Tech Blog': """<?xml version="1.0"?>
                <rss version="2.0">
                    <channel>
                        <title>Netflix Tech Blog</title>
                        <item>
                            <title>Scaling Machine Learning Infrastructure at Netflix</title>
                            <description>How we built a distributed ML platform using Kubernetes and Python</description>
                            <link>https://netflixtechblog.com/ml-infrastructure</link>
                            <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
                        </item>
                        <item>
                            <title>Microservices Architecture Evolution</title>
                            <description>Our journey from monolith to microservices using Docker containers</description>
                            <link>https://netflixtechblog.com/microservices</link>
                            <pubDate>Sun, 31 Dec 2023 15:00:00 GMT</pubDate>
                        </item>
                    </channel>
                </rss>""",
            
            'AWS Blog': """<?xml version="1.0"?>
                <rss version="2.0">
                    <channel>
                        <title>AWS Blog</title>
                        <item>
                            <title>New AI Services on AWS Cloud Platform</title>
                            <description>Introducing advanced machine learning capabilities for serverless applications</description>
                            <link>https://aws.amazon.com/blogs/ai-services</link>
                            <pubDate>Tue, 02 Jan 2024 08:00:00 GMT</pubDate>
                        </item>
                        <item>
                            <title>Container Security Best Practices</title>
                            <description>Securing Docker containers in Amazon ECS and EKS</description>
                            <link>https://aws.amazon.com/blogs/container-security</link>
                            <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                        </item>
                    </channel>
                </rss>""",
            
            'Facebook Engineering': """<?xml version="1.0"?>
                <rss version="2.0">
                    <channel>
                        <title>Facebook Engineering</title>
                        <item>
                            <title>React 18 Performance Optimizations</title>
                            <description>New concurrent features and performance improvements in React</description>
                            <link>https://engineering.fb.com/react-18</link>
                            <pubDate>Wed, 03 Jan 2024 14:00:00 GMT</pubDate>
                        </item>
                        <item>
                            <title>GraphQL Federation at Scale</title>
                            <description>Building distributed GraphQL APIs across multiple teams</description>
                            <link>https://engineering.fb.com/graphql-federation</link>
                            <pubDate>Tue, 02 Jan 2024 16:00:00 GMT</pubDate>
                        </item>
                    </channel>
                </rss>"""
        }
    
    @pytest.fixture
    def mock_external_trend(self):
        """Mock external trend data"""
        return {
            'topic': 'Kubernetes',
            'score': 0.92,
            'metrics': {
                'search_rank': 2,
                'article_count': 42,
                'tweet_volume': 2500
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(
        self, 
        integration_config, 
        mock_rss_responses, 
        mock_external_trend,
        temp_workspace
    ):
        """Test complete end-to-end workflow"""
        
        # Mock HTTP responses for RSS feeds
        async def mock_get_rss(url):
            """Mock aiohttp responses based on URL"""
            mock_response = Mock()
            
            if 'netflixtechblog.com' in url:
                mock_response.text = AsyncMock(return_value=mock_rss_responses['Netflix Tech Blog'])
            elif 'aws.amazon.com' in url:
                mock_response.text = AsyncMock(return_value=mock_rss_responses['AWS Blog'])
            elif 'engineering.fb.com' in url:
                mock_response.text = AsyncMock(return_value=mock_rss_responses['Facebook Engineering'])
            else:
                # Return empty feed for other sources
                mock_response.text = AsyncMock(return_value='<?xml version="1.0"?><rss><channel></channel></rss>')
            
            mock_response.raise_for_status = Mock()
            return mock_response
        
        # Mock aiohttp session
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = Mock(return_value=Mock(
            __aenter__=AsyncMock(side_effect=lambda: mock_get_rss(mock_session.get.call_args[0][0])),
            __aexit__=AsyncMock(return_value=None)
        ))
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.upload_file = Mock()
        mock_s3_client.head_bucket = Mock()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
                
                # Initialize worker
                worker = WeeklyTrendWorker(config=integration_config)
                worker.s3_client = mock_s3_client
                
                # Mock external trend spotter
                worker.trend_spotter.get_weekly_trend = Mock(return_value=mock_external_trend)
                
                # Run the complete workflow
                results = await worker.run_weekly_discovery()
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['trends_discovered'] > 0
        assert results['error_message'] is None
        assert results['completed_at'] is not None
        
        # Verify CSV file was created and uploaded
        assert results['local_backup_file'] is not None
        assert results['csv_file_uploaded'] is not None
        
        csv_path = Path(results['local_backup_file'])
        assert csv_path.exists()
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0
        
        # Check that we have trends from RSS analysis
        rss_trends = [r for r in rows if r['discovery_method'] == 'rss_analysis']
        assert len(rss_trends) > 0
        
        # Check that external trend was integrated
        external_trends = [r for r in rows if 'api' in r['discovery_method']]
        combined_trends = [r for r in rows if r['discovery_method'] == 'combined']
        assert len(external_trends) > 0 or len(combined_trends) > 0
        
        # Verify expected topics appear
        all_topics = [row['topic'].lower() for row in rows]
        assert any('machine learning' in topic or 'ai' in topic for topic in all_topics)
        assert any('docker' in topic or 'container' in topic for topic in all_topics)
        assert any('kubernetes' in topic for topic in all_topics)
        
        # Verify S3 upload was called
        mock_s3_client.upload_file.assert_called_once()
        
        # Verify status file was created
        status_file = Path(integration_config['status_file_path'])
        assert status_file.exists()
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        
        assert status_data['worker_type'] == 'weekly_trend_worker'
        assert status_data['last_run_status'] == 'completed'
        assert status_data['trends_discovered'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_rss_failures(
        self,
        integration_config,
        mock_external_trend,
        temp_workspace
    ):
        """Test workflow resilience when some RSS feeds fail"""
        
        # Mock mixed success/failure responses
        async def mock_get_with_failures(url):
            """Mock responses with some failures"""
            mock_response = Mock()
            
            if 'netflixtechblog.com' in url:
                # This one succeeds
                mock_response.text = AsyncMock(return_value="""<?xml version="1.0"?>
                    <rss version="2.0">
                        <channel>
                            <item>
                                <title>Working RSS Feed with AI Content</title>
                                <description>This feed works and talks about artificial intelligence</description>
                                <link>https://example.com/ai</link>
                            </item>
                        </channel>
                    </rss>""")
                mock_response.raise_for_status = Mock()
            else:
                # All others fail
                mock_response.raise_for_status = Mock(side_effect=Exception("HTTP 404"))
            
            return mock_response
        
        # Mock aiohttp session with failures
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = Mock(return_value=Mock(
            __aenter__=AsyncMock(side_effect=lambda: mock_get_with_failures(mock_session.get.call_args[0][0])),
            __aexit__=AsyncMock(return_value=None)
        ))
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.upload_file = Mock()
        mock_s3_client.head_bucket = Mock()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
                
                # Initialize worker
                worker = WeeklyTrendWorker(config=integration_config)
                worker.s3_client = mock_s3_client
                
                # Mock external trend spotter
                worker.trend_spotter.get_weekly_trend = Mock(return_value=mock_external_trend)
                
                # Run workflow
                results = await worker.run_weekly_discovery()
        
        # Should still complete successfully despite RSS failures
        assert results['status'] == 'completed'
        assert results['error_message'] is None
        
        # Should have some trends (from working RSS + external)
        assert results['trends_discovered'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_s3_failure_with_local_backup(
        self,
        integration_config,
        mock_rss_responses,
        mock_external_trend,
        temp_workspace
    ):
        """Test workflow when S3 upload fails but local backup succeeds"""
        
        # Mock RSS responses
        async def mock_get_rss(url):
            mock_response = Mock()
            if 'netflixtechblog.com' in url:
                mock_response.text = AsyncMock(return_value=mock_rss_responses['Netflix Tech Blog'])
            else:
                mock_response.text = AsyncMock(return_value='<?xml version="1.0"?><rss><channel></channel></rss>')
            mock_response.raise_for_status = Mock()
            return mock_response
        
        # Mock aiohttp session
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = Mock(return_value=Mock(
            __aenter__=AsyncMock(side_effect=lambda: mock_get_rss(mock_session.get.call_args[0][0])),
            __aexit__=AsyncMock(return_value=None)
        ))
        
        # Mock S3 client with upload failure
        mock_s3_client = Mock()
        mock_s3_client.upload_file = Mock(side_effect=Exception("S3 Upload Failed"))
        mock_s3_client.head_bucket = Mock()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
                
                # Initialize worker
                worker = WeeklyTrendWorker(config=integration_config)
                worker.s3_client = mock_s3_client
                
                # Mock external trend spotter
                worker.trend_spotter.get_weekly_trend = Mock(return_value=mock_external_trend)
                
                # Run workflow
                results = await worker.run_weekly_discovery()
        
        # Should still complete successfully
        assert results['status'] == 'completed'
        assert results['error_message'] is None
        
        # Should have local backup but no S3 upload
        assert results['local_backup_file'] is not None
        assert results['csv_file_uploaded'] is None
        
        # Verify local backup exists and contains data
        csv_path = Path(results['local_backup_file'])
        assert csv_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_prevention(
        self,
        integration_config,
        temp_workspace
    ):
        """Test that concurrent execution is properly prevented"""
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.head_bucket = Mock()
        
        with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
            
            # Initialize first worker
            worker1 = WeeklyTrendWorker(config=integration_config)
            worker1.s3_client = mock_s3_client
            
            # Acquire lock with first worker
            assert worker1._acquire_lock() is True
            
            # Initialize second worker with same config
            worker2 = WeeklyTrendWorker(config=integration_config)
            worker2.s3_client = mock_s3_client
            
            # Mock components to avoid network calls
            worker2.news_scanner.scan_tech_news = AsyncMock(return_value=[])
            worker2.trend_spotter.get_weekly_trend = Mock(return_value={})
            
            # Second worker should detect concurrent execution
            results = await worker2.run_weekly_discovery()
            
            assert results['status'] == 'failed'
            assert 'concurrent' in results['error_message'].lower() or 'running' in results['error_message'].lower()
            
            # Cleanup
            worker1._release_lock()
    
    def test_configuration_validation_and_setup(self, temp_workspace):
        """Test that worker properly validates and sets up configuration"""
        
        # Test with minimal config
        minimal_config = {
            's3_bucket': 'test-bucket',
            'local_backup_dir': str(temp_workspace / 'backups'),
            'status_file_path': str(temp_workspace / 'status.json'),
            'lock_file_path': str(temp_workspace / 'worker.lock')
        }
        
        with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client'):
            worker = WeeklyTrendWorker(config=minimal_config)
        
        # Check that directories were created
        assert Path(minimal_config['local_backup_dir']).exists()
        assert Path(minimal_config['status_file_path']).parent.exists()
        
        # Check that config has defaults for missing values
        assert 'max_trends_per_run' in worker.config
        assert 'score_threshold' in worker.config
        assert 'retry_attempts' in worker.config
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        integration_config,
        temp_workspace
    ):
        """Test error handling and recovery mechanisms"""
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.head_bucket = Mock()
        
        with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
            
            # Initialize worker
            worker = WeeklyTrendWorker(config=integration_config)
            worker.s3_client = mock_s3_client
            
            # Mock news scanner to raise exception
            worker.news_scanner.scan_tech_news = AsyncMock(side_effect=Exception("Critical RSS Error"))
            
            # Run workflow
            results = await worker.run_weekly_discovery()
        
        # Should handle error gracefully
        assert results['status'] == 'failed'
        assert results['error_message'] is not None
        assert 'Critical RSS Error' in results['error_message']
        assert results['completed_at'] is not None
        
        # Status file should be updated even on failure
        status_file = Path(integration_config['status_file_path'])
        assert status_file.exists()
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        
        assert status_data['last_run_status'] == 'failed'
        assert status_data['error_message'] is not None
    
    @pytest.mark.asyncio
    async def test_csv_format_compliance(
        self,
        integration_config,
        mock_rss_responses,
        temp_workspace
    ):
        """Test that CSV output complies with specification format"""
        
        # Mock RSS responses 
        async def mock_get_rss(url):
            mock_response = Mock()
            mock_response.text = AsyncMock(return_value=mock_rss_responses['Netflix Tech Blog'])
            mock_response.raise_for_status = Mock()
            return mock_response
        
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = Mock(return_value=Mock(
            __aenter__=AsyncMock(side_effect=lambda: mock_get_rss(mock_session.get.call_args[0][0])),
            __aexit__=AsyncMock(return_value=None)
        ))
        
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.upload_file = Mock()
        mock_s3_client.head_bucket = Mock()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.services.topic_discovery.weekly_trend_worker.boto3.client', return_value=mock_s3_client):
                
                # Initialize worker
                worker = WeeklyTrendWorker(config=integration_config)
                worker.s3_client = mock_s3_client
                worker.trend_spotter.get_weekly_trend = Mock(return_value={})
                
                # Run workflow
                results = await worker.run_weekly_discovery()
        
        # Read and verify CSV format
        csv_path = Path(results['local_backup_file'])
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames
        
        # Verify required columns from specification
        expected_columns = [
            'topic', 'source', 'score', 'news_score', 'external_score',
            'sources', 'article_count', 'discovery_method', 'confidence_level',
            'discovered_at', 'duplicate_flag'
        ]
        
        assert headers == expected_columns
        
        # Verify data format in rows
        for row in rows:
            assert row['source'] == 'weekly_worker'  # As per specification
            assert row['discovery_method'] in ['rss_analysis', 'api_analysis', 'combined']
            assert row['confidence_level'] in ['low', 'medium', 'high']
            assert row['duplicate_flag'] in ['True', 'False', 'true', 'false']
            
            # Verify numeric fields can be parsed
            assert float(row['score']) >= 0
            assert float(row['news_score']) >= 0
            assert float(row['external_score']) >= 0
            assert int(row['article_count']) >= 0
            
            # Verify timestamp format
            datetime.fromisoformat(row['discovered_at'].replace('Z', '+00:00'))