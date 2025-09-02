# tests/test_news_scanner.py
"""
Unit tests for the refactored TechNewsScanner class.
Tests RSS feed processing, topic extraction, and error handling.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
import feedparser

from src.services.topic_discovery.news_scanner import TechNewsScanner


class TestTechNewsScanner:
    """Test suite for TechNewsScanner"""
    
    @pytest.fixture
    def scanner(self):
        """Create a TechNewsScanner instance for testing"""
        return TechNewsScanner()
    
    @pytest.fixture
    def mock_feed_content(self):
        """Mock RSS feed content"""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Tech Blog</title>
                <item>
                    <title>Revolutionary AI Breakthrough in Machine Learning</title>
                    <description>A new AI model achieves breakthrough performance in natural language processing tasks.</description>
                    <link>https://example.com/ai-breakthrough</link>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Docker Container Security Best Practices</title>
                    <description>Learn how to secure your Docker containers with these essential security practices.</description>
                    <link>https://example.com/docker-security</link>
                    <pubDate>Sun, 31 Dec 2023 15:30:00 GMT</pubDate>
                </item>
                <item>
                    <title>React 18 Performance Optimizations</title>
                    <description>Explore the new performance features in React 18 that can speed up your applications.</description>
                    <link>https://example.com/react-performance</link>
                    <pubDate>Sat, 30 Dec 2023 10:45:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""
    
    def test_scanner_initialization(self, scanner):
        """Test that scanner initializes with correct RSS sources"""
        assert len(scanner.news_sources) == 6
        
        # Check specific sources as per requirements
        source_names = [source['name'] for source in scanner.news_sources]
        assert 'Netflix Tech Blog' in source_names
        assert 'AWS Blog' in source_names
        assert 'Facebook Engineering' in source_names
        assert 'Stripe Engineering' in source_names
        assert 'Hacker News' in source_names
        assert 'GDB Blog' in source_names
        
        # Check that all sources have required fields
        for source in scanner.news_sources:
            assert 'name' in source
            assert 'rss_url' in source
            assert 'weight' in source
            assert 'timeout' in source
            assert isinstance(source['weight'], float)
            assert source['weight'] > 0
    
    @pytest.mark.asyncio
    async def test_scan_rss_feed_success(self, scanner, mock_feed_content):
        """Test successful RSS feed scanning"""
        source = {
            'name': 'Test Blog',
            'rss_url': 'https://example.com/feed',
            'weight': 0.8,
            'timeout': 30
        }
        
        # Mock aiohttp response
        mock_response = Mock()
        mock_response.text = AsyncMock(return_value=mock_feed_content)
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.get = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            articles = await scanner._scan_rss_feed(source)
            
        assert len(articles) == 3
        
        # Check article structure
        article = articles[0]
        assert article['title'] == 'Revolutionary AI Breakthrough in Machine Learning'
        assert 'AI model' in article['description']
        assert article['source'] == 'Test Blog'
        assert article['weight'] == 0.8
        assert 'discovered_at' in article
        assert isinstance(article['discovered_at'], datetime)
    
    @pytest.mark.asyncio
    async def test_scan_rss_feed_http_error(self, scanner):
        """Test RSS feed scanning with HTTP error"""
        source = {
            'name': 'Test Blog',
            'rss_url': 'https://example.com/feed',
            'weight': 0.8,
            'timeout': 30
        }
        
        # Mock HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=None, history=None, status=404
        )
        
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(aiohttp.ClientResponseError):
                await scanner._scan_rss_feed(source)
    
    @pytest.mark.asyncio
    async def test_scan_rss_feed_timeout(self, scanner):
        """Test RSS feed scanning with timeout"""
        source = {
            'name': 'Test Blog', 
            'rss_url': 'https://example.com/feed',
            'weight': 0.8,
            'timeout': 1  # Very short timeout
        }
        
        # Mock timeout
        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.side_effect = asyncio.TimeoutError()
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(asyncio.TimeoutError):
                await scanner._scan_rss_feed(source)
    
    @pytest.mark.asyncio
    async def test_scan_tech_news_with_mixed_results(self, scanner, mock_feed_content):
        """Test scanning with some successful and some failed feeds"""
        
        async def mock_scan_with_error_handling(source):
            """Mock function that simulates mixed success/failure"""
            if 'Netflix' in source['name']:
                return [
                    {
                        'title': 'Netflix Scaling with Microservices',
                        'description': 'How Netflix uses microservices architecture',
                        'source': source['name'],
                        'weight': source['weight'],
                        'discovered_at': datetime.now(timezone.utc)
                    }
                ]
            elif 'AWS' in source['name']:
                return None  # Simulate failure
            else:
                return []  # Simulate no articles
        
        with patch.object(scanner, '_scan_rss_feed_with_error_handling', side_effect=mock_scan_with_error_handling):
            topics = await scanner.scan_tech_news()
        
        # Should return topics from successful sources
        assert isinstance(topics, list)
    
    def test_extract_topics_from_articles(self, scanner):
        """Test topic extraction from articles"""
        articles = [
            {
                'title': 'Machine Learning Breakthrough in AI Research',
                'description': 'Scientists develop new neural network architecture using Python and TensorFlow',
                'source': 'Tech Blog',
                'weight': 0.9,
                'discovered_at': datetime.now(timezone.utc)
            },
            {
                'title': 'Docker Container Security',
                'description': 'Best practices for securing Kubernetes clusters in production',
                'source': 'DevOps Blog', 
                'weight': 0.8,
                'discovered_at': datetime.now(timezone.utc)
            },
            {
                'title': 'React Performance Tips',
                'description': 'Optimizing React applications with new JavaScript features',
                'source': 'Frontend Blog',
                'weight': 0.7,
                'discovered_at': datetime.now(timezone.utc)
            }
        ]
        
        topics = scanner._extract_topics_from_articles(articles)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        
        # Check topic structure
        topic = topics[0]
        required_fields = [
            'topic', 'score', 'news_score', 'external_score', 'source',
            'sources', 'article_count', 'discovery_method', 'confidence_level',
            'discovered_at'
        ]
        
        for field in required_fields:
            assert field in topic
        
        # Check that high-scoring topics appear
        topic_names = [topic['topic'] for topic in topics]
        assert any('Machine Learning' in name or 'AI' in name for name in topic_names)
        assert any('Docker' in name for name in topic_names)
        assert any('React' in name for name in topic_names)
    
    def test_calculate_enhanced_relevance_score(self, scanner):
        """Test enhanced relevance scoring algorithm"""
        # Test title boost
        title_text = "machine learning breakthrough announced"
        full_text = "researchers announce machine learning breakthrough in ai development"
        
        score = scanner._calculate_enhanced_relevance_score(full_text, "machine learning", title_text)
        assert score > 0
        
        # Test without title presence
        no_title_score = scanner._calculate_enhanced_relevance_score(full_text, "machine learning", "other topic")
        assert score > no_title_score  # Title presence should boost score
        
        # Test minimum score
        min_score = scanner._calculate_enhanced_relevance_score("unrelated content", "nonexistent", "title")
        assert min_score == 0.1  # Minimum score threshold
    
    def test_calculate_confidence_level(self, scanner):
        """Test confidence level calculation"""
        # High confidence
        high_confidence = scanner._calculate_confidence_level(
            total_score=10.0, 
            article_count=8, 
            source_count=4, 
            confidence_factors=[]
        )
        assert high_confidence == "high"
        
        # Medium confidence
        medium_confidence = scanner._calculate_confidence_level(
            total_score=3.0,
            article_count=4,
            source_count=2,
            confidence_factors=[]
        )
        assert medium_confidence == "medium"
        
        # Low confidence
        low_confidence = scanner._calculate_confidence_level(
            total_score=1.0,
            article_count=2,
            source_count=1,
            confidence_factors=[]
        )
        assert low_confidence == "low"
    
    def test_rss_sources_configuration(self, scanner):
        """Test that RSS sources match requirements specification"""
        expected_urls = {
            'https://netflixtechblog.com/feed',
            'https://feeds.feedburner.com/GDBcode', 
            'https://engineering.fb.com/feed/',
            'https://aws.amazon.com/blogs/aws/feed/',
            'https://stripe.com/blog/engineering/feed.xml',
            'https://news.ycombinator.com/rss'
        }
        
        actual_urls = {source['rss_url'] for source in scanner.news_sources}
        assert actual_urls == expected_urls
        
        # Check weights are reasonable
        for source in scanner.news_sources:
            assert 0.5 <= source['weight'] <= 1.0
            
        # Check timeouts are configured
        for source in scanner.news_sources:
            assert source['timeout'] >= 10


@pytest.mark.asyncio
async def test_integration_scan_all_feeds():
    """Integration test - scan all configured feeds (requires network)"""
    pytest.skip("Integration test - requires network access")
    
    scanner = TechNewsScanner()
    
    # This would actually hit the real RSS feeds
    topics = await scanner.scan_tech_news()
    
    assert isinstance(topics, list)
    # We can't assert exact counts as feeds change
    print(f"Found {len(topics)} topics from real feeds")
    
    if topics:
        topic = topics[0]
        assert 'topic' in topic
        assert 'score' in topic
        assert topic['source'] == 'rss_analysis'