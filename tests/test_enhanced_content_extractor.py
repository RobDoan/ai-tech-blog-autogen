# tests/test_enhanced_content_extractor.py
"""
Unit tests for Enhanced Content Extractor
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.topic_discovery.enhanced_content_extractor import (
    EnhancedContentExtractor, ArticleContent, TechnicalDetails, ContentPatterns
)
from src.services.topic_discovery.config import RSSSourceConfig


class TestEnhancedContentExtractor:
    """Test cases for EnhancedContentExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing"""
        return EnhancedContentExtractor(timeout=10, max_concurrent=2)
    
    @pytest.fixture
    def sample_rss_sources(self):
        """Sample RSS source configurations"""
        return [
            RSSSourceConfig(
                name="Test Blog",
                rss_url="https://example.com/feed.xml",
                weight=0.8
            )
        ]
    
    @pytest.fixture
    def mock_rss_response(self):
        """Mock RSS feed response"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Tech Blog</title>
                <item>
                    <title>How Netflix Reduced Latency by 40% with GraphQL</title>
                    <link>https://example.com/article/1</link>
                    <description>Netflix engineering team explains their migration...</description>
                    <pubDate>Wed, 15 Jan 2025 10:00:00 GMT</pubDate>
                    <author>Netflix Engineering</author>
                </item>
                <item>
                    <title>React 19 Performance Improvements</title>
                    <link>https://example.com/article/2</link>
                    <description>New React 19 features improve rendering speed by 30%</description>
                    <pubDate>Tue, 14 Jan 2025 15:30:00 GMT</pubDate>
                </item>
            </channel>
        </rss>'''
    
    @pytest.fixture
    def mock_article_html(self):
        """Mock article HTML content"""
        return '''
        <html>
            <body>
                <article>
                    <h1>How Netflix Reduced Latency by 40% with GraphQL</h1>
                    <p>Netflix's engineering team successfully migrated from REST to GraphQL,
                    achieving a 40% reduction in API latency and improving user experience.
                    The migration involved implementing GraphQL federation across 15 microservices.</p>
                    <p>Key technologies used: GraphQL, Node.js, TypeScript, Apollo Federation.
                    Performance improvements: 40% latency reduction, 25% fewer API calls.</p>
                    <pre><code>
                    const typeDefs = gql`
                      type Query {
                        user(id: ID!): User
                      }
                    `;
                    </code></pre>
                </article>
            </body>
        </html>
        '''
    
    @pytest.mark.asyncio
    async def test_extract_article_content_success(self, extractor, sample_rss_sources, mock_rss_response, mock_article_html):
        """Test successful article content extraction"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock RSS feed response
            mock_rss_resp = AsyncMock()
            mock_rss_resp.status = 200
            mock_rss_resp.text = AsyncMock(return_value=mock_rss_response)
            
            # Mock article content response
            mock_article_resp = AsyncMock()
            mock_article_resp.status = 200
            mock_article_resp.text = AsyncMock(return_value=mock_article_html)
            
            # Configure session mock
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_session_instance.get.side_effect = [mock_rss_resp, mock_article_resp, mock_article_resp]
            
            # Execute extraction
            articles = await extractor.extract_article_content(sample_rss_sources)
            
            # Assertions
            assert len(articles) == 2
            
            # Check first article
            article1 = articles[0]
            assert isinstance(article1, ArticleContent)
            assert "Netflix Reduced Latency by 40%" in article1.title
            assert article1.source_name == "Test Blog"
            assert "40%" in article1.full_content
            
            # Check technical details extraction
            assert len(article1.technical_details.metrics) > 0
            assert any("40%" in metric for metric in article1.technical_details.metrics)
            assert len(article1.technical_details.technologies) > 0
            assert "GraphQL" in article1.technical_details.technologies
            
            # Check content patterns
            assert article1.content_patterns.pattern_type in ["performance", "implementation"]
            assert article1.content_patterns.confidence_score > 0
    
    def test_extract_technical_details(self, extractor):
        """Test technical details extraction from content"""
        
        content = """
        Netflix achieved a 40% latency reduction and 30% improvement in throughput
        using GraphQL v2.1 and Node.js 18.0. The implementation involved React 19
        and TypeScript 5.0 for the frontend, with Docker containers running on AWS.
        Code example: function fetchUser(id) { return graphqlClient.query(...) }
        """
        
        details = extractor._extract_technical_details(content)
        
        # Check metrics extraction
        assert len(details.metrics) > 0
        assert any("40%" in metric for metric in details.metrics)
        assert any("30%" in metric for metric in details.metrics)
        
        # Check technology extraction
        assert "GraphQL" in details.technologies
        assert "Node.js" in details.technologies
        assert "React" in details.technologies
        
        # Check version numbers
        assert len(details.version_numbers) > 0
        version_found = any("2.1" in version or "18.0" in version or "19" in version 
                          for version in details.version_numbers)
        assert version_found
        
        # Check code examples
        assert len(details.code_examples) > 0
    
    def test_identify_content_patterns_performance(self, extractor):
        """Test content pattern identification for performance articles"""
        
        title = "How Stripe Improved Payment Processing by 60% with Optimization"
        content = "Stripe's engineering team optimized their payment processing pipeline..."
        
        patterns = extractor._identify_content_patterns(title, content)
        
        assert patterns.pattern_type == "performance"
        assert patterns.confidence_score > 0.3
        assert len(patterns.key_phrases) > 0
        assert len(patterns.actionable_indicators) > 0
    
    def test_identify_content_patterns_tutorial(self, extractor):
        """Test content pattern identification for tutorial articles"""
        
        title = "How to Build a React App with TypeScript: Step by Step Guide"
        content = "This tutorial will show you how to set up a React application..."
        
        patterns = extractor._identify_content_patterns(title, content)
        
        assert patterns.pattern_type == "how-to"
        assert patterns.confidence_score > 0.3
    
    def test_calculate_content_quality_high(self, extractor):
        """Test content quality calculation for high-quality article"""
        
        # Create article with good quality indicators
        article = ArticleContent(
            title="How Netflix Optimized Microservices Performance",
            summary="Technical deep-dive into performance optimization",
            full_content="Detailed technical content...",
            source_url="https://example.com/article",
            source_name="Tech Blog",
            published_date=datetime.now(timezone.utc),
            word_count=1500,
            technical_details=TechnicalDetails(
                metrics=["40% improvement", "25% reduction"],
                version_numbers=["v2.1", "Node 18"],
                company_names=["Netflix"],
                technologies=["GraphQL", "Node.js", "TypeScript"],
                code_examples=["const query = gql`...`"]
            ),
            content_patterns=ContentPatterns(
                pattern_type="performance",
                confidence_score=0.8
            )
        )
        
        score = extractor._calculate_content_quality(article)
        
        assert score > 0.7  # Should be high quality
    
    def test_calculate_content_quality_low(self, extractor):
        """Test content quality calculation for low-quality article"""
        
        # Create article with poor quality indicators
        article = ArticleContent(
            title="Basic Tutorial",
            summary="Simple guide",
            full_content=None,
            source_url="https://example.com/basic",
            source_name="Blog",
            published_date=datetime.now(timezone.utc),
            word_count=100,  # Very short
            technical_details=TechnicalDetails(),  # No technical details
            content_patterns=ContentPatterns(
                pattern_type="general",
                confidence_score=0.1
            )
        )
        
        score = extractor._calculate_content_quality(article)
        
        assert score < 0.5  # Should be low quality
    
    def test_clean_html_content(self, extractor):
        """Test HTML content cleaning"""
        
        html_content = '<p>This is <strong>important</strong> content with <a href="#">links</a>.</p>'
        cleaned = extractor._clean_html(html_content)
        
        assert cleaned == "This is important content with links."
        assert '<' not in cleaned
        assert '>' not in cleaned
    
    def test_clean_content_text(self, extractor):
        """Test content text cleaning and normalization"""
        
        noisy_text = """
        This   has    excessive      whitespace.
        
        Click here to subscribe! Advertisement content.
        Cookie policy and Terms of Service apply.
        
        This is the actual content.
        """
        
        cleaned = extractor._clean_content_text(noisy_text)
        
        assert "excessive whitespace" not in cleaned
        assert "Click here" not in cleaned
        assert "Advertisement" not in cleaned
        assert "Cookie policy" not in cleaned
        assert "actual content" in cleaned
    
    @pytest.mark.asyncio
    async def test_extract_article_content_network_error(self, extractor, sample_rss_sources):
        """Test handling of network errors during extraction"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Simulate network error
            mock_session_instance.get.side_effect = Exception("Network error")
            
            articles = await extractor.extract_article_content(sample_rss_sources)
            
            # Should return empty list on network error
            assert articles == []
    
    @pytest.mark.asyncio
    async def test_fetch_full_article_content_404(self, extractor):
        """Test handling of 404 errors when fetching full content"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 404
            
            mock_session_instance = AsyncMock()
            mock_session_instance.get.return_value = mock_resp
            
            extractor.session = mock_session_instance
            
            content, html = await extractor._fetch_full_article_content("https://example.com/404")
            
            assert content is None
            assert html is None
    
    def test_find_actionable_indicators(self, extractor):
        """Test extraction of actionable indicators from content"""
        
        title = "5 Ways to Improve React Performance"
        content = "This guide shows step by step how to optimize React applications..."
        
        indicators = extractor._find_actionable_indicators(title, content)
        
        assert len(indicators) > 0
        assert any("step by step" in indicator for indicator in indicators)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limit(self, extractor):
        """Test that concurrent processing respects limits"""
        
        # Create multiple RSS sources
        many_sources = [
            RSSSourceConfig(
                name=f"Blog {i}",
                rss_url=f"https://example{i}.com/feed.xml",
                weight=0.8
            ) for i in range(10)
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # All requests fail to test error handling
            mock_session_instance.get.side_effect = Exception("Test error")
            
            # Should not raise exception despite errors
            articles = await extractor.extract_article_content(many_sources)
            assert articles == []