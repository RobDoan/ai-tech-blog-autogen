# tests/test_ai_semantic_analyzer.py
"""
Unit tests for AI Semantic Analyzer
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.topic_discovery.ai_semantic_analyzer import (
    AISemanticAnalyzer, SemanticInsight, ImplicitTopic, TechnicalConcept
)
from src.services.topic_discovery.enhanced_content_extractor import ArticleContent, TechnicalDetails


class TestAISemanticAnalyzer:
    """Test cases for AISemanticAnalyzer"""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response"""
        return {
            "implicit_topics": [
                {
                    "topic": "GraphQL Federation",
                    "relevance_score": 0.9,
                    "context": "Microservices architecture",
                    "technical_depth": "advanced"
                }
            ],
            "technical_concepts": [
                {
                    "concept": "API Gateway Optimization",
                    "implementation_approach": "GraphQL federation with Apollo",
                    "problem_solved": "Reducing API latency",
                    "technologies_used": ["GraphQL", "Apollo", "Node.js"],
                    "complexity_level": "intermediate",
                    "business_impact": "40% improvement in response time"
                }
            ],
            "problems_solved": ["High API latency", "Complex data fetching"],
            "solutions_implemented": ["GraphQL federation", "Query optimization"],
            "performance_metrics": ["40% latency reduction", "25% fewer API calls"],
            "key_insights": [
                "Federation enables distributed GraphQL architecture",
                "Query batching reduces network overhead",
                "Type-safe schemas improve developer experience"
            ],
            "target_audience": "senior_developers",
            "content_angle": "case_study",
            "confidence_score": 0.85
        }
    
    @pytest.fixture
    def sample_article(self):
        """Sample article for testing"""
        return ArticleContent(
            title="How Netflix Reduced API Latency by 40% with GraphQL Federation",
            summary="Netflix engineering team explains their GraphQL migration",
            full_content="Detailed technical implementation of GraphQL federation...",
            source_url="https://netflixtechblog.com/graphql-federation",
            source_name="Netflix Tech Blog",
            published_date=datetime.now(timezone.utc),
            technical_details=TechnicalDetails(
                metrics=["40% reduction"],
                technologies=["GraphQL", "Node.js"],
                company_names=["Netflix"]
            ),
            word_count=1500
        )
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mock API key"""
        with patch('src.services.topic_discovery.ai_semantic_analyzer.OPENAI_API_KEY', 'test-key'):
            return AISemanticAnalyzer(api_key='test-api-key', timeout=30)
    
    @pytest.mark.asyncio
    async def test_analyze_content_semantics_success(self, analyzer, sample_article, mock_openai_response):
        """Test successful semantic analysis of articles"""
        
        with patch.object(analyzer.client.chat.completions, 'create') as mock_create:
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"result": "success"}'
            mock_create.return_value = mock_response
            
            with patch.object(analyzer, '_parse_ai_response') as mock_parse:
                # Mock parsed response
                mock_insight = SemanticInsight(
                    article_id="test123",
                    source_article=sample_article.source_url,
                    confidence_score=0.85
                )
                mock_parse.return_value = mock_insight
                
                # Execute analysis
                insights = await analyzer.analyze_content_semantics([sample_article])
                
                # Assertions
                assert len(insights) == 1
                assert isinstance(insights[0], SemanticInsight)
                assert insights[0].source_article == sample_article.source_url
                assert insights[0].confidence_score > 0.8
                
                # Verify OpenAI was called
                mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_content_semantics_with_timeout(self, analyzer, sample_article):
        """Test semantic analysis with timeout handling"""
        
        with patch.object(analyzer.client.chat.completions, 'create') as mock_create:
            # Simulate timeout
            mock_create.side_effect = asyncio.TimeoutError("Request timed out")
            
            insights = await analyzer.analyze_content_semantics([sample_article])
            
            # Should handle timeout gracefully
            assert len(insights) == 1  # Returns empty insight on error
            assert insights[0].confidence_score == 0.0
    
    def test_parse_ai_response_success(self, analyzer, mock_openai_response):
        """Test parsing of successful AI response"""
        
        insight = analyzer._parse_ai_response(mock_openai_response, "test123", "https://example.com")
        
        # Check basic properties
        assert insight.article_id == "test123"
        assert insight.source_article == "https://example.com"
        assert insight.confidence_score == 0.85
        
        # Check implicit topics
        assert len(insight.implicit_topics) == 1
        topic = insight.implicit_topics[0]
        assert isinstance(topic, ImplicitTopic)
        assert topic.topic == "GraphQL Federation"
        assert topic.relevance_score == 0.9
        
        # Check technical concepts
        assert len(insight.technical_concepts) == 1
        concept = insight.technical_concepts[0]
        assert isinstance(concept, TechnicalConcept)
        assert concept.concept == "API Gateway Optimization"
        assert "GraphQL" in concept.technologies_used
        
        # Check other fields
        assert "High API latency" in insight.problems_solved
        assert "GraphQL federation" in insight.solutions_implemented
        assert len(insight.performance_metrics) > 0
    
    def test_parse_ai_response_malformed(self, analyzer):
        """Test parsing of malformed AI response"""
        
        malformed_response = {
            "implicit_topics": [
                {
                    "topic": "GraphQL",
                    # Missing required fields
                }
            ],
            "technical_concepts": [],
            "confidence_score": "invalid_float"  # Invalid data type
        }
        
        insight = analyzer._parse_ai_response(malformed_response, "test123", "https://example.com")
        
        # Should handle errors gracefully
        assert insight.article_id == "test123"
        assert insight.confidence_score == 0.5  # Default value
        assert len(insight.implicit_topics) == 1  # Partially parsed
    
    def test_prepare_content_for_analysis(self, analyzer, sample_article):
        """Test content preparation for AI analysis"""
        
        prepared = analyzer._prepare_content_for_analysis(sample_article)
        
        # Check required fields
        assert "title" in prepared
        assert "content" in prepared
        assert "source" in prepared
        assert "technical_details" in prepared
        
        # Check content truncation
        assert len(prepared["content"].split()) <= 1500
        
        # Check technical details structure
        tech_details = prepared["technical_details"]
        assert "metrics" in tech_details
        assert "technologies" in tech_details
        assert "company_names" in tech_details
    
    def test_build_semantic_analysis_prompt(self, analyzer):
        """Test semantic analysis prompt building"""
        
        content = {
            "title": "GraphQL Performance Optimization",
            "content": "Technical content about GraphQL...",
            "source": "Tech Blog",
            "technical_details": {
                "metrics": ["40% improvement"],
                "technologies": ["GraphQL", "Node.js"],
                "company_names": ["Netflix"]
            },
            "content_pattern": "performance"
        }
        
        prompt = analyzer._build_semantic_analysis_prompt(content)
        
        # Check prompt contains key information
        assert "GraphQL Performance Optimization" in prompt
        assert "40% improvement" in prompt
        assert "GraphQL" in prompt
        assert "Netflix" in prompt
    
    @pytest.mark.asyncio
    async def test_analyze_content_relationships(self, analyzer):
        """Test content relationship analysis"""
        
        insights = [
            SemanticInsight(
                article_id="1",
                source_article="https://example.com/1",
                implicit_topics=[
                    ImplicitTopic("GraphQL", 0.9, "API design", "advanced")
                ],
                technical_concepts=[
                    TechnicalConcept("API Federation", "GraphQL", "Latency", ["GraphQL"])
                ]
            ),
            SemanticInsight(
                article_id="2", 
                source_article="https://example.com/2",
                implicit_topics=[
                    ImplicitTopic("Microservices", 0.8, "Architecture", "intermediate")
                ],
                technical_concepts=[
                    TechnicalConcept("Service Mesh", "Istio", "Reliability", ["Kubernetes"])
                ]
            )
        ]
        
        with patch.object(analyzer, '_query_ai_for_relationships') as mock_query:
            mock_query.return_value = {
                "relationships": [],
                "emerging_themes": []
            }
            
            result = await analyzer.analyze_content_relationships(insights)
            
            # Should call relationship analysis
            mock_query.assert_called_once()
            assert "relationships" in result
            assert "emerging_themes" in result
    
    def test_get_system_prompt(self, analyzer):
        """Test system prompt generation"""
        
        prompt = analyzer._get_system_prompt()
        
        # Check prompt contains required elements
        assert "technical content analyzer" in prompt.lower()
        assert "json" in prompt.lower()
        assert "implicit_topics" in prompt
        assert "technical_concepts" in prompt
        assert "confidence_score" in prompt
    
    def test_get_fallback_analysis(self, analyzer):
        """Test fallback analysis generation"""
        
        fallback = analyzer._get_fallback_analysis()
        
        # Check structure
        assert "implicit_topics" in fallback
        assert "technical_concepts" in fallback
        assert "confidence_score" in fallback
        assert fallback["confidence_score"] == 0.1
    
    @pytest.mark.asyncio
    async def test_multiple_articles_processing(self, analyzer, mock_openai_response):
        """Test processing multiple articles with concurrency control"""
        
        articles = [
            ArticleContent(
                title=f"Article {i}",
                summary=f"Summary {i}",
                full_content=None,
                source_url=f"https://example.com/{i}",
                source_name="Test Blog",
                published_date=datetime.now(timezone.utc),
                word_count=500
            ) for i in range(5)
        ]
        
        with patch.object(analyzer, '_analyze_single_article') as mock_analyze:
            mock_analyze.return_value = SemanticInsight(
                article_id="test",
                source_article="https://example.com",
                confidence_score=0.7
            )
            
            insights = await analyzer.analyze_content_semantics(articles)
            
            # Should process all articles
            assert len(insights) == 5
            assert mock_analyze.call_count == 5
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, analyzer, sample_article):
        """Test handling of OpenAI API errors"""
        
        with patch.object(analyzer.client.chat.completions, 'create') as mock_create:
            # Simulate API error
            mock_create.side_effect = Exception("API Error")
            
            insights = await analyzer.analyze_content_semantics([sample_article])
            
            # Should handle error gracefully
            assert len(insights) == 1
            assert insights[0].confidence_score == 0.0  # Error indicator
    
    @pytest.mark.asyncio
    async def test_close_client(self, analyzer):
        """Test proper client cleanup"""
        
        with patch.object(analyzer.client, 'close') as mock_close:
            await analyzer.close()
            mock_close.assert_called_once()
    
    def test_analyzer_initialization_no_api_key(self):
        """Test analyzer initialization without API key"""
        
        with patch('src.services.topic_discovery.ai_semantic_analyzer.OPENAI_API_KEY', None):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                AISemanticAnalyzer(api_key=None)