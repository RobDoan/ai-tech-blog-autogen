# tests/test_ai_fallback_handler.py
"""
Unit tests for AI Fallback Handler
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.topic_discovery.ai_fallback_handler import (
    AIFallbackHandler, ErrorType, RetryConfig, FallbackMetrics
)
from src.services.topic_discovery.enhanced_content_extractor import (
    ArticleContent, TechnicalDetails
)
from src.services.topic_discovery.ai_semantic_analyzer import SemanticInsight
from src.services.topic_discovery.blog_title_generator import BlogTitleCandidate


class TestAIFallbackHandler:
    """Test cases for AIFallbackHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create fallback handler instance"""
        return AIFallbackHandler(
            retry_config=RetryConfig(max_retries=2, base_delay=0.1),
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=5
        )
    
    @pytest.fixture
    def sample_article(self):
        """Sample article for testing fallback methods"""
        return ArticleContent(
            title="How Netflix Improved Performance with GraphQL",
            summary="Netflix reduced API latency by 40% using GraphQL federation",
            full_content="Detailed implementation of GraphQL federation at Netflix...",
            source_url="https://netflixtechblog.com/graphql-performance",
            source_name="Netflix Tech Blog",
            published_date=datetime.now(timezone.utc),
            technical_details=TechnicalDetails(
                metrics=["40% reduction"],
                technologies=["GraphQL", "Node.js"],
                company_names=["Netflix"]
            ),
            word_count=1200
        )
    
    @pytest.mark.asyncio
    async def test_with_fallback_success(self, handler):
        """Test successful AI function execution without fallback"""
        
        async def successful_ai_function():
            return "AI Success"
        
        def fallback_function():
            return "Fallback Result"
        
        result = await handler.with_fallback(
            successful_ai_function,
            fallback_function,
            operation_name="Test Operation"
        )
        
        assert result == "AI Success"
        assert handler.metrics.successful_calls == 1
        assert handler.metrics.fallback_activations == 0
    
    @pytest.mark.asyncio
    async def test_with_fallback_after_retries(self, handler):
        """Test fallback activation after exhausting retries"""
        
        async def failing_ai_function():
            raise Exception("Network error")
        
        def fallback_function():
            return "Fallback Result"
        
        result = await handler.with_fallback(
            failing_ai_function,
            fallback_function,
            operation_name="Test Operation"
        )
        
        assert result == "Fallback Result"
        assert handler.metrics.failed_calls > 0
        assert handler.metrics.fallback_activations == 1
        assert handler.metrics.retry_attempts > 0
    
    @pytest.mark.asyncio
    async def test_with_fallback_circuit_breaker_open(self, handler):
        """Test circuit breaker preventing AI calls"""
        
        # Force circuit breaker open
        handler.circuit_open = True
        handler.circuit_open_time = 0  # Just opened
        
        async def ai_function():
            return "AI Success"
        
        def fallback_function():
            return "Circuit Breaker Fallback"
        
        result = await handler.with_fallback(
            ai_function,
            fallback_function,
            operation_name="Test Operation"
        )
        
        assert result == "Circuit Breaker Fallback"
        assert handler.metrics.fallback_activations == 1
        assert handler.metrics.total_api_calls == 0  # No AI calls made
    
    def test_classify_error_types(self, handler):
        """Test error classification for different error types"""
        
        test_cases = [
            (Exception("Rate limit exceeded"), ErrorType.RATE_LIMIT),
            (Exception("Request timeout"), ErrorType.API_TIMEOUT),
            (Exception("Invalid API key"), ErrorType.AUTHENTICATION),
            (Exception("Quota exceeded"), ErrorType.QUOTA_EXCEEDED),
            (Exception("Network connection failed"), ErrorType.NETWORK_ERROR),
            (Exception("Invalid JSON response"), ErrorType.INVALID_RESPONSE),
            (Exception("Unknown error"), ErrorType.UNKNOWN_ERROR)
        ]
        
        for error, expected_type in test_cases:
            classified = handler._classify_error(error)
            assert classified == expected_type
    
    def test_should_retry_logic(self, handler):
        """Test retry decision logic for different error types"""
        
        # Should retry these errors
        retryable_errors = [
            ErrorType.RATE_LIMIT,
            ErrorType.API_TIMEOUT,
            ErrorType.NETWORK_ERROR,
            ErrorType.UNKNOWN_ERROR,
            ErrorType.INVALID_RESPONSE
        ]
        
        for error_type in retryable_errors:
            assert handler._should_retry(error_type, 0)  # First attempt
        
        # Should not retry these errors
        non_retryable_errors = [
            ErrorType.AUTHENTICATION,
            ErrorType.QUOTA_EXCEEDED
        ]
        
        for error_type in non_retryable_errors:
            assert not handler._should_retry(error_type, 0)
    
    def test_calculate_retry_delay(self, handler):
        """Test retry delay calculation with exponential backoff"""
        
        # Test exponential backoff
        delay1 = handler._calculate_retry_delay(1)
        delay2 = handler._calculate_retry_delay(2)
        
        assert delay2 > delay1
        
        # Test max delay cap
        delay_high = handler._calculate_retry_delay(10)
        assert delay_high <= handler.retry_config.max_delay
    
    def test_circuit_breaker_opening(self, handler):
        """Test circuit breaker opening after consecutive failures"""
        
        # Simulate consecutive failures
        handler.consecutive_failures = handler.circuit_breaker_threshold
        handler._check_circuit_breaker()
        
        assert handler.circuit_open
        assert handler.circuit_open_time is not None
    
    def test_circuit_breaker_reset(self, handler):
        """Test circuit breaker reset after successful operation"""
        
        # Set up circuit breaker state
        handler.consecutive_failures = 2
        handler.circuit_open = False
        
        # Reset after success
        handler._reset_circuit_breaker()
        
        assert handler.consecutive_failures == 0
        assert not handler.circuit_open
    
    def test_fallback_semantic_analysis(self, handler, sample_article):
        """Test keyword-based fallback semantic analysis"""
        
        insights = handler.fallback_semantic_analysis([sample_article])
        
        assert len(insights) == 1
        insight = insights[0]
        
        # Check basic structure
        assert isinstance(insight, SemanticInsight)
        assert insight.source_article == sample_article.source_url
        assert insight.confidence_score == 0.6  # Fallback confidence
        
        # Check content extraction
        assert len(insight.implicit_topics) > 0
        assert len(insight.technical_concepts) > 0
        assert "GraphQL" in str(insight.technical_concepts)
    
    def test_create_fallback_semantic_insight(self, handler, sample_article):
        """Test creation of fallback semantic insight"""
        
        insight = handler._create_fallback_semantic_insight(sample_article)
        
        # Check extracted technologies
        assert any("GraphQL" in str(topic) for topic in insight.implicit_topics)
        
        # Check extracted metrics
        assert "40%" in " ".join(insight.performance_metrics)
        
        # Check key insights generation
        assert len(insight.key_insights) > 0
    
    def test_fallback_title_generation(self, handler):
        """Test template-based fallback title generation"""
        
        insights = [
            SemanticInsight(
                article_id="test1",
                source_article="https://example.com/1",
                technical_concepts=[],
                key_insights=["GraphQL performance optimization"],
                confidence_score=0.8
            )
        ]
        
        titles = handler.fallback_title_generation(insights, max_titles=5)
        
        assert len(titles) > 0
        assert all(isinstance(t, BlogTitleCandidate) for t in titles)
        assert all(t.generated_by == "fallback_template" for t in titles)
        assert all(t.confidence == 0.5 for t in titles)  # Fallback confidence
    
    def test_generate_template_title(self, handler):
        """Test individual template title generation"""
        
        template = "How to Build {technology} Applications"
        technologies = ["React", "Vue"]
        companies = ["Netflix"]
        pattern_type = "tutorial"
        
        title = handler._generate_template_title(template, technologies, companies, pattern_type)
        
        assert title is not None
        assert "React" in title
        assert "Applications" in title
    
    def test_extract_technologies_from_insight(self, handler):
        """Test technology extraction from semantic insights"""
        
        from src.services.topic_discovery.ai_semantic_analyzer import TechnicalConcept, ImplicitTopic
        
        insight = SemanticInsight(
            article_id="test",
            source_article="https://example.com",
            technical_concepts=[
                TechnicalConcept(
                    concept="API Design",
                    implementation_approach="GraphQL",
                    problem_solved="Performance",
                    technologies_used=["GraphQL", "Apollo", "Node.js"]
                )
            ],
            implicit_topics=[
                ImplicitTopic("React Development", 0.8, "Frontend", "intermediate")
            ]
        )
        
        technologies = handler._extract_technologies_from_insight(insight)
        
        assert "GraphQL" in technologies
        assert "Apollo" in technologies
        assert "React" in technologies  # From topic
    
    def test_generate_emergency_fallback_titles(self, handler):
        """Test emergency fallback title generation"""
        
        emergency_titles = handler._generate_emergency_fallback_titles()
        
        assert len(emergency_titles) > 0
        assert all(isinstance(t, BlogTitleCandidate) for t in emergency_titles)
        assert all(t.generated_by == "emergency_fallback" for t in emergency_titles)
        assert all(t.confidence == 0.3 for t in emergency_titles)  # Very low confidence
    
    @pytest.mark.asyncio
    async def test_retry_with_rate_limit(self, handler):
        """Test retry behavior with rate limit errors"""
        
        call_count = 0
        
        async def rate_limited_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Rate limit exceeded")
            return "Success after retries"
        
        def fallback_function():
            return "Fallback"
        
        result = await handler.with_fallback(
            rate_limited_function,
            fallback_function,
            operation_name="Rate Limit Test"
        )
        
        assert result == "Success after retries"
        assert call_count == 3  # Initial + 2 retries
        assert handler.metrics.retry_attempts == 2
    
    @pytest.mark.asyncio 
    async def test_authentication_error_no_retry(self, handler):
        """Test that authentication errors are not retried"""
        
        call_count = 0
        
        async def auth_error_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid API key")
        
        def fallback_function():
            return "Auth Fallback"
        
        result = await handler.with_fallback(
            auth_error_function,
            fallback_function,
            operation_name="Auth Test"
        )
        
        assert result == "Auth Fallback"
        assert call_count == 1  # No retries for auth errors
        assert handler.metrics.retry_attempts == 0
    
    def test_metrics_summary_generation(self, handler):
        """Test comprehensive metrics summary generation"""
        
        # Set up some metrics
        handler.metrics.total_api_calls = 10
        handler.metrics.successful_calls = 7
        handler.metrics.failed_calls = 3
        handler.metrics.fallback_activations = 2
        handler.metrics.retry_attempts = 5
        handler.metrics.error_distribution[ErrorType.RATE_LIMIT] = 2
        handler.metrics.error_distribution[ErrorType.NETWORK_ERROR] = 1
        
        summary = handler.get_metrics_summary()
        
        # Check structure
        assert "performance" in summary
        assert "timing" in summary
        assert "circuit_breaker" in summary
        assert "error_distribution" in summary
        
        # Check calculated values
        assert summary["performance"]["success_rate_percent"] == 70.0
        assert summary["performance"]["fallback_rate_percent"] == 20.0
        
        # Check error distribution
        assert "rate_limit" in summary["error_distribution"]
        assert summary["error_distribution"]["rate_limit"] == 2
    
    def test_update_avg_response_time(self, handler):
        """Test response time tracking"""
        
        # First measurement
        avg1 = handler._update_avg_response_time(1.0)
        assert avg1 == 1.0
        
        # Update with second measurement
        handler.metrics.avg_response_time = avg1
        avg2 = handler._update_avg_response_time(2.0)
        
        # Should be weighted average
        assert 1.0 < avg2 < 2.0
    
    @pytest.mark.asyncio
    async def test_fallback_function_error(self, handler):
        """Test handling when both AI function and fallback fail"""
        
        async def failing_ai_function():
            raise Exception("AI Error")
        
        def failing_fallback():
            raise Exception("Fallback Error")
        
        with pytest.raises(RuntimeError, match="Both .* and fallback failed"):
            await handler.with_fallback(
                failing_ai_function,
                failing_fallback,
                operation_name="Double Failure Test"
            )
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration"""
        
        custom_config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=1.5
        )
        
        handler = AIFallbackHandler(
            retry_config=custom_config,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=600
        )
        
        assert handler.retry_config.max_retries == 5
        assert handler.retry_config.base_delay == 0.5
        assert handler.circuit_breaker_threshold == 10
        assert handler.circuit_breaker_timeout == 600