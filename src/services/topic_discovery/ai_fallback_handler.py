# src/services/topic_discovery/ai_fallback_handler.py
"""
AI Integration Error Handling and Fallback Mechanisms

This module provides robust error handling, retry logic, and fallback mechanisms
for AI-powered components to ensure the blog title discovery system remains
operational even when AI services are unavailable or failing.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ai_semantic_analyzer import ImplicitTopic, SemanticInsight, TechnicalConcept
from .blog_title_generator import BlogTitleCandidate
from .enhanced_content_extractor import (
    ArticleContent,
)


class ErrorType(Enum):
    """Types of AI integration errors"""
    RATE_LIMIT = "rate_limit"
    API_TIMEOUT = "api_timeout"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class FallbackMetrics:
    """Metrics tracking for fallback behavior"""
    total_api_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_activations: int = 0
    retry_attempts: int = 0

    # Cost tracking
    estimated_cost: float = 0.0
    cost_savings_from_fallback: float = 0.0

    # Performance tracking
    avg_response_time: float = 0.0
    fallback_response_time: float = 0.0

    # Error distribution
    error_distribution: dict[ErrorType, int] = field(default_factory=dict)


class AIFallbackHandler:
    """
    Comprehensive error handling and fallback system for AI integrations
    with exponential backoff, circuit breaker patterns, and intelligent fallbacks.
    """

    def __init__(self,
                 retry_config: RetryConfig | None = None,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: int = 300):  # 5 minutes
        """
        Initialize AI Fallback Handler
        
        Args:
            retry_config: Configuration for retry behavior
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Seconds to wait before trying circuit again
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_time = None
        self.consecutive_failures = 0

        # Metrics tracking
        self.metrics = FallbackMetrics()

        # Initialize fallback data
        self._initialize_fallback_templates()
        self._initialize_keyword_extractors()

        self.logger.info("AI Fallback Handler initialized")

    def _initialize_fallback_templates(self):
        """Initialize template-based fallback systems"""

        # Template patterns for title generation when AI fails
        self.title_templates = {
            "performance": [
                "How to Optimize {technology} Performance",
                "5 Ways to Improve {technology} Speed",
                "Performance Tips for {technology} Applications",
                "{company} Performance Optimization Strategies"
            ],
            "implementation": [
                "Building Applications with {technology}",
                "Getting Started with {technology}",
                "Complete {technology} Implementation Guide",
                "{technology} Development Best Practices"
            ],
            "comparison": [
                "{tech1} vs {tech2}: Which to Choose?",
                "Comparing {tech1} and {tech2} Features",
                "Choosing Between {tech1} and {tech2}",
                "{tech1} or {tech2} for Your Project?"
            ],
            "how-to": [
                "How to Use {technology} Effectively",
                "Step-by-Step {technology} Tutorial",
                "Learning {technology}: A Beginner's Guide",
                "Mastering {technology} Development"
            ]
        }

        # Semantic analysis fallback keywords
        self.fallback_topics = {
            "frontend": ["React", "Vue", "Angular", "JavaScript", "TypeScript", "CSS", "HTML"],
            "backend": ["Python", "Node.js", "Java", "Go", "API", "Database", "Server"],
            "cloud": ["AWS", "Azure", "Docker", "Kubernetes", "Serverless", "Microservices"],
            "ai_ml": ["Machine Learning", "AI", "Data Science", "Neural Networks", "Deep Learning"]
        }

    def _initialize_keyword_extractors(self):
        """Initialize keyword-based extraction patterns for fallbacks"""

        import re

        # Patterns for extracting information without AI
        self.extraction_patterns = {
            "companies": re.compile(r'\b(Netflix|Google|Amazon|Facebook|Microsoft|Apple|Stripe|Uber|Airbnb|Spotify|GitHub|OpenAI|Anthropic)\b', re.IGNORECASE),
            "technologies": re.compile(r'\b(React|Vue|Angular|Python|JavaScript|TypeScript|Docker|Kubernetes|AWS|Azure|API|GraphQL|REST)\b', re.IGNORECASE),
            "metrics": re.compile(r'\b(\d+(?:\.\d+)?%|\d+x\s+faster|\d+(?:\.\d+)?\s*(?:ms|seconds?|MB|GB))\b', re.IGNORECASE),
            "versions": re.compile(r'\b(?:v?\d+\.\d+(?:\.\d+)?|[A-Z][a-z]+\s+\d+(?:\.\d+)?)\b'),
            "action_words": re.compile(r'\b(how|why|implement|build|create|optimize|improve|solve|fix|debug)\b', re.IGNORECASE)
        }

    async def with_fallback(self,
                           ai_function: Callable,
                           fallback_function: Callable,
                           *args,
                           operation_name: str = "AI Operation",
                           **kwargs) -> Any:
        """
        Execute AI function with comprehensive error handling and fallback
        
        Args:
            ai_function: Primary AI function to execute
            fallback_function: Fallback function if AI fails
            *args: Arguments for the functions
            operation_name: Name of operation for logging
            **kwargs: Keyword arguments for the functions
            
        Returns:
            Result from AI function or fallback function
        """
        # Check circuit breaker
        if self._is_circuit_open():
            self.logger.warning(f"Circuit breaker is open for {operation_name}, using fallback")
            self.metrics.fallback_activations += 1
            return await self._execute_with_timing(fallback_function, *args, **kwargs)

        # Attempt AI function with retries
        last_error = None
        start_time = time.time()

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.metrics.total_api_calls += 1

                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.info(f"Retrying {operation_name} (attempt {attempt + 1}) after {delay:.1f}s")
                    await asyncio.sleep(delay)
                    self.metrics.retry_attempts += 1

                # Execute AI function
                result = await self._execute_with_timing(ai_function, *args, **kwargs)

                # Success - update metrics and reset circuit breaker
                self.metrics.successful_calls += 1
                self.metrics.avg_response_time = self._update_avg_response_time(time.time() - start_time)
                self._reset_circuit_breaker()

                return result

            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

                self.logger.warning(f"{operation_name} attempt {attempt + 1} failed: {error_type.value}")

                # Update error metrics
                self.metrics.failed_calls += 1
                if error_type not in self.metrics.error_distribution:
                    self.metrics.error_distribution[error_type] = 0
                self.metrics.error_distribution[error_type] += 1

                # Check if we should retry
                if not self._should_retry(error_type, attempt):
                    break

                # Update circuit breaker state
                self.consecutive_failures += 1

        # All retries failed - check circuit breaker
        self._check_circuit_breaker()

        # Execute fallback
        self.logger.error(f"{operation_name} failed after all retries, using fallback. Last error: {str(last_error)}")
        self.metrics.fallback_activations += 1

        try:
            fallback_start = time.time()
            result = await self._execute_with_timing(fallback_function, *args, **kwargs)
            self.metrics.fallback_response_time = time.time() - fallback_start
            return result
        except Exception as fallback_error:
            self.logger.error(f"Fallback function also failed: {str(fallback_error)}")
            raise RuntimeError(f"Both {operation_name} and fallback failed") from last_error

    async def _execute_with_timing(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function and track timing"""

        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate handling"""

        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str:
            return ErrorType.API_TIMEOUT
        elif "authentication" in error_str or "unauthorized" in error_str or "api key" in error_str:
            return ErrorType.AUTHENTICATION
        elif "quota" in error_str or "exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "json" in error_str or "parse" in error_str or "invalid" in error_str:
            return ErrorType.INVALID_RESPONSE
        else:
            return ErrorType.UNKNOWN_ERROR

    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if error should trigger a retry"""

        # Don't retry on final attempt
        if attempt >= self.retry_config.max_retries:
            return False

        # Don't retry authentication errors
        if error_type == ErrorType.AUTHENTICATION:
            return False

        # Don't retry quota exceeded (will likely fail again soon)
        if error_type == ErrorType.QUOTA_EXCEEDED:
            return False

        # Retry rate limits, timeouts, network errors, and unknown errors
        return error_type in [ErrorType.RATE_LIMIT, ErrorType.API_TIMEOUT,
                             ErrorType.NETWORK_ERROR, ErrorType.UNKNOWN_ERROR,
                             ErrorType.INVALID_RESPONSE]

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff and jitter"""

        # Exponential backoff
        delay = self.retry_config.base_delay * (self.retry_config.backoff_multiplier ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            jitter = delay * 0.1 * random.random()
            delay += jitter

        return delay

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open"""

        if not self.circuit_open:
            return False

        # Check if timeout has expired
        if self.circuit_open_time and \
           time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            self.logger.info("Circuit breaker timeout expired, attempting to close circuit")
            self.circuit_open = False
            self.circuit_open_time = None
            return False

        return True

    def _check_circuit_breaker(self):
        """Check if circuit breaker should be opened"""

        if self.consecutive_failures >= self.circuit_breaker_threshold and not self.circuit_open:
            self.logger.error(f"Opening circuit breaker after {self.consecutive_failures} consecutive failures")
            self.circuit_open = True
            self.circuit_open_time = time.time()

    def _reset_circuit_breaker(self):
        """Reset circuit breaker after successful operation"""

        if self.consecutive_failures > 0:
            self.logger.info("Resetting circuit breaker after successful operation")
            self.consecutive_failures = 0
            self.circuit_open = False
            self.circuit_open_time = None

    def _update_avg_response_time(self, response_time: float) -> float:
        """Update running average of response times"""

        if self.metrics.avg_response_time == 0:
            return response_time

        # Simple moving average with weight on recent times
        alpha = 0.3  # Weight for new measurement
        return (alpha * response_time) + ((1 - alpha) * self.metrics.avg_response_time)

    # Fallback implementations for specific AI components

    def fallback_semantic_analysis(self, articles: list[ArticleContent]) -> list[SemanticInsight]:
        """Fallback semantic analysis using keyword extraction"""

        self.logger.info("Using fallback semantic analysis (keyword-based)")
        insights = []

        for article in articles:
            try:
                insight = self._create_fallback_semantic_insight(article)
                insights.append(insight)
            except Exception as e:
                self.logger.error(f"Error in fallback semantic analysis for article '{article.title}': {str(e)}")

        return insights

    def _create_fallback_semantic_insight(self, article: ArticleContent) -> SemanticInsight:
        """Create semantic insight using keyword extraction"""

        import hashlib
        article_id = hashlib.md5(article.source_url.encode()).hexdigest()[:12]

        # Extract information using patterns
        content_text = f"{article.title} {article.summary} {article.full_content or ''}".lower()

        # Extract companies
        companies = self.extraction_patterns["companies"].findall(content_text)

        # Extract technologies
        technologies = self.extraction_patterns["technologies"].findall(content_text)

        # Extract metrics
        metrics = self.extraction_patterns["metrics"].findall(content_text)

        # Create implicit topics based on extracted information
        implicit_topics = []
        for tech in technologies[:3]:
            topic = ImplicitTopic(
                topic=f"{tech} Development",
                relevance_score=0.7,
                context=f"Mentioned in context of {article.title}",
                technical_depth="intermediate"
            )
            implicit_topics.append(topic)

        # Create technical concepts
        technical_concepts = []
        if technologies and companies:
            concept = TechnicalConcept(
                concept=f"{technologies[0]} Implementation",
                implementation_approach="Standard industry practices",
                problem_solved="Development challenges",
                technologies_used=technologies[:3],
                complexity_level="intermediate"
            )
            technical_concepts.append(concept)

        # Generate key insights
        key_insights = []
        if metrics:
            key_insights.append(f"Performance improvements mentioned: {', '.join(metrics[:2])}")
        if companies:
            key_insights.append(f"Industry examples from: {', '.join(companies[:2])}")
        if technologies:
            key_insights.append(f"Key technologies: {', '.join(technologies[:3])}")

        return SemanticInsight(
            article_id=article_id,
            source_article=article.source_url,
            implicit_topics=implicit_topics,
            technical_concepts=technical_concepts,
            problems_solved=["Development challenges", "Implementation issues"],
            solutions_implemented=["Best practices", "Framework adoption"],
            performance_metrics=list(set(metrics)),
            key_insights=key_insights or ["Technical content analysis"],
            target_audience="developers",
            content_angle="technical",
            confidence_score=0.6  # Lower confidence for fallback
        )

    def fallback_title_generation(self, insights: list[SemanticInsight], max_titles: int = 20) -> list[BlogTitleCandidate]:
        """Fallback title generation using templates"""

        self.logger.info("Using fallback title generation (template-based)")
        candidates = []

        try:
            for insight in insights[:10]:  # Limit insights to process
                # Extract technologies and companies for template substitution
                technologies = self._extract_technologies_from_insight(insight)
                companies = self._extract_companies_from_insight(insight)

                # Generate titles for each pattern type
                for pattern_type, templates in self.title_templates.items():
                    for template in templates[:2]:  # Use top 2 templates per pattern
                        try:
                            title = self._generate_template_title(template, technologies, companies, pattern_type)
                            if title:
                                candidate = BlogTitleCandidate(
                                    title=title,
                                    pattern_type=pattern_type,
                                    source_insights=[insight.article_id],
                                    generated_by="fallback_template",
                                    confidence=0.5  # Lower confidence for fallback
                                )
                                candidates.append(candidate)

                                if len(candidates) >= max_titles:
                                    break
                        except Exception as e:
                            self.logger.debug(f"Template generation failed: {str(e)}")
                            continue

                if len(candidates) >= max_titles:
                    break

            return candidates[:max_titles]

        except Exception as e:
            self.logger.error(f"Fallback title generation failed: {str(e)}")
            return self._generate_emergency_fallback_titles()

    def _extract_technologies_from_insight(self, insight: SemanticInsight) -> list[str]:
        """Extract technologies from semantic insight"""

        technologies = set()

        # From technical concepts
        for concept in insight.technical_concepts:
            technologies.update(concept.technologies_used)

        # From implicit topics
        for topic in insight.implicit_topics:
            # Simple extraction from topic names
            topic_words = topic.topic.split()
            for word in topic_words:
                if word in ["React", "Vue", "Angular", "Python", "JavaScript", "Docker", "Kubernetes"]:
                    technologies.add(word)

        return list(technologies)[:3]

    def _extract_companies_from_insight(self, insight: SemanticInsight) -> list[str]:
        """Extract companies from semantic insight"""

        companies = set()

        # Simple pattern matching in key insights
        for insight_text in insight.key_insights:
            company_matches = self.extraction_patterns["companies"].findall(insight_text)
            companies.update(company_matches)

        return list(companies)[:2]

    def _generate_template_title(self, template: str, technologies: list[str],
                                companies: list[str], pattern_type: str) -> str | None:
        """Generate title from template with available data"""

        substitutions = {
            "technology": technologies[0] if technologies else "Modern Technology",
            "tech1": technologies[0] if len(technologies) >= 1 else "React",
            "tech2": technologies[1] if len(technologies) >= 2 else "Vue",
            "company": companies[0] if companies else "Leading Companies",
        }

        try:
            # Simple template substitution
            for key, value in substitutions.items():
                template = template.replace(f"{{{key}}}", value)

            # Clean up any remaining template variables
            template = template.replace("{", "").replace("}", "")

            return template
        except Exception:
            return None

    def _generate_emergency_fallback_titles(self) -> list[BlogTitleCandidate]:
        """Generate emergency fallback titles when all else fails"""

        emergency_titles = [
            "5 Essential Web Development Trends",
            "Modern JavaScript Development Practices",
            "Building Scalable Applications",
            "Frontend Framework Comparison",
            "API Design Best Practices",
            "Cloud Deployment Strategies",
            "Performance Optimization Techniques",
            "Security in Web Applications"
        ]

        candidates = []
        for i, title in enumerate(emergency_titles):
            candidate = BlogTitleCandidate(
                title=title,
                pattern_type="general",
                source_insights=[f"emergency_{i}"],
                generated_by="emergency_fallback",
                confidence=0.3  # Very low confidence
            )
            candidates.append(candidate)

        return candidates

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary"""

        total_calls = self.metrics.total_api_calls
        success_rate = (self.metrics.successful_calls / total_calls * 100) if total_calls > 0 else 0
        fallback_rate = (self.metrics.fallback_activations / total_calls * 100) if total_calls > 0 else 0

        return {
            "performance": {
                "total_api_calls": self.metrics.total_api_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "success_rate_percent": round(success_rate, 2),
                "fallback_activations": self.metrics.fallback_activations,
                "fallback_rate_percent": round(fallback_rate, 2),
                "retry_attempts": self.metrics.retry_attempts
            },
            "timing": {
                "avg_api_response_time": round(self.metrics.avg_response_time, 3),
                "avg_fallback_response_time": round(self.metrics.fallback_response_time, 3)
            },
            "circuit_breaker": {
                "currently_open": self.circuit_open,
                "consecutive_failures": self.consecutive_failures,
                "threshold": self.circuit_breaker_threshold
            },
            "error_distribution": {
                error_type.value: count
                for error_type, count in self.metrics.error_distribution.items()
            },
            "cost_estimates": {
                "estimated_total_cost": round(self.metrics.estimated_cost, 4),
                "estimated_savings_from_fallback": round(self.metrics.cost_savings_from_fallback, 4)
            }
        }
