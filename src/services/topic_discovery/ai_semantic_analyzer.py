# src/services/topic_discovery/ai_semantic_analyzer.py
"""
AI Semantic Analyzer for Blog Title Discovery

This module leverages OpenAI's language models to perform deep semantic analysis
of article content, extracting implicit topics, technical concepts, and actionable insights
for generating specific, engaging blog post titles.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI

from src.py_env import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE

from .enhanced_content_extractor import ArticleContent


@dataclass
class ImplicitTopic:
    """Represents an implicit topic extracted from content"""
    topic: str
    relevance_score: float  # 0-1 score of how relevant this topic is
    context: str  # Context where this topic was found
    technical_depth: str  # "beginner", "intermediate", "advanced"


@dataclass
class TechnicalConcept:
    """Represents a technical concept with implementation details"""
    concept: str
    implementation_approach: str
    problem_solved: str
    technologies_used: list[str] = field(default_factory=list)
    complexity_level: str = "intermediate"  # "simple", "intermediate", "complex"
    business_impact: str = ""


@dataclass
class SemanticInsight:
    """Comprehensive semantic analysis results from AI processing"""
    article_id: str  # Unique identifier based on URL hash
    source_article: str  # URL of source article

    # Core semantic analysis
    implicit_topics: list[ImplicitTopic] = field(default_factory=list)
    technical_concepts: list[TechnicalConcept] = field(default_factory=list)
    problems_solved: list[str] = field(default_factory=list)
    solutions_implemented: list[str] = field(default_factory=list)
    performance_metrics: list[str] = field(default_factory=list)

    # AI analysis metadata
    confidence_score: float = 0.0  # Overall confidence in analysis
    processing_time: float = 0.0  # Time taken for analysis
    ai_model_used: str = ""
    analysis_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Content synthesis
    key_insights: list[str] = field(default_factory=list)  # Top insights for title generation
    target_audience: str = "developers"  # Inferred audience
    content_angle: str = "technical"  # "tutorial", "analysis", "comparison", etc.


class AISemanticAnalyzer:
    """
    AI-powered semantic analyzer that extracts deep insights from technical articles
    using OpenAI's language models for intelligent blog title generation.
    """

    def __init__(self,
                 api_key: str | None = None,
                 model: str = OPENAI_MODEL,
                 temperature: float = OPENAI_TEMPERATURE,
                 timeout: int = 60,
                 max_retries: int = 3):
        """
        Initialize AI Semantic Analyzer
        
        Args:
            api_key: OpenAI API key (uses environment default if None)
            model: OpenAI model to use for analysis
            temperature: Model temperature for controlled output
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.logger = logging.getLogger(__name__)

        # OpenAI client configuration
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("OpenAI API key is required for semantic analysis")

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        self.logger.info(f"AI Semantic Analyzer initialized with model: {self.model}")

    async def analyze_content_semantics(self, articles: list[ArticleContent]) -> list[SemanticInsight]:
        """
        Perform semantic analysis on multiple articles using AI
        
        Args:
            articles: List of article content to analyze
            
        Returns:
            List of semantic insights extracted from articles
        """
        if not articles:
            return []

        self.logger.info(f"Starting AI semantic analysis for {len(articles)} articles")

        # Process articles with concurrency control
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
        tasks = [
            self._analyze_single_article(semaphore, article)
            for article in articles
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return successful analyses
        insights = []
        for result in results:
            if isinstance(result, SemanticInsight):
                insights.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Semantic analysis error: {str(result)}")

        self.logger.info(f"Completed semantic analysis for {len(insights)} articles")
        return insights

    async def _analyze_single_article(self, semaphore: asyncio.Semaphore, article: ArticleContent) -> SemanticInsight:
        """Analyze a single article using AI"""

        async with semaphore:
            start_time = datetime.now()

            try:
                # Create unique identifier for article
                import hashlib
                article_id = hashlib.md5(article.source_url.encode()).hexdigest()[:12]

                # Prepare content for analysis
                analysis_content = self._prepare_content_for_analysis(article)

                # Perform AI semantic analysis
                ai_response = await self._query_ai_for_semantics(analysis_content)

                # Parse AI response into structured insights
                insights = self._parse_ai_response(ai_response, article_id, article.source_url)

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                insights.processing_time = processing_time
                insights.ai_model_used = self.model

                self.logger.debug(f"Analyzed article '{article.title[:50]}...' in {processing_time:.2f}s")
                return insights

            except Exception as e:
                self.logger.error(f"Error analyzing article '{article.title}': {str(e)}")
                # Return empty insights for failed analysis
                return SemanticInsight(
                    article_id=f"error_{datetime.now().timestamp()}",
                    source_article=article.source_url,
                    confidence_score=0.0
                )

    def _prepare_content_for_analysis(self, article: ArticleContent) -> dict[str, Any]:
        """Prepare article content for AI analysis"""

        # Use full content if available, otherwise use summary
        main_content = article.full_content or article.summary

        # Limit content size to stay within token limits
        content_words = main_content.split()
        if len(content_words) > 1500:  # Roughly 2000 tokens with some buffer
            main_content = " ".join(content_words[:1500])

        return {
            "title": article.title,
            "content": main_content,
            "source": article.source_name,
            "technical_details": {
                "metrics": article.technical_details.metrics,
                "technologies": article.technical_details.technologies,
                "company_names": article.technical_details.company_names,
                "version_numbers": article.technical_details.version_numbers
            },
            "content_pattern": article.content_patterns.pattern_type,
            "word_count": article.word_count
        }

    async def _query_ai_for_semantics(self, content: dict[str, Any]) -> dict[str, Any]:
        """Query AI model for semantic analysis"""

        # Construct analysis prompt
        prompt = self._build_semantic_analysis_prompt(content)

        try:
            # Make API request to OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1500,  # Sufficient for structured analysis
                response_format={"type": "json_object"}  # Request JSON response
            )

            # Parse JSON response
            response_content = response.choices[0].message.content
            return json.loads(response_content)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response as JSON: {str(e)}")
            return self._get_fallback_analysis()

        except Exception as e:
            self.logger.error(f"AI API request failed: {str(e)}")
            return self._get_fallback_analysis()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for semantic analysis"""

        return """You are an expert technical content analyzer specializing in extracting semantic insights from technology articles for blog title generation.

Your task is to perform deep semantic analysis of technical articles and extract:
1. Implicit topics not explicitly stated in titles
2. Technical concepts with implementation details  
3. Problems solved and solutions implemented
4. Performance improvements with specific metrics
5. Key insights for generating specific, actionable blog titles

Respond ONLY with valid JSON following this exact structure:
{
  "implicit_topics": [
    {
      "topic": "specific topic name",
      "relevance_score": 0.8,
      "context": "context where found",
      "technical_depth": "beginner|intermediate|advanced"
    }
  ],
  "technical_concepts": [
    {
      "concept": "concept name",
      "implementation_approach": "how it was implemented",
      "problem_solved": "problem it addresses",
      "technologies_used": ["tech1", "tech2"],
      "complexity_level": "simple|intermediate|complex",
      "business_impact": "business value created"
    }
  ],
  "problems_solved": ["problem1", "problem2"],
  "solutions_implemented": ["solution1", "solution2"],
  "performance_metrics": ["metric1", "metric2"],
  "key_insights": ["insight1", "insight2", "insight3"],
  "target_audience": "developers|architects|managers|beginners",
  "content_angle": "tutorial|analysis|comparison|case_study|implementation",
  "confidence_score": 0.85
}

Focus on extracting specific, actionable insights that would make compelling blog post titles.
Prioritize concrete technical details over general concepts."""

    def _build_semantic_analysis_prompt(self, content: dict[str, Any]) -> str:
        """Build the analysis prompt for AI"""

        prompt = f"""
Analyze this technical article for semantic insights:

**Title:** {content['title']}

**Source:** {content['source']}

**Content Pattern:** {content['content_pattern']}

**Technical Details Found:**
- Metrics: {', '.join(content['technical_details']['metrics'][:5])}
- Technologies: {', '.join(content['technical_details']['technologies'][:5])}
- Companies: {', '.join(content['technical_details']['company_names'][:3])}
- Versions: {', '.join(content['technical_details']['version_numbers'][:3])}

**Article Content:**
{content['content'][:3000]}

Perform deep semantic analysis focusing on:
1. What specific technical problems are being solved?
2. What implementation approaches are described?
3. What performance improvements or metrics are mentioned?
4. What implicit topics could generate actionable blog titles?
5. What are the key insights for content creators?

Extract insights that would lead to specific blog titles like:
- "How [Company] reduced [Metric] by [Percentage] using [Technology]"
- "Why [Technology] outperforms [Alternative] for [Use Case]"
- "[Number] ways to optimize [Process] with [Technology]"
"""

        return prompt.strip()

    def _parse_ai_response(self, ai_response: dict[str, Any], article_id: str, source_url: str) -> SemanticInsight:
        """Parse AI response into structured SemanticInsight"""

        try:
            # Extract implicit topics
            implicit_topics = []
            for topic_data in ai_response.get("implicit_topics", []):
                topic = ImplicitTopic(
                    topic=topic_data.get("topic", ""),
                    relevance_score=float(topic_data.get("relevance_score", 0.5)),
                    context=topic_data.get("context", ""),
                    technical_depth=topic_data.get("technical_depth", "intermediate")
                )
                implicit_topics.append(topic)

            # Extract technical concepts
            technical_concepts = []
            for concept_data in ai_response.get("technical_concepts", []):
                concept = TechnicalConcept(
                    concept=concept_data.get("concept", ""),
                    implementation_approach=concept_data.get("implementation_approach", ""),
                    problem_solved=concept_data.get("problem_solved", ""),
                    technologies_used=concept_data.get("technologies_used", []),
                    complexity_level=concept_data.get("complexity_level", "intermediate"),
                    business_impact=concept_data.get("business_impact", "")
                )
                technical_concepts.append(concept)

            # Create semantic insight
            insight = SemanticInsight(
                article_id=article_id,
                source_article=source_url,
                implicit_topics=implicit_topics,
                technical_concepts=technical_concepts,
                problems_solved=ai_response.get("problems_solved", []),
                solutions_implemented=ai_response.get("solutions_implemented", []),
                performance_metrics=ai_response.get("performance_metrics", []),
                key_insights=ai_response.get("key_insights", []),
                target_audience=ai_response.get("target_audience", "developers"),
                content_angle=ai_response.get("content_angle", "technical"),
                confidence_score=float(ai_response.get("confidence_score", 0.5))
            )

            return insight

        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
            return SemanticInsight(
                article_id=article_id,
                source_article=source_url,
                confidence_score=0.1
            )

    def _get_fallback_analysis(self) -> dict[str, Any]:
        """Provide fallback analysis when AI fails"""

        return {
            "implicit_topics": [],
            "technical_concepts": [],
            "problems_solved": [],
            "solutions_implemented": [],
            "performance_metrics": [],
            "key_insights": ["AI analysis failed - using fallback"],
            "target_audience": "developers",
            "content_angle": "technical",
            "confidence_score": 0.1
        }

    async def analyze_content_relationships(self, insights: list[SemanticInsight]) -> dict[str, Any]:
        """
        Analyze relationships between multiple articles to identify patterns and trends
        
        Args:
            insights: List of semantic insights to analyze for relationships
            
        Returns:
            Dictionary containing relationship analysis
        """
        if len(insights) < 2:
            return {"relationships": [], "emerging_themes": []}

        self.logger.info(f"Analyzing content relationships across {len(insights)} articles")

        try:
            # Prepare data for relationship analysis
            relationship_data = self._prepare_relationship_data(insights)

            # Query AI for relationship analysis
            ai_response = await self._query_ai_for_relationships(relationship_data)

            return ai_response

        except Exception as e:
            self.logger.error(f"Error in relationship analysis: {str(e)}")
            return {"relationships": [], "emerging_themes": []}

    def _prepare_relationship_data(self, insights: list[SemanticInsight]) -> dict[str, Any]:
        """Prepare insights data for relationship analysis"""

        # Extract key data for analysis
        articles_data = []
        for insight in insights[:10]:  # Limit to 10 most recent for token management
            articles_data.append({
                "source": insight.source_article,
                "topics": [t.topic for t in insight.implicit_topics][:3],
                "concepts": [c.concept for c in insight.technical_concepts][:3],
                "problems": insight.problems_solved[:3],
                "solutions": insight.solutions_implemented[:3],
                "audience": insight.target_audience,
                "angle": insight.content_angle
            })

        return {"articles": articles_data}

    async def _query_ai_for_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Query AI for relationship analysis"""

        prompt = f"""
Analyze these technical articles for relationships and emerging themes:

{json.dumps(data, indent=2)}

Identify:
1. Common topics appearing across multiple articles
2. Emerging technology trends
3. Related problems being solved
4. Potential content series opportunities

Respond with JSON:
{{
  "relationships": [
    {{
      "theme": "theme name",
      "related_articles": ["url1", "url2"],
      "connection_strength": 0.8,
      "description": "how they relate"
    }}
  ],
  "emerging_themes": [
    {{
      "theme": "theme name",
      "frequency": 3,
      "trend_strength": 0.9,
      "related_concepts": ["concept1", "concept2"]
    }}
  ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing technical content relationships and trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"Relationship analysis AI request failed: {str(e)}")
            return {"relationships": [], "emerging_themes": []}

    async def close(self):
        """Close the async OpenAI client"""
        if hasattr(self.client, 'close'):
            await self.client.close()
