# src/services/topic_discovery/blog_title_generator.py
"""
Blog Title Generator with AI-Powered Creation

This module generates specific, actionable blog post titles using OpenAI's language models
based on semantic insights extracted from technical articles. It creates titles that follow
proven engagement patterns and include concrete details.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI

from src.py_env import OPENAI_API_KEY, OPENAI_MODEL

from .ai_semantic_analyzer import SemanticInsight


@dataclass
class BlogTitleCandidate:
    """Represents a generated blog title candidate with metadata"""

    title: str
    pattern_type: str  # "performance", "comparison", "implementation", "problem-solution", "how-to"
    source_insights: list[str] = field(default_factory=list)  # Source insight IDs
    generated_by: str = "ai_semantic_analysis"

    # Title analysis
    specificity_score: float = 0.0  # 0-1 based on concrete details
    engagement_score: float = 0.0  # 0-1 based on actionability
    technical_depth: str = "intermediate"  # "beginner", "intermediate", "advanced"

    # Supporting data
    key_technologies: list[str] = field(default_factory=list)
    metrics_mentioned: list[str] = field(default_factory=list)
    companies_mentioned: list[str] = field(default_factory=list)

    # Generation metadata
    confidence: float = 0.0  # AI confidence in title quality
    generation_reasoning: str = ""  # Why this title was generated
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BlogTitleGenerator:
    """
    AI-powered blog title generator that creates specific, engaging titles
    based on semantic insights from technical articles.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = OPENAI_MODEL,
        temperature: float = 0.4,  # Slightly more creative for titles
        timeout: int = 45,
        max_retries: int = 3,
    ):
        """
        Initialize Blog Title Generator

        Args:
            api_key: OpenAI API key
            model: OpenAI model for title generation
            temperature: Model temperature (0.4 for balanced creativity/accuracy)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.logger = logging.getLogger(__name__)

        # OpenAI configuration
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("OpenAI API key is required for title generation")

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries
        )

        # Title pattern templates for fallback
        self._initialize_title_templates()

        self.logger.info(f"Blog Title Generator initialized with model: {self.model}")

    def _initialize_title_templates(self):
        """Initialize template patterns for different title types"""

        self.title_templates = {
            "performance": [
                "How {company} {action} {metric} by {amount} using {technology}",
                "Why {company} achieved {metric} improvement with {technology}",
                "{company}'s {metric} optimization using {technology}",
                "How {technology} helped {company} {action} {metric} by {amount}",
            ],
            "comparison": [
                "Why {tech1} outperforms {tech2} for {use_case}",
                "{tech1} vs {tech2}: Which is better for {use_case}?",
                "How {tech1} compares to {tech2} in {metric}",
                "Choosing between {tech1} and {tech2} for {use_case}",
            ],
            "implementation": [
                "How to implement {technology} for {use_case}",
                "{number} ways to optimize {process} with {technology}",
                "Building {solution} with {technology}: A complete guide",
                "Implementing {technology}: Lessons from {company}",
            ],
            "problem-solution": [
                "How {company} solved {problem} with {technology}",
                "Solving {problem}: {company}'s approach using {technology}",
                "How to fix {problem} using {technology}",
                "{company}'s innovative solution to {problem}",
            ],
            "how-to": [
                "How to {action} using {technology}",
                "A guide to {process} with {technology}",
                "{number} steps to {outcome} using {technology}",
                "How to get started with {technology} for {use_case}",
            ],
        }

    async def generate_specific_titles(
        self, insights: list[SemanticInsight], max_titles: int = 30
    ) -> list[BlogTitleCandidate]:
        """
        Generate specific, actionable blog titles from semantic insights

        Args:
            insights: List of semantic insights from articles
            max_titles: Maximum number of titles to generate

        Returns:
            List of generated blog title candidates
        """
        if not insights:
            return []

        self.logger.info(
            f"Generating blog titles from {len(insights)} semantic insights"
        )

        # Filter high-quality insights
        quality_insights = [
            insight
            for insight in insights
            if insight.confidence_score > 0.3
            and (
                insight.technical_concepts
                or insight.performance_metrics
                or insight.key_insights
            )
        ]

        if not quality_insights:
            self.logger.warning(
                "No high-quality insights available for title generation"
            )
            return []

        # Process insights with concurrency control
        semaphore = asyncio.Semaphore(
            2
        )  # Limit to 2 concurrent title generation requests
        tasks = []

        # Generate titles for each insight
        for insight in quality_insights[:10]:  # Process up to 10 insights
            task = self._generate_titles_for_insight(semaphore, insight)
            tasks.append(task)

        # Collect all generated titles
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_titles = []
        for result in results:
            if isinstance(result, list):
                all_titles.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Title generation error: {str(result)}")

        # Remove duplicates and rank by quality
        unique_titles = self._deduplicate_titles(all_titles)
        ranked_titles = sorted(
            unique_titles,
            key=lambda x: (x.specificity_score + x.engagement_score) / 2,
            reverse=True,
        )

        # Limit to requested number
        final_titles = ranked_titles[:max_titles]

        self.logger.info(
            f"Generated {len(final_titles)} unique, high-quality blog titles"
        )
        return final_titles

    async def _generate_titles_for_insight(
        self, semaphore: asyncio.Semaphore, insight: SemanticInsight
    ) -> list[BlogTitleCandidate]:
        """Generate titles for a single semantic insight"""

        async with semaphore:
            try:
                # Determine the best title patterns for this insight
                suitable_patterns = self._identify_suitable_patterns(insight)

                # Generate AI-powered titles
                try:
                    ai_titles = await self._generate_ai_titles(
                        insight, suitable_patterns
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error in AI title generation for {insight.article_id}: {str(e)}"
                    )
                    ai_titles = []

                # Generate template-based fallback titles
                try:
                    template_titles = self._generate_template_titles(
                        insight, suitable_patterns
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error in template title generation for {insight.article_id}: {str(e)}"
                    )
                    template_titles = []

                # Combine and analyze all titles
                all_candidates = ai_titles + template_titles

                # Analyze and score each title
                scored_titles = []
                for candidate in all_candidates:
                    try:
                        self._analyze_title_metrics(candidate, insight)
                        if (
                            candidate.specificity_score > 0.3
                        ):  # Filter low-quality titles
                            scored_titles.append(candidate)
                    except Exception as e:
                        self.logger.error(
                            f"Error analyzing title metrics for {candidate.title}: {str(e)}"
                        )
                        continue

                return scored_titles[:5]  # Return top 5 titles per insight

            except Exception as e:
                self.logger.error(
                    f"Error generating titles for insight {insight.article_id}: {str(e)}"
                )
                return []

    def _identify_suitable_patterns(self, insight: SemanticInsight) -> list[str]:
        """Identify which title patterns are suitable for this insight"""

        patterns = []

        # Performance pattern if metrics available
        if insight.performance_metrics or any(
            "improved" in s.lower() or "reduced" in s.lower()
            for s in insight.key_insights
        ):
            patterns.append("performance")

        # Comparison pattern if multiple technologies mentioned
        all_technologies = set()
        for tc in insight.technical_concepts:
            if tc.technologies_used:
                all_technologies.update(tc.technologies_used)
        if len(all_technologies) > 1:
            patterns.append("comparison")

        # Implementation pattern if solutions described
        if insight.solutions_implemented or any(
            tc.implementation_approach for tc in insight.technical_concepts
        ):
            patterns.append("implementation")

        # Problem-solution pattern if problems and solutions identified
        if insight.problems_solved and insight.solutions_implemented:
            patterns.append("problem-solution")

        # How-to pattern if content angle suggests tutorial
        if insight.content_angle in ["tutorial", "guide"]:
            patterns.append("how-to")

        # Default to implementation if no specific patterns identified
        if not patterns:
            patterns.append("implementation")

        return patterns

    async def _generate_ai_titles(
        self, insight: SemanticInsight, patterns: list[str]
    ) -> list[BlogTitleCandidate]:
        """Generate titles using AI based on semantic insights"""

        try:
            # Prepare insight data for AI
            insight_summary = self._prepare_insight_for_ai(insight, patterns)

            # Create AI prompt for title generation
            prompt = self._build_title_generation_prompt(insight_summary, patterns)

            # Query AI for title generation
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_title_generation_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )

            # Parse AI response
            ai_response = json.loads(response.choices[0].message.content)

            # Convert to BlogTitleCandidate objects
            candidates = []
            for title_data in ai_response.get("titles", []):
                candidate = BlogTitleCandidate(
                    title=title_data.get("title", ""),
                    pattern_type=title_data.get("pattern_type", "general"),
                    source_insights=[insight.article_id],
                    generated_by="ai_generation",
                    technical_depth=insight.target_audience,
                    key_technologies=title_data.get("technologies", []),
                    metrics_mentioned=title_data.get("metrics", []),
                    companies_mentioned=title_data.get("companies", []),
                    confidence=float(title_data.get("confidence", 0.5)),
                    generation_reasoning=title_data.get("reasoning", ""),
                )
                candidates.append(candidate)

            return candidates

        except Exception as e:
            self.logger.error(f"AI title generation failed: {str(e)}")
            return []

    def _prepare_insight_for_ai(
        self, insight: SemanticInsight, patterns: list[str]
    ) -> dict[str, Any]:
        """Prepare semantic insight data for AI title generation"""

        # Extract key data points
        technologies = list(
            {tech for tc in insight.technical_concepts for tech in tc.technologies_used}
        )

        companies = list(
            {
                tc.concept
                for tc in insight.technical_concepts
                if any(
                    comp in tc.concept
                    for comp in ["Netflix", "Google", "Amazon", "Facebook", "Microsoft"]
                )
            }
        )

        return {
            "key_insights": insight.key_insights[:5],
            "technical_concepts": [
                {
                    "concept": tc.concept,
                    "problem_solved": tc.problem_solved,
                    "implementation": tc.implementation_approach,
                    "technologies": tc.technologies_used,
                    "business_impact": tc.business_impact,
                }
                for tc in insight.technical_concepts[:3]
            ],
            "performance_metrics": insight.performance_metrics[:3],
            "problems_solved": insight.problems_solved[:3],
            "solutions_implemented": insight.solutions_implemented[:3],
            "technologies": technologies[:5],
            "companies": companies[:3],
            "target_audience": insight.target_audience,
            "content_angle": insight.content_angle,
            "suitable_patterns": patterns,
        }

    def _get_title_generation_system_prompt(self) -> str:
        """Get system prompt for AI title generation"""

        return """You are an expert blog title generator specializing in creating specific, actionable titles for technical content.

Generate blog titles that:
1. Include specific metrics, company names, technologies, or version numbers
2. Follow proven engagement patterns
3. Are actionable and solve specific problems
4. Appeal to the target technical audience
5. Are between 8-15 words long

Avoid generic titles. Focus on specific, concrete details that make readers want to click.

Respond ONLY with valid JSON in this format:
{
  "titles": [
    {
      "title": "How Stripe reduced payment latency by 40% with GraphQL federation",
      "pattern_type": "performance",
      "technologies": ["GraphQL", "Federation"],
      "metrics": ["40% latency reduction"],
      "companies": ["Stripe"],
      "confidence": 0.85,
      "reasoning": "Specific metric, company, and technology mentioned"
    }
  ]
}

Generate 3-5 high-quality, specific titles per request."""

    def _build_title_generation_prompt(
        self, insight_data: dict[str, Any], patterns: list[str]
    ) -> str:
        """Build the prompt for AI title generation"""

        prompt = f"""
Generate specific, actionable blog titles based on this technical insight:

**Key Insights:**
{chr(10).join(f"- {insight}" for insight in insight_data["key_insights"])}

**Technical Concepts:**
{chr(10).join(f"- {tc['concept']}: {tc['problem_solved']}" for tc in insight_data["technical_concepts"])}

**Performance Metrics:**
{chr(10).join(f"- {metric}" for metric in insight_data["performance_metrics"])}

**Technologies:** {", ".join(insight_data["technologies"])}
**Companies:** {", ".join(insight_data["companies"])}
**Target Audience:** {insight_data["target_audience"]}
**Content Angle:** {insight_data["content_angle"]}

**Suitable Title Patterns:** {", ".join(patterns)}

Focus on creating titles that include:
- Specific metrics or improvements
- Company names when available
- Concrete technologies or versions
- Clear value propositions
- Actionable outcomes

Examples of good specific titles:
- "How Netflix reduced React bundle size by 35% with selective hydration"
- "Why Stripe migrated from REST to GraphQL for payment APIs"
- "5 ways GitHub improved CI/CD performance with Actions optimization"

Generate titles following the specified patterns with maximum specificity.
"""

        return prompt.strip()

    def _generate_template_titles(
        self, insight: SemanticInsight, patterns: list[str]
    ) -> list[BlogTitleCandidate]:
        """Generate titles using template patterns as fallback"""

        candidates = []

        # Extract data for template substitution
        technologies = list(
            {tech for tc in insight.technical_concepts for tech in tc.technologies_used}
        )[:3]

        companies = list(
            {
                tc.concept
                for tc in insight.technical_concepts
                if any(
                    comp in tc.concept
                    for comp in ["Netflix", "Google", "Amazon", "Facebook", "Microsoft"]
                )
            }
        )[:2]

        metrics = insight.performance_metrics[:2]
        problems = insight.problems_solved[:2]

        # Generate titles for each suitable pattern
        for pattern in patterns:
            templates = self.title_templates.get(pattern, [])

            for template in templates[:2]:  # Use top 2 templates per pattern
                try:
                    # Substitute template variables
                    title = self._substitute_template_variables(
                        template, technologies, companies, metrics, problems
                    )

                    if title and len(title.split()) >= 6:  # Ensure meaningful titles
                        candidate = BlogTitleCandidate(
                            title=title,
                            pattern_type=pattern,
                            source_insights=[insight.article_id],
                            generated_by="template_generation",
                            technical_depth=insight.target_audience,
                            key_technologies=technologies[:2],
                            metrics_mentioned=metrics,
                            companies_mentioned=companies,
                            confidence=0.6,  # Template-based confidence
                        )
                        candidates.append(candidate)

                except Exception as e:
                    self.logger.debug(f"Template substitution failed: {str(e)}")
                    continue

        return candidates

    def _substitute_template_variables(
        self,
        template: str,
        technologies: list[str],
        companies: list[str],
        metrics: list[str],
        problems: list[str],
    ) -> str | None:
        """Substitute variables in title templates"""

        substitutions = {
            "technology": technologies[0] if technologies else "modern technology",
            "tech1": technologies[0] if technologies else "React",
            "tech2": technologies[1] if len(technologies) > 1 else "Vue",
            "company": companies[0] if companies else "leading companies",
            "metric": metrics[0] if metrics else "performance",
            "amount": "30%",  # Default improvement amount
            "action": "improved",
            "use_case": "web applications",
            "problem": problems[0] if problems else "performance issues",
            "process": "development workflow",
            "solution": "scalable system",
            "outcome": "better performance",
            "number": "5",
        }

        try:
            return template.format(**substitutions)
        except KeyError:
            return None

    def _analyze_title_metrics(
        self, candidate: BlogTitleCandidate, insight: SemanticInsight
    ):
        """Analyze and score title metrics"""

        title_lower = candidate.title.lower()

        # Calculate specificity score
        specificity_indicators = [
            r"\b\d+%",
            r"\b\d+x\b",
            r"\bv?\d+\.\d+",  # Numbers, percentages, versions
            r"\b(netflix|google|amazon|facebook|microsoft|stripe|uber|airbnb)\b",  # Companies
            r"\b(react|vue|angular|python|javascript|node\.js|docker|kubernetes)\b",  # Technologies
        ]

        specificity_score = 0.0
        for pattern in specificity_indicators:
            matches = len(re.findall(pattern, title_lower))
            specificity_score += min(matches * 0.25, 0.5)

        candidate.specificity_score = min(specificity_score, 1.0)

        # Calculate engagement score
        engagement_indicators = [
            r"\bhow\b",
            r"\bwhy\b",
            r"\bways?\b",
            r"\bguide\b",  # Action words
            r"\bimproved?\b",
            r"\breduced?\b",
            r"\boptimized?\b",  # Results
            r"\bbetter\b",
            r"\bfaster\b",
            r"\beasier\b",  # Benefits
        ]

        engagement_score = 0.3  # Base score
        for pattern in engagement_indicators:
            if re.search(pattern, title_lower):
                engagement_score += 0.2

        candidate.engagement_score = min(engagement_score, 1.0)

        # Set technical depth based on insight
        candidate.technical_depth = insight.target_audience

    def _deduplicate_titles(
        self, titles: list[BlogTitleCandidate]
    ) -> list[BlogTitleCandidate]:
        """Remove duplicate and very similar titles"""

        unique_titles = []
        seen_titles = set()

        for title in titles:
            # Normalize title for comparison
            normalized = re.sub(r"\W+", " ", title.title.lower()).strip()

            # Check for exact duplicates
            if normalized in seen_titles:
                continue

            # Check for very similar titles (>80% word overlap)
            is_similar = False
            title_words = set(normalized.split())

            for existing in unique_titles:
                existing_normalized = re.sub(
                    r"\W+", " ", existing.title.lower()
                ).strip()
                existing_words = set(existing_normalized.split())

                if len(title_words) > 0 and len(existing_words) > 0:
                    overlap = len(title_words.intersection(existing_words))
                    similarity = overlap / max(len(title_words), len(existing_words))

                    if similarity > 0.8:
                        is_similar = True
                        # Keep the title with higher combined score
                        if (title.specificity_score + title.engagement_score) > (
                            existing.specificity_score + existing.engagement_score
                        ):
                            unique_titles.remove(existing)
                            break
                        else:
                            is_similar = True
                            break

            if not is_similar:
                seen_titles.add(normalized)
                unique_titles.append(title)

        return unique_titles

    async def close(self):
        """Close the async OpenAI client"""
        if hasattr(self.client, "close"):
            await self.client.close()
