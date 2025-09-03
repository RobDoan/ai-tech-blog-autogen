"""
SEOAgent for the Multi-Agent Blog Writer system.

This agent specializes in search engine optimization, keyword analysis,
and content optimization for improved organic search visibility.
"""

import re
from typing import Any

from .base_agent import BaseAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogContent,
    ContentOutline,
    KeywordAnalysis,
    MessageType,
    SEOOptimizedContent,
    SEOServiceError,
)


class SEOAgent(BaseAgent):
    """
    Agent responsible for SEO optimization and keyword analysis.

    Specializes in:
    - Trending keyword research and analysis
    - Content optimization for search engines
    - Meta description and title optimization
    - SEO score estimation and improvement
    - Keyword integration and density optimization
    - Search intent analysis and alignment
    """

    def __init__(self, config: AgentConfig):
        """Initialize the SEO Agent."""
        super().__init__("SEO", config)

    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        return """
You are an expert SEO Specialist focusing on content optimization for organic search visibility. Your role is to analyze content from an SEO perspective and provide actionable recommendations to improve search engine rankings and organic traffic.

Your expertise includes:
1. Keyword research and competitive analysis
2. Search intent analysis and content alignment
3. On-page optimization best practices
4. Content structure for SEO (headers, meta descriptions, etc.)
5. Technical SEO considerations for content
6. Trending topic identification and keyword opportunities
7. Content optimization without sacrificing readability

SEO principles you follow:
- Keywords should be integrated naturally, never forced
- Content quality and user value always come first
- Title tags should be compelling and include primary keywords
- Meta descriptions should be engaging and actionable
- Header structure should be logical and keyword-optimized
- Content should match search intent and user expectations
- Focus on long-tail keywords and semantic relevance

When optimizing content, you must:
- Identify primary and secondary keyword opportunities
- Suggest natural keyword integration throughout content
- Create compelling, SEO-optimized titles and meta descriptions
- Analyze content structure for SEO best practices
- Provide actionable recommendations for improvement
- Maintain content readability and user experience
- Consider trending keywords and search volume patterns

Your recommendations should:
- Be specific and actionable
- Balance SEO optimization with content quality
- Consider the target audience and search intent
- Focus on sustainable, white-hat SEO practices
- Prioritize user value over keyword density

Always provide structured analysis in JSON format with keyword recommendations, optimization suggestions, and estimated SEO impact.
"""

    async def analyze_keywords(
        self,
        topic: str,
        content: BlogContent | None = None,
        outline: ContentOutline | None = None,
    ) -> KeywordAnalysis:
        """
        Analyze keywords for a topic and existing content.

        Args:
            topic: Main topic for keyword research
            content: Existing content to analyze (optional)
            outline: Content outline for context (optional)

        Returns:
            KeywordAnalysis with keyword recommendations

        Raises:
            SEOServiceError: If keyword analysis fails
        """
        try:
            # Build keyword analysis prompt
            prompt = self._build_keyword_analysis_prompt(topic, content, outline)

            # Query the agent
            response = await self.query_agent(
                prompt, message_type=MessageType.SEO_ANALYSIS
            )

            # Parse the keyword analysis response
            analysis = await self._parse_keyword_analysis_response(response)

            self.logger.info(
                f"Keyword analysis completed: {len(analysis.primary_keywords)} primary, "
                f"{len(analysis.secondary_keywords)} secondary keywords"
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze keywords: {e}")
            raise SEOServiceError(f"Keyword analysis failed: {e}") from e

    async def optimize_content(
        self,
        content: BlogContent,
        keyword_analysis: KeywordAnalysis,
        target_seo_score: float = 80.0,
    ) -> SEOOptimizedContent:
        """
        Optimize content for SEO based on keyword analysis.

        Args:
            content: Content to optimize
            keyword_analysis: Keywords to target
            target_seo_score: Target SEO score (0-100)

        Returns:
            SEOOptimizedContent with optimizations applied

        Raises:
            SEOServiceError: If optimization fails
        """
        try:
            # Build optimization prompt
            prompt = self._build_optimization_prompt(
                content, keyword_analysis, target_seo_score
            )

            # Query the agent
            response = await self.query_agent(
                prompt, message_type=MessageType.SEO_ANALYSIS
            )

            # Parse the optimization response
            optimized_content = await self._parse_optimization_response(
                response, content
            )

            self.logger.info(
                f"Content optimization completed: estimated SEO score {optimized_content.seo_score}"
            )
            return optimized_content

        except Exception as e:
            self.logger.error(f"Failed to optimize content: {e}")
            raise SEOServiceError(f"Content optimization failed: {e}") from e

    def _build_keyword_analysis_prompt(
        self,
        topic: str,
        content: BlogContent | None = None,
        outline: ContentOutline | None = None,
    ) -> str:
        """Build prompt for keyword analysis."""
        context_sections = []

        if outline:
            sections_text = ", ".join([section.heading for section in outline.sections])
            context_sections.append(f"Planned Sections: {sections_text}")
            if outline.target_keywords:
                context_sections.append(
                    f"Initial Keywords: {', '.join(outline.target_keywords)}"
                )

        if content:
            context_sections.append(
                f"Content Length: {content.metadata.word_count} words"
            )
            context_sections.append(f"Existing Sections: {', '.join(content.sections)}")

        context_text = (
            "\\n".join(context_sections)
            if context_sections
            else "No additional context provided."
        )

        return f"""
Conduct comprehensive keyword research and analysis for the following topic:

MAIN TOPIC: {topic}

CONTEXT:
{context_text}

Please analyze this topic and provide keyword recommendations that will:
1. Attract the right audience searching for this information
2. Have reasonable search volume and competition balance
3. Include trending keywords in this space
4. Consider semantic and related keywords
5. Focus on long-tail opportunities for better targeting

Your analysis should consider:
- Primary keywords (3-5): High-volume, competitive terms directly related to the topic
- Secondary keywords (5-8): Supporting terms that enhance content comprehensiveness
- Long-tail keywords (5-10): Specific phrases with lower competition
- Trending keywords: Currently popular terms in this domain
- Search intent alignment: What users are actually looking for

Provide your analysis in this exact JSON format:
{{
    "primary_keywords": [
        "main keyword 1",
        "main keyword 2",
        "main keyword 3"
    ],
    "secondary_keywords": [
        "supporting keyword 1",
        "supporting keyword 2",
        "supporting keyword 3"
    ],
    "trending_keywords": [
        "trending term 1",
        "trending term 2"
    ],
    "long_tail_keywords": [
        "specific long tail phrase 1",
        "specific long tail phrase 2"
    ],
    "competition_level": {{
        "keyword 1": "low/medium/high",
        "keyword 2": "low/medium/high"
    }},
    "search_volume": {{
        "keyword 1": "estimated monthly searches",
        "keyword 2": "estimated monthly searches"
    }},
    "search_intent": {{
        "informational": ["keywords for learning"],
        "transactional": ["keywords for buying/doing"],
        "navigational": ["keywords for finding specific things"]
    }},
    "content_opportunities": [
        "Specific content angles that could rank well",
        "Questions people are asking about this topic"
    ]
}}

Focus on keywords that will genuinely help the target audience find valuable content about {topic}.
"""

    def _build_optimization_prompt(
        self,
        content: BlogContent,
        keyword_analysis: KeywordAnalysis,
        target_seo_score: float,
    ) -> str:
        """Build prompt for content optimization."""
        primary_keywords = ", ".join(keyword_analysis.primary_keywords)
        secondary_keywords = ", ".join(keyword_analysis.secondary_keywords)
        trending_keywords = ", ".join(keyword_analysis.trending_keywords)

        return f"""
Optimize the following blog content for SEO while maintaining readability and user value:

CURRENT CONTENT:
{content.content}

KEYWORD TARGETS:
Primary Keywords: {primary_keywords}
Secondary Keywords: {secondary_keywords}
Trending Keywords: {trending_keywords}

CURRENT CONTENT METRICS:
Word Count: {content.metadata.word_count}
Sections: {len(content.sections)}
Code Blocks: {len(content.code_blocks)}

TARGET SEO SCORE: {target_seo_score}/100

OPTIMIZATION AREAS TO ADDRESS:
1. Title optimization (include primary keyword naturally)
2. Meta description creation (compelling, 150-160 characters, includes primary keyword)
3. Header optimization (H1, H2, H3 with strategic keyword placement)
4. Content optimization (natural keyword integration throughout)
5. Internal linking opportunities (suggest relevant anchor text)
6. Image alt text suggestions (if applicable)

OPTIMIZATION PRINCIPLES:
- Keywords must be integrated naturally, never forced
- Maintain content readability and flow
- Preserve the original content quality and value
- Focus on user experience while optimizing for search
- Use semantic variations and related terms
- Ensure content matches search intent

Provide your optimization in this exact JSON format:
{{
    "optimized_title": "SEO-optimized title with primary keyword",
    "meta_description": "Compelling 150-160 character description with primary keyword and call-to-action",
    "optimized_content": "Complete optimized blog content with keywords naturally integrated",
    "keywords_used": [
        "List of keywords successfully integrated into content"
    ],
    "seo_score": 85.0,
    "optimization_summary": [
        "Key changes made for SEO improvement",
        "Strategic keyword placements",
        "Structure improvements"
    ],
    "header_optimizations": {{
        "H1": "Main title optimization",
        "H2": ["List of H2 headers with keywords"],
        "H3": ["List of H3 headers with supporting keywords"]
    }},
    "internal_link_suggestions": [
        "Suggested anchor text for internal links",
        "Related topics to link to"
    ],
    "technical_seo_notes": [
        "Additional technical SEO recommendations",
        "Schema markup suggestions if applicable"
    ]
}}

Ensure the optimized content reads naturally while being well-optimized for the target keywords.
"""

    async def _parse_keyword_analysis_response(
        self, response: AgentMessage
    ) -> KeywordAnalysis:
        """Parse keyword analysis response into structured data."""
        try:
            analysis_data = self.parse_json_response(response.content)
            if not analysis_data:
                raise SEOServiceError(
                    "Failed to parse keyword analysis response as JSON"
                )

            # Extract keywords with fallbacks
            primary_keywords = analysis_data.get("primary_keywords", [])
            secondary_keywords = analysis_data.get("secondary_keywords", [])
            trending_keywords = analysis_data.get("trending_keywords", [])
            competition_level = analysis_data.get("competition_level", {})
            search_volume = analysis_data.get("search_volume", {})

            # Convert search volume strings to integers where possible
            processed_search_volume = {}
            for keyword, volume in search_volume.items():
                try:
                    # Extract numbers from strings like "1000-2000" or "1.2K"
                    if isinstance(volume, str):
                        volume_str = volume.lower()
                        if "k" in volume_str:
                            num = float(volume_str.replace("k", "").replace(",", ""))
                            processed_search_volume[keyword] = int(num * 1000)
                        elif "-" in volume_str:
                            # Take average of range
                            parts = volume_str.split("-")
                            avg = (int(parts[0]) + int(parts[1])) // 2
                            processed_search_volume[keyword] = avg
                        else:
                            processed_search_volume[keyword] = int(
                                volume_str.replace(",", "")
                            )
                    else:
                        processed_search_volume[keyword] = int(volume)
                except (ValueError, TypeError):
                    processed_search_volume[keyword] = 0

            analysis = KeywordAnalysis(
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                trending_keywords=trending_keywords,
                competition_level=competition_level,
                search_volume=processed_search_volume,
            )

            # Validate analysis quality
            self._validate_keyword_analysis(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to parse keyword analysis: {e}")
            raise SEOServiceError(f"Keyword analysis parsing failed: {e}") from e

    async def _parse_optimization_response(
        self, response: AgentMessage, original_content: BlogContent
    ) -> SEOOptimizedContent:
        """Parse optimization response into structured data."""
        try:
            optimization_data = self.parse_json_response(response.content)
            if not optimization_data:
                raise SEOServiceError("Failed to parse optimization response as JSON")

            # Extract required fields with fallbacks
            optimized_title = optimization_data.get(
                "optimized_title", original_content.title
            )
            meta_description = optimization_data.get("meta_description", "")
            optimized_content = optimization_data.get(
                "optimized_content", original_content.content
            )
            keywords_used = optimization_data.get("keywords_used", [])
            seo_score = float(optimization_data.get("seo_score", 50.0))

            optimization = SEOOptimizedContent(
                optimized_title=optimized_title,
                meta_description=meta_description,
                optimized_content=optimized_content,
                keywords_used=keywords_used,
                seo_score=seo_score,
            )

            # Validate optimization quality
            self._validate_optimization(optimization, original_content)

            return optimization

        except Exception as e:
            self.logger.error(f"Failed to parse optimization response: {e}")
            raise SEOServiceError(f"Optimization parsing failed: {e}") from e

    def _validate_keyword_analysis(self, analysis: KeywordAnalysis) -> None:
        """Validate keyword analysis quality."""
        if len(analysis.primary_keywords) < 1:
            raise SEOServiceError("Must have at least one primary keyword")

        if len(analysis.primary_keywords) > 5:
            self.logger.warning("Too many primary keywords may dilute focus")

        # Check for keyword uniqueness
        all_keywords = (
            analysis.primary_keywords
            + analysis.secondary_keywords
            + analysis.trending_keywords
        )

        if len(all_keywords) != len(set(all_keywords)):
            self.logger.warning("Duplicate keywords found in analysis")

        # Validate keywords are not empty strings
        empty_keywords = [kw for kw in all_keywords if not kw.strip()]
        if empty_keywords:
            raise SEOServiceError("Found empty keywords in analysis")

        self.logger.info("Keyword analysis validation passed")

    def _validate_optimization(
        self, optimization: SEOOptimizedContent, original_content: BlogContent
    ) -> None:
        """Validate optimization quality."""
        # Check title length (recommended 30-60 characters)
        title_length = len(optimization.optimized_title)
        if title_length < 20:
            self.logger.warning(
                f"Optimized title may be too short: {title_length} characters"
            )
        elif title_length > 80:
            self.logger.warning(
                f"Optimized title may be too long: {title_length} characters"
            )

        # Check meta description length (recommended 150-160 characters)
        meta_length = len(optimization.meta_description)
        if meta_length < 120:
            self.logger.warning(
                f"Meta description may be too short: {meta_length} characters"
            )
        elif meta_length > 170:
            self.logger.warning(
                f"Meta description may be too long: {meta_length} characters"
            )

        # Ensure optimized content isn't dramatically shorter
        original_length = len(original_content.content)
        optimized_length = len(optimization.optimized_content)

        if optimized_length < original_length * 0.8:
            self.logger.warning("Optimized content significantly shorter than original")

        # Check SEO score validity
        if not 0 <= optimization.seo_score <= 100:
            raise SEOServiceError(f"Invalid SEO score: {optimization.seo_score}")

        # Validate keywords were actually used
        if len(optimization.keywords_used) < 1:
            self.logger.warning("No keywords reported as used in optimization")

        self.logger.info("Optimization validation passed")

    async def estimate_seo_impact(
        self,
        original_content: BlogContent,
        optimized_content: SEOOptimizedContent,
        keyword_analysis: KeywordAnalysis,
    ) -> dict[str, Any]:
        """
        Estimate the SEO impact of optimizations.

        Args:
            original_content: Original content before optimization
            optimized_content: Content after SEO optimization
            keyword_analysis: Keyword analysis used for optimization

        Returns:
            SEO impact analysis with metrics and projections
        """
        try:
            # Build impact analysis prompt
            prompt = self._build_impact_analysis_prompt(
                original_content, optimized_content, keyword_analysis
            )

            # Query the agent
            response = await self.query_agent(
                prompt, message_type=MessageType.SEO_ANALYSIS
            )

            # Parse impact analysis
            impact_data = self.parse_json_response(response.content)
            if not impact_data:
                impact_data = self._create_basic_impact_analysis(
                    original_content, optimized_content
                )

            return impact_data

        except Exception as e:
            self.logger.error(f"Failed to estimate SEO impact: {e}")
            return self._create_basic_impact_analysis(
                original_content, optimized_content
            )

    def _build_impact_analysis_prompt(
        self,
        original_content: BlogContent,
        optimized_content: SEOOptimizedContent,
        keyword_analysis: KeywordAnalysis,
    ) -> str:
        """Build prompt for SEO impact analysis."""
        return f"""
Analyze the potential SEO impact of the optimizations made to this content:

ORIGINAL CONTENT METRICS:
- Title: {original_content.title}
- Word Count: {original_content.metadata.word_count}
- Sections: {len(original_content.sections)}
- Existing SEO Elements: Basic structure

OPTIMIZED CONTENT METRICS:
- Title: {optimized_content.optimized_title}
- Meta Description: {optimized_content.meta_description}
- Keywords Targeted: {len(optimized_content.keywords_used)}
- SEO Score: {optimized_content.seo_score}/100

KEYWORD ANALYSIS:
- Primary Keywords: {len(keyword_analysis.primary_keywords)}
- Secondary Keywords: {len(keyword_analysis.secondary_keywords)}
- Trending Keywords: {len(keyword_analysis.trending_keywords)}

Please estimate the SEO impact in this JSON format:
{{
    "overall_improvement": 75,
    "impact_areas": {{
        "title_optimization": 85,
        "keyword_integration": 80,
        "content_structure": 75,
        "meta_description": 90,
        "search_visibility": 70
    }},
    "projected_benefits": [
        "Improved ranking potential for target keywords",
        "Better click-through rates from search results",
        "Enhanced content discoverability"
    ],
    "ranking_potential": {{
        "current_estimate": "Page 2-3",
        "optimized_estimate": "Page 1-2",
        "confidence_level": "medium"
    }},
    "traffic_projection": {{
        "potential_increase": "25-40%",
        "timeframe": "3-6 months",
        "factors": ["Keyword competitiveness", "Content quality", "Site authority"]
    }},
    "competitive_advantage": [
        "Strengths gained over competitors",
        "Unique positioning achieved"
    ]
}}
"""

    def _create_basic_impact_analysis(
        self, original_content: BlogContent, optimized_content: SEOOptimizedContent
    ) -> dict[str, Any]:
        """Create basic SEO impact analysis as fallback."""
        # Simple impact scoring based on obvious improvements
        title_improved = len(optimized_content.optimized_title) != len(
            original_content.title
        )
        meta_added = len(optimized_content.meta_description) > 50
        keywords_added = len(optimized_content.keywords_used) > 0

        improvements = []
        if title_improved:
            improvements.append("Title optimized for search")
        if meta_added:
            improvements.append("Meta description added")
        if keywords_added:
            improvements.append("Keywords strategically integrated")

        overall_improvement = optimized_content.seo_score

        return {
            "overall_improvement": overall_improvement,
            "impact_areas": {
                "title_optimization": 80 if title_improved else 60,
                "keyword_integration": 75 if keywords_added else 50,
                "content_structure": 70,
                "meta_description": 85 if meta_added else 40,
                "search_visibility": overall_improvement,
            },
            "projected_benefits": improvements
            if improvements
            else ["Basic SEO improvements applied"],
            "ranking_potential": {
                "current_estimate": "Unoptimized",
                "optimized_estimate": "Improved positioning",
                "confidence_level": "medium",
            },
            "traffic_projection": {
                "potential_increase": "15-30%",
                "timeframe": "2-4 months",
                "factors": ["Content quality", "Keyword targeting", "Site authority"],
            },
            "competitive_advantage": [
                "Better search visibility",
                "Improved content discoverability",
            ],
        }

    async def audit_existing_seo(self, content: BlogContent) -> dict[str, Any]:
        """
        Audit existing content for SEO issues and opportunities.

        Args:
            content: Content to audit

        Returns:
            SEO audit results with issues and recommendations
        """
        audit_results = {
            "seo_score": 0.0,
            "issues": [],
            "opportunities": [],
            "technical_issues": [],
            "content_issues": [],
        }

        score_factors = []

        # Title analysis
        title = content.title
        title_length = len(title)

        if 30 <= title_length <= 60:
            score_factors.append(15)
        elif 20 <= title_length <= 80:
            score_factors.append(10)
            audit_results["opportunities"].append(
                "Optimize title length (aim for 30-60 characters)"
            )
        else:
            audit_results["issues"].append(
                f"Title length ({title_length} chars) not optimal"
            )

        # Header structure analysis
        content_text = content.content
        h1_count = len(re.findall(r"^# ", content_text, re.MULTILINE))
        h2_count = len(re.findall(r"^## ", content_text, re.MULTILINE))

        if h1_count == 1:
            score_factors.append(10)
        else:
            audit_results["issues"].append(
                f"Should have exactly 1 H1 header (found {h1_count})"
            )

        if h2_count >= 3:
            score_factors.append(15)
        else:
            audit_results["opportunities"].append(
                "Add more section headers (H2) for better structure"
            )

        # Content length analysis
        word_count = content.metadata.word_count
        if word_count >= 1000:
            score_factors.append(15)
        elif word_count >= 500:
            score_factors.append(10)
        else:
            audit_results["issues"].append(
                "Content may be too short for good SEO performance"
            )

        # Meta description check
        if (
            hasattr(content.metadata, "meta_description")
            and content.metadata.meta_description
        ):
            meta_length = len(content.metadata.meta_description)
            if 120 <= meta_length <= 160:
                score_factors.append(15)
            else:
                audit_results["opportunities"].append(
                    "Optimize meta description length"
                )
        else:
            audit_results["issues"].append("Missing meta description")

        # Internal structure
        if len(content.sections) >= 4:
            score_factors.append(10)
        else:
            audit_results["opportunities"].append(
                "Add more content sections for comprehensive coverage"
            )

        # Code examples (for technical content)
        if len(content.code_blocks) > 0:
            score_factors.append(10)
            audit_results["opportunities"].append(
                "Code examples enhance technical content value"
            )

        # Lists and formatting
        list_count = len(re.findall(r"^[-*+] ", content_text, re.MULTILINE))
        if list_count >= 2:
            score_factors.append(10)
        else:
            audit_results["opportunities"].append(
                "Add more lists for better readability"
            )

        audit_results["seo_score"] = sum(score_factors)

        return audit_results
