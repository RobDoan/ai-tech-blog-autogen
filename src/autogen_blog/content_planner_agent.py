"""
ContentPlannerAgent for the Multi-Agent Blog Writer system.

This agent specializes in content strategy, audience analysis, and structural planning.
It creates detailed blog outlines with sections, key points, and strategic guidance
for the writing process.
"""

from typing import Any

from .base_agent import BaseAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogInput,
    ContentOutline,
    ContentQualityError,
    MessageType,
    Section,
)


class ContentPlannerAgent(BaseAgent):
    """
    Agent responsible for creating structured blog outlines and content strategy.

    Specializes in:
    - Analyzing topics and audience requirements
    - Creating comprehensive content outlines
    - Identifying key points and structure
    - Planning for code examples and technical content
    - Refining outlines based on feedback
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Content Planner Agent."""
        super().__init__("ContentPlanner", config)

    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        return """
You are an expert Content Strategy Planner specializing in technical blog content. Your role is to create comprehensive, well-structured blog outlines that serve as the foundation for high-quality technical articles.

Your expertise includes:
1. Audience analysis and content positioning
2. Information architecture and content structure
3. Technical depth assessment and appropriate complexity levels
4. SEO-conscious content planning
5. Identifying opportunities for code examples and practical demonstrations

When creating a blog outline, you must consider:
- Target audience knowledge level and needs
- Logical flow from introduction to conclusion
- Appropriate depth and breadth of coverage
- Balance between theory and practical application
- SEO keywords and search intent alignment
- Content length and reading experience optimization

Always provide structured, actionable outlines in JSON format with:
- Clear, compelling title
- Engaging introduction summary
- Detailed sections with key points
- Conclusion that reinforces value
- Estimated word counts and keywords
- Technical complexity assessment

Focus on creating outlines that will result in valuable, comprehensive, and engaging blog posts that serve the reader's needs while establishing authority in the subject matter.
"""

    async def create_outline(self, blog_input: BlogInput) -> ContentOutline:
        """
        Create a comprehensive blog outline based on input requirements.

        Args:
            blog_input: Input data including topic, description, and requirements

        Returns:
            ContentOutline with structured plan for the blog post

        Raises:
            ContentQualityError: If outline generation fails or produces low-quality results
        """
        try:
            # Build prompt for outline creation
            prompt = self._build_outline_prompt(blog_input)

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.OUTLINE)

            # Validate and parse the response
            outline = await self._parse_outline_response(response, blog_input)

            self.logger.info(
                f"Created outline with {len(outline.sections)} sections for: {blog_input.topic}"
            )
            return outline

        except Exception as e:
            self.logger.error(f"Failed to create outline: {e}")
            raise ContentQualityError(f"Outline creation failed: {e}")

    async def refine_outline(
        self, current_outline: ContentOutline, feedback: str, blog_input: BlogInput
    ) -> ContentOutline:
        """
        Refine an existing outline based on feedback.

        Args:
            current_outline: The existing outline to refine
            feedback: Feedback for improvement
            blog_input: Original input requirements

        Returns:
            Refined ContentOutline

        Raises:
            ContentQualityError: If refinement fails
        """
        try:
            # Build refinement prompt
            prompt = self._build_refinement_prompt(
                current_outline, feedback, blog_input
            )

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.OUTLINE)

            # Parse the refined outline
            refined_outline = await self._parse_outline_response(response, blog_input)

            self.logger.info(
                f"Refined outline based on feedback: {len(refined_outline.sections)} sections"
            )
            return refined_outline

        except Exception as e:
            self.logger.error(f"Failed to refine outline: {e}")
            raise ContentQualityError(f"Outline refinement failed: {e}")

    def _build_outline_prompt(self, blog_input: BlogInput) -> str:
        """Build the prompt for creating a blog outline."""
        context_section = ""
        if blog_input.description:
            context_section = f"\\nAdditional Context: {blog_input.description}"
        if blog_input.book_reference:
            context_section += f"\\nReference Material: {blog_input.book_reference}"

        return f"""
Create a comprehensive blog outline for the following topic:

Topic: {blog_input.topic}
Target Audience: {blog_input.target_audience.value}
Preferred Length: {blog_input.preferred_length} words{context_section}

Please create a detailed blog outline that includes:

1. A compelling, SEO-friendly title
2. An engaging introduction summary (what readers will learn and why it matters)
3. 3-6 main sections with:
   - Clear, descriptive headings
   - 3-5 key points to cover in each section
   - Indication of whether code examples are needed
   - Estimated word count for each section
4. A conclusion summary that reinforces value and next steps
5. Target keywords for SEO optimization

Consider the target audience level when determining:
- Technical depth and complexity
- Amount of background explanation needed
- Types of examples and analogies to use
- Appropriate prerequisites to assume

Respond with a JSON object in this exact format:
{{
    "title": "Compelling blog post title",
    "introduction": "Summary of what the introduction will cover",
    "sections": [
        {{
            "heading": "Section heading",
            "key_points": [
                "Key point 1",
                "Key point 2",
                "Key point 3"
            ],
            "code_examples_needed": true/false,
            "estimated_words": 300
        }}
    ],
    "conclusion": "Summary of what the conclusion will cover",
    "target_keywords": [
        "keyword1",
        "keyword2",
        "keyword3"
    ]
}}
"""

    def _build_refinement_prompt(
        self, current_outline: ContentOutline, feedback: str, blog_input: BlogInput
    ) -> str:
        """Build the prompt for refining a blog outline based on feedback."""
        return f"""
Please refine the following blog outline based on the feedback provided:

Current Outline:
Title: {current_outline.title}
Introduction: {current_outline.introduction}

Sections:
{self._format_sections_for_prompt(current_outline.sections)}

Conclusion: {current_outline.conclusion}
Target Keywords: {", ".join(current_outline.target_keywords)}

Feedback to Address:
{feedback}

Original Requirements:
- Topic: {blog_input.topic}
- Target Audience: {blog_input.target_audience.value}
- Preferred Length: {blog_input.preferred_length} words
- Additional Context: {blog_input.description or "None"}

Please provide a refined outline that addresses the feedback while maintaining the overall structure and quality. Respond with the same JSON format as before:

{{
    "title": "Refined blog post title",
    "introduction": "Updated introduction summary",
    "sections": [
        {{
            "heading": "Section heading",
            "key_points": [
                "Key point 1",
                "Key point 2",
                "Key point 3"
            ],
            "code_examples_needed": true/false,
            "estimated_words": 300
        }}
    ],
    "conclusion": "Updated conclusion summary",
    "target_keywords": [
        "keyword1",
        "keyword2",
        "keyword3"
    ]
}}
"""

    def _format_sections_for_prompt(self, sections: list[Section]) -> str:
        """Format sections for inclusion in prompts."""
        formatted = []
        for i, section in enumerate(sections, 1):
            points = "\\n    - ".join(section.key_points)
            formatted.append(
                f"{i}. {section.heading}\\n"
                f"   Key Points:\\n    - {points}\\n"
                f"   Code Examples Needed: {section.code_examples_needed}\\n"
                f"   Estimated Words: {section.estimated_words}"
            )
        return "\\n\\n".join(formatted)

    async def _parse_outline_response(
        self, response: AgentMessage, blog_input: BlogInput
    ) -> ContentOutline:
        """
        Parse the agent's response into a ContentOutline object.

        Args:
            response: Response from the agent
            blog_input: Original input for validation

        Returns:
            ContentOutline object

        Raises:
            ContentQualityError: If parsing fails or outline is invalid
        """
        try:
            # Parse JSON response
            outline_data = self.parse_json_response(response.content)
            if not outline_data:
                raise ContentQualityError("Failed to parse outline response as JSON")

            # Validate required fields
            required_fields = ["title", "introduction", "sections", "conclusion"]
            for field in required_fields:
                if field not in outline_data:
                    raise ContentQualityError(f"Missing required field: {field}")

            # Parse sections
            sections = []
            for section_data in outline_data["sections"]:
                section = Section(
                    heading=section_data.get("heading", "Untitled Section"),
                    key_points=section_data.get("key_points", []),
                    code_examples_needed=section_data.get(
                        "code_examples_needed", False
                    ),
                    estimated_words=section_data.get("estimated_words", 200),
                )
                sections.append(section)

            # Calculate total word count
            total_estimated = (
                sum(s.estimated_words for s in sections) + 200
            )  # Add intro/conclusion

            # Create ContentOutline
            outline = ContentOutline(
                title=outline_data["title"],
                introduction=outline_data["introduction"],
                sections=sections,
                conclusion=outline_data["conclusion"],
                estimated_word_count=total_estimated,
                target_keywords=outline_data.get("target_keywords", []),
            )

            # Validate outline quality
            await self._validate_outline_quality(outline, blog_input)

            return outline

        except Exception as e:
            self.logger.error(f"Failed to parse outline response: {e}")
            raise ContentQualityError(f"Outline parsing failed: {e}")

    async def _validate_outline_quality(
        self, outline: ContentOutline, blog_input: BlogInput
    ) -> None:
        """
        Validate that the outline meets quality standards.

        Args:
            outline: The outline to validate
            blog_input: Original requirements for validation

        Raises:
            ContentQualityError: If outline doesn't meet quality standards
        """
        # Check minimum sections
        if len(outline.sections) < 2:
            raise ContentQualityError("Outline must have at least 2 sections")

        if len(outline.sections) > 8:
            raise ContentQualityError("Outline has too many sections (max 8)")

        # Check title quality
        if len(outline.title.split()) < 3:
            raise ContentQualityError("Title too short - needs at least 3 words")

        if len(outline.title) > 100:
            raise ContentQualityError("Title too long - max 100 characters")

        # Check if title relates to topic
        topic_words = set(blog_input.topic.lower().split())
        title_words = set(outline.title.lower().split())
        if not topic_words.intersection(title_words):
            self.logger.warning("Title may not relate closely to topic")

        # Validate sections have content
        for section in outline.sections:
            if not section.heading:
                raise ContentQualityError("All sections must have headings")

            if len(section.key_points) < 1:
                raise ContentQualityError(
                    f"Section '{section.heading}' has no key points"
                )

            if section.estimated_words < 50:
                raise ContentQualityError(
                    f"Section '{section.heading}' estimated word count too low"
                )

        # Check total length alignment
        length_diff = abs(outline.estimated_word_count - blog_input.preferred_length)
        max_allowed_diff = blog_input.preferred_length * 0.5  # 50% variance allowed

        if length_diff > max_allowed_diff:
            self.logger.warning(
                f"Outline length ({outline.estimated_word_count}) differs significantly "
                f"from target ({blog_input.preferred_length})"
            )

        # Validate introduction and conclusion
        if len(outline.introduction.split()) < 10:
            raise ContentQualityError("Introduction summary too short")

        if len(outline.conclusion.split()) < 5:
            raise ContentQualityError("Conclusion summary too short")

        self.logger.info("Outline validation passed")

    async def analyze_outline_completeness(
        self, outline: ContentOutline
    ) -> dict[str, Any]:
        """
        Analyze how complete and well-structured an outline is.

        Args:
            outline: The outline to analyze

        Returns:
            Analysis results with scores and recommendations
        """
        analysis = {
            "completeness_score": 0.0,
            "structure_score": 0.0,
            "seo_readiness": 0.0,
            "recommendations": [],
            "strengths": [],
        }

        # Completeness scoring
        completeness_factors = []

        # Check if sections have sufficient key points
        avg_points_per_section = sum(len(s.key_points) for s in outline.sections) / len(
            outline.sections
        )
        if avg_points_per_section >= 3:
            completeness_factors.append(0.3)
            analysis["strengths"].append("Sections have comprehensive key points")
        else:
            analysis["recommendations"].append(
                "Add more key points to sections (aim for 3-5 per section)"
            )

        # Check introduction and conclusion depth
        intro_words = len(outline.introduction.split())
        conclusion_words = len(outline.conclusion.split())

        if intro_words >= 15 and conclusion_words >= 10:
            completeness_factors.append(0.25)
            analysis["strengths"].append(
                "Introduction and conclusion are well-developed"
            )
        else:
            analysis["recommendations"].append(
                "Expand introduction and conclusion summaries"
            )

        # Check for code examples planning
        code_sections = sum(1 for s in outline.sections if s.code_examples_needed)
        if code_sections > 0:
            completeness_factors.append(0.2)
            analysis["strengths"].append("Planned for practical code examples")

        # Check keyword planning
        if len(outline.target_keywords) >= 3:
            completeness_factors.append(0.25)
            analysis["strengths"].append("Good SEO keyword planning")
        else:
            analysis["recommendations"].append(
                "Include more target keywords (aim for 5-8)"
            )

        analysis["completeness_score"] = sum(completeness_factors)

        # Structure scoring
        structure_factors = []

        # Check section balance
        word_counts = [s.estimated_words for s in outline.sections]
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            balanced_sections = sum(
                1 for wc in word_counts if 0.5 * avg_words <= wc <= 2 * avg_words
            )
            balance_ratio = balanced_sections / len(word_counts)

            if balance_ratio >= 0.8:
                structure_factors.append(0.4)
                analysis["strengths"].append("Well-balanced section lengths")
            else:
                analysis["recommendations"].append(
                    "Balance section lengths more evenly"
                )

        # Check logical flow (basic heuristics)
        if len(outline.sections) >= 3:
            structure_factors.append(0.3)

        # Check title quality
        title_words = len(outline.title.split())
        if 4 <= title_words <= 12:
            structure_factors.append(0.3)
            analysis["strengths"].append("Title length is optimal")
        else:
            analysis["recommendations"].append(
                "Adjust title length (aim for 4-12 words)"
            )

        analysis["structure_score"] = sum(structure_factors)

        # SEO readiness
        seo_factors = []

        if outline.target_keywords:
            seo_factors.append(0.4)

            # Check if keywords appear in title
            title_lower = outline.title.lower()
            keyword_in_title = any(
                kw.lower() in title_lower for kw in outline.target_keywords
            )
            if keyword_in_title:
                seo_factors.append(0.3)
                analysis["strengths"].append("Keywords incorporated in title")

        # Check for practical/actionable content
        practical_indicators = ["how to", "guide", "tutorial", "step", "example"]
        title_lower = outline.title.lower()
        if any(indicator in title_lower for indicator in practical_indicators):
            seo_factors.append(0.3)
            analysis["strengths"].append("Title indicates practical content")

        analysis["seo_readiness"] = sum(seo_factors)

        return analysis
