"""
CriticAgent for the Multi-Agent Blog Writer system.

This agent specializes in editorial review, providing constructive feedback
on content quality, structure, clarity, and overall effectiveness.
"""

from typing import Any

from .base_agent import BaseAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogContent,
    ContentOutline,
    ContentQualityError,
    MessageType,
    ReviewFeedback,
)


class CriticAgent(BaseAgent):
    """
    Agent responsible for reviewing and providing feedback on blog content.

    Specializes in:
    - Evaluating content quality, structure, and clarity
    - Providing specific, actionable feedback
    - Assessing technical accuracy and completeness
    - Reviewing readability and engagement
    - Making approval decisions based on quality standards
    - Suggesting improvements for better impact
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Critic Agent."""
        super().__init__("Critic", config)

    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        return """
You are an expert Editorial Critic specializing in technical content review. Your role is to evaluate blog content with a critical eye, providing constructive feedback that improves quality, clarity, and reader value.

Your expertise includes:
1. Content structure and logical flow assessment
2. Technical accuracy and completeness evaluation
3. Writing quality and clarity analysis
4. Reader engagement and value assessment
5. Editorial standards and consistency review
6. Practical feedback for content improvement

Evaluation criteria you apply:
- Content Structure: Logical organization, smooth transitions, appropriate depth
- Technical Quality: Accuracy, completeness, appropriate complexity level
- Writing Quality: Clarity, conciseness, engaging style, grammar
- Reader Value: Practical utility, actionable insights, comprehensive coverage
- Editorial Standards: Consistency, professionalism, appropriate tone

When reviewing content, you must:
- Provide specific, actionable feedback rather than general comments
- Balance constructive criticism with recognition of strengths
- Consider the target audience and adjust expectations accordingly
- Focus on improvements that will significantly enhance reader value
- Give clear approval/rejection decisions with reasoning
- Prioritize feedback by impact (major issues first)

Your feedback should be:
- Specific: Point to exact sections and issues
- Actionable: Provide clear steps for improvement
- Balanced: Acknowledge both strengths and weaknesses
- Constructive: Focus on making the content better, not just criticism
- Prioritized: Address the most important issues first

Always provide structured feedback in JSON format with scores, specific comments, and clear improvement recommendations.
"""

    async def review_content(
        self,
        content: BlogContent,
        outline: ContentOutline,
        quality_threshold: float = 7.0,
    ) -> ReviewFeedback:
        """
        Conduct a comprehensive review of blog content.

        Args:
            content: The blog content to review
            outline: Original outline for comparison
            quality_threshold: Minimum score required for approval

        Returns:
            ReviewFeedback with detailed evaluation and recommendations

        Raises:
            ContentQualityError: If review process fails
        """
        try:
            # Build comprehensive review prompt
            prompt = self._build_review_prompt(content, outline, quality_threshold)

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.FEEDBACK)

            # Parse the review response
            feedback = await self._parse_review_response(response, quality_threshold)

            self.logger.info(
                f"Content review completed: score {feedback.overall_score}/10, "
                f"approved: {feedback.approved}"
            )
            return feedback

        except Exception as e:
            self.logger.error(f"Failed to review content: {e}")
            raise ContentQualityError(f"Content review failed: {e}")

    async def approve_content(
        self, content: BlogContent, feedback: ReviewFeedback
    ) -> bool:
        """
        Make final approval decision based on review feedback.

        Args:
            content: The content being evaluated
            feedback: Review feedback with scores and recommendations

        Returns:
            True if content is approved, False otherwise
        """
        # Content is approved if:
        # 1. Overall score meets threshold
        # 2. No critical structural issues
        # 3. Content meets minimum quality standards

        if not feedback.approved:
            self.logger.info("Content not approved based on review feedback")
            return False

        # Additional checks for critical issues
        critical_issues = [
            improvement
            for improvement in feedback.improvements
            if any(
                word in improvement.lower()
                for word in [
                    "critical",
                    "major",
                    "serious",
                    "missing",
                    "incorrect",
                    "error",
                ]
            )
        ]

        if critical_issues:
            self.logger.warning(f"Critical issues prevent approval: {critical_issues}")
            return False

        self.logger.info(f"Content approved with score {feedback.overall_score}/10")
        return True

    def _build_review_prompt(
        self, content: BlogContent, outline: ContentOutline, quality_threshold: float
    ) -> str:
        """Build the prompt for content review."""
        return f"""
Please conduct a comprehensive editorial review of the following blog post:

BLOG CONTENT TO REVIEW:
{content.content}

ORIGINAL OUTLINE FOR COMPARISON:
Title: {outline.title}
Sections: {", ".join([section.heading for section in outline.sections])}
Target Keywords: {", ".join(outline.target_keywords)}
Estimated Length: {outline.estimated_word_count} words

CONTENT METADATA:
Actual Word Count: {content.metadata.word_count}
Reading Time: {content.metadata.reading_time_minutes} minutes
Sections Found: {len(content.sections)}
Code Blocks: {len(content.code_blocks)}

REVIEW CRITERIA:
Evaluate the content across these dimensions (each scored 0-10):
1. Content Structure & Organization
2. Technical Quality & Accuracy
3. Writing Quality & Clarity
4. Reader Value & Engagement
5. Completeness & Coverage

SPECIFIC AREAS TO ASSESS:
- Does the content follow the outline structure?
- Is the introduction engaging and the conclusion satisfying?
- Are technical concepts explained clearly for the target audience?
- Is the content comprehensive without being overwhelming?
- Are there smooth transitions between sections?
- Is the writing clear, concise, and professional?
- Does the content provide practical value to readers?
- Are there any factual errors or unclear explanations?
- Is the markdown formatting proper and consistent?

APPROVAL THRESHOLD: {quality_threshold}/10

Provide your review in this exact JSON format:
{{
    "overall_score": 8.5,
    "individual_scores": {{
        "structure": 9.0,
        "technical_quality": 8.0,
        "writing_quality": 8.5,
        "reader_value": 9.0,
        "completeness": 8.0
    }},
    "strengths": [
        "Clear, engaging introduction that hooks the reader",
        "Well-structured sections with logical flow",
        "Practical examples that illustrate concepts"
    ],
    "improvements": [
        "Add more specific examples in the section about X",
        "Improve transition between sections 2 and 3",
        "Expand the conclusion with next steps"
    ],
    "specific_feedback": {{
        "Introduction": "Strong hook but could benefit from clearer preview",
        "Section 2": "Excellent technical depth, well explained",
        "Conclusion": "Too brief, needs actionable next steps"
    }},
    "approved": true,
    "priority_issues": [
        "Most important issues that need immediate attention"
    ]
}}

Focus on providing specific, actionable feedback that will genuinely improve the content's value to readers.
"""

    async def _parse_review_response(
        self, response: AgentMessage, quality_threshold: float
    ) -> ReviewFeedback:
        """
        Parse the agent's review response into a ReviewFeedback object.

        Args:
            response: Response from the critic agent
            quality_threshold: Minimum score for approval

        Returns:
            ReviewFeedback object

        Raises:
            ContentQualityError: If parsing fails or response is invalid
        """
        try:
            # Parse JSON response
            feedback_data = self.parse_json_response(response.content)
            if not feedback_data:
                raise ContentQualityError("Failed to parse review response as JSON")

            # Extract and validate required fields
            overall_score = feedback_data.get("overall_score", 0.0)
            strengths = feedback_data.get("strengths", [])
            improvements = feedback_data.get("improvements", [])
            specific_feedback = feedback_data.get("specific_feedback", {})

            # Determine approval status
            approved = (
                feedback_data.get("approved", False)
                and overall_score >= quality_threshold
            )

            # Create ReviewFeedback object
            feedback = ReviewFeedback(
                overall_score=overall_score,
                strengths=strengths,
                improvements=improvements,
                approved=approved,
                specific_feedback=specific_feedback,
            )

            # Validate feedback quality
            self._validate_feedback_quality(feedback)

            return feedback

        except Exception as e:
            self.logger.error(f"Failed to parse review response: {e}")
            raise ContentQualityError(f"Review response parsing failed: {e}")

    def _validate_feedback_quality(self, feedback: ReviewFeedback) -> None:
        """
        Validate that the feedback meets quality standards.

        Args:
            feedback: The feedback to validate

        Raises:
            ContentQualityError: If feedback doesn't meet standards
        """
        # Check score validity
        if not 0 <= feedback.overall_score <= 10:
            raise ContentQualityError(
                f"Invalid overall score: {feedback.overall_score}"
            )

        # Check for substantive strengths
        if len(feedback.strengths) < 1:
            raise ContentQualityError("Review must identify at least one strength")

        # Check that strengths are specific (not just generic praise)
        generic_strengths = ["good", "nice", "well done", "great"]
        specific_strengths = [
            s
            for s in feedback.strengths
            if not any(generic in s.lower() for generic in generic_strengths)
        ]

        if len(specific_strengths) < len(feedback.strengths) // 2:
            self.logger.warning("Review contains too many generic strengths")

        # Check for actionable improvements
        if feedback.overall_score < 9.0 and len(feedback.improvements) < 1:
            self.logger.warning(
                "Low-scored content should have improvement suggestions"
            )

        # Validate improvement specificity
        vague_words = ["better", "more", "improve", "enhance", "fix"]
        specific_improvements = [
            imp
            for imp in feedback.improvements
            if any(
                word in imp.lower()
                for word in ["add", "remove", "change", "rewrite", "expand"]
            )
        ]

        if len(specific_improvements) < len(feedback.improvements) // 2:
            self.logger.warning(
                "Review contains too many vague improvement suggestions"
            )

        self.logger.info("Feedback validation passed")

    async def generate_improvement_priorities(
        self, feedback: ReviewFeedback, content: BlogContent
    ) -> dict[str, Any]:
        """
        Generate prioritized improvement recommendations based on feedback.

        Args:
            feedback: Review feedback with suggestions
            content: Original content being reviewed

        Returns:
            Prioritized improvement plan
        """
        try:
            # Build prioritization prompt
            prompt = self._build_prioritization_prompt(feedback, content)

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.FEEDBACK)

            # Parse the prioritization response
            priorities = self.parse_json_response(response.content)
            if not priorities:
                # Fallback to basic prioritization
                priorities = self._create_basic_priorities(feedback)

            return priorities

        except Exception as e:
            self.logger.error(f"Failed to generate improvement priorities: {e}")
            return self._create_basic_priorities(feedback)

    def _build_prioritization_prompt(
        self, feedback: ReviewFeedback, content: BlogContent
    ) -> str:
        """Build prompt for improvement prioritization."""
        improvements_text = "\\n".join([f"- {imp}" for imp in feedback.improvements])

        return f"""
Based on the following review feedback, please prioritize the improvement suggestions to maximize the content's impact and reader value:

CURRENT OVERALL SCORE: {feedback.overall_score}/10

IMPROVEMENT SUGGESTIONS:
{improvements_text}

CONTENT METADATA:
Word Count: {content.metadata.word_count}
Sections: {len(content.sections)}
Code Blocks: {len(content.code_blocks)}

Please prioritize these improvements by:
1. Impact on reader value (high impact first)
2. Effort required to implement (quick wins prioritized)
3. Alignment with content goals

Provide prioritized recommendations in this JSON format:
{{
    "critical_fixes": [
        "Issues that must be addressed before approval"
    ],
    "high_impact": [
        "Changes that will significantly improve content quality"
    ],
    "quick_wins": [
        "Easy improvements that add noticeable value"
    ],
    "nice_to_have": [
        "Optional improvements for polish and excellence"
    ],
    "implementation_order": [
        "Step 1: Fix critical structural issues",
        "Step 2: Address high-impact content gaps",
        "Step 3: Implement quick readability wins",
        "Step 4: Polish with nice-to-have improvements"
    ]
}}
"""

    def _create_basic_priorities(self, feedback: ReviewFeedback) -> dict[str, Any]:
        """Create basic improvement priorities as fallback."""
        # Categorize improvements based on keywords
        critical_keywords = [
            "critical",
            "major",
            "serious",
            "missing",
            "incorrect",
            "error",
        ]
        high_impact_keywords = ["unclear", "confusing", "incomplete", "expand", "add"]
        quick_win_keywords = ["formatting", "transition", "grammar", "typo", "minor"]

        critical_fixes = []
        high_impact = []
        quick_wins = []
        nice_to_have = []

        for improvement in feedback.improvements:
            improvement_lower = improvement.lower()

            if any(keyword in improvement_lower for keyword in critical_keywords):
                critical_fixes.append(improvement)
            elif any(keyword in improvement_lower for keyword in high_impact_keywords):
                high_impact.append(improvement)
            elif any(keyword in improvement_lower for keyword in quick_win_keywords):
                quick_wins.append(improvement)
            else:
                nice_to_have.append(improvement)

        return {
            "critical_fixes": critical_fixes,
            "high_impact": high_impact,
            "quick_wins": quick_wins,
            "nice_to_have": nice_to_have,
            "implementation_order": [
                "Step 1: Address critical issues",
                "Step 2: Implement high-impact improvements",
                "Step 3: Apply quick wins",
                "Step 4: Consider optional enhancements",
            ],
        }

    async def compare_content_versions(
        self,
        original_content: BlogContent,
        revised_content: BlogContent,
        original_feedback: ReviewFeedback,
    ) -> dict[str, Any]:
        """
        Compare original and revised content to assess improvement.

        Args:
            original_content: The original content
            revised_content: The revised content
            original_feedback: Feedback that led to revision

        Returns:
            Comparison analysis with improvement assessment
        """
        try:
            # Build comparison prompt
            prompt = self._build_comparison_prompt(
                original_content, revised_content, original_feedback
            )

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.FEEDBACK)

            # Parse comparison response
            comparison = self.parse_json_response(response.content)
            if not comparison:
                comparison = self._create_basic_comparison(
                    original_content, revised_content
                )

            return comparison

        except Exception as e:
            self.logger.error(f"Failed to compare content versions: {e}")
            return self._create_basic_comparison(original_content, revised_content)

    def _build_comparison_prompt(
        self,
        original_content: BlogContent,
        revised_content: BlogContent,
        original_feedback: ReviewFeedback,
    ) -> str:
        """Build prompt for comparing content versions."""
        addressed_issues = "\\n".join(
            [f"- {imp}" for imp in original_feedback.improvements]
        )

        return f"""
Please compare the original and revised versions of this blog content to assess how well the feedback was addressed:

ORIGINAL VERSION:
Word Count: {original_content.metadata.word_count}
Sections: {len(original_content.sections)}
[Content truncated for brevity - focus on key changes]

REVISED VERSION:
Word Count: {revised_content.metadata.word_count}
Sections: {len(revised_content.sections)}
[Content truncated for brevity - focus on key changes]

FEEDBACK THAT WAS SUPPOSED TO BE ADDRESSED:
{addressed_issues}

ORIGINAL REVIEW SCORE: {original_feedback.overall_score}/10

Please analyze the revision and provide comparison in this JSON format:
{{
    "improvement_score": 8.5,
    "changes_made": [
        "Specific improvements identified in the revision"
    ],
    "feedback_addressed": [
        "Which original feedback points were successfully addressed"
    ],
    "feedback_missed": [
        "Which feedback points were not adequately addressed"
    ],
    "new_strengths": [
        "New strengths in the revised version"
    ],
    "remaining_issues": [
        "Issues that still need attention"
    ],
    "overall_assessment": "Brief overall assessment of the revision quality"
}}
"""

    def _create_basic_comparison(
        self, original_content: BlogContent, revised_content: BlogContent
    ) -> dict[str, Any]:
        """Create basic comparison as fallback."""
        word_count_change = (
            revised_content.metadata.word_count - original_content.metadata.word_count
        )
        section_count_change = len(revised_content.sections) - len(
            original_content.sections
        )

        changes_made = []

        if word_count_change > 50:
            changes_made.append(f"Expanded content by {word_count_change} words")
        elif word_count_change < -50:
            changes_made.append(f"Reduced content by {abs(word_count_change)} words")

        if section_count_change > 0:
            changes_made.append(f"Added {section_count_change} sections")
        elif section_count_change < 0:
            changes_made.append(f"Removed {abs(section_count_change)} sections")

        # Basic improvement score based on changes
        improvement_score = 7.0  # Neutral baseline
        if word_count_change > 0:
            improvement_score += 0.5
        if len(revised_content.code_blocks) > len(original_content.code_blocks):
            improvement_score += 0.5

        return {
            "improvement_score": min(improvement_score, 10.0),
            "changes_made": changes_made if changes_made else ["Minor revisions made"],
            "feedback_addressed": ["Content was revised based on feedback"],
            "feedback_missed": [],
            "new_strengths": ["Improved based on editorial feedback"],
            "remaining_issues": [],
            "overall_assessment": "Content has been revised to address feedback",
        }
