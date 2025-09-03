"""
Dialogue Generation for Conversational Blog Writer.

This module provides dialogue generation capabilities, conversation flow management,
and technical contextualization for creating natural developer-focused conversations.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .information_synthesizer import CodeExample, SynthesizedKnowledge, TechnicalDetail
from .multi_agent_models import BlogGenerationError, BlogInput, ContentOutline
from .persona_system import (
    DialogueExchange,
    DialogueSection,
    ProblemPresenter,
    SolutionProvider,
)


class DialogueGenerationError(BlogGenerationError):
    """Raised when dialogue generation fails."""

    pass


class DialogueFlowError(DialogueGenerationError):
    """Raised when dialogue flow is invalid."""

    pass


class ConversationType(str, Enum):
    """Types of conversational structures."""

    PROBLEM_SOLUTION = "problem_solution"
    TUTORIAL_WALKTHROUGH = "tutorial_walkthrough"
    CONCEPT_EXPLORATION = "concept_exploration"
    COMPARISON_DISCUSSION = "comparison_discussion"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"


@dataclass
class ConversationContext:
    """Context information for dialogue generation."""

    main_topic: str
    technical_concepts: list[str] = field(default_factory=list)
    target_audience_level: str = "intermediate"
    conversation_goals: list[str] = field(default_factory=list)
    available_examples: list[CodeExample] = field(default_factory=list)
    key_insights: list[str] = field(default_factory=list)
    problem_solution_pairs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DialogueFlow:
    """Represents the flow structure of a conversation."""

    introduction_exchanges: int = 2
    main_discussion_exchanges: int = 8
    example_exchanges: int = 4
    conclusion_exchanges: int = 2
    max_consecutive_speaker: int = 2
    transition_phrases: list[str] = field(default_factory=list)


class ConversationFlowManager:
    """Manages conversation flow and transitions."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Default transition phrases
        self.transition_phrases = {
            "topic_change": [
                "Speaking of that, let me ask about",
                "That brings up another point",
                "Building on that idea",
                "Another thing to consider is",
            ],
            "example_introduction": [
                "Let me show you a practical example",
                "Here's how that would look in practice",
                "To illustrate this concept",
                "For instance",
            ],
            "problem_to_solution": [
                "That's a common challenge. Here's how I approach it",
                "I've seen this before. The solution is",
                "Good question. Let me walk you through",
                "There's actually a straightforward way to handle this",
            ],
            "agreement": [
                "Exactly! And to add to that",
                "That's right, and",
                "Absolutely. Another benefit is",
                "Good point. Also",
            ],
            "clarification": [
                "Just to clarify",
                "Let me make sure I understand",
                "What you're describing sounds like",
                "If I'm following correctly",
            ],
        }

    def create_flow_plan(
        self, outline: ContentOutline, context: ConversationContext
    ) -> DialogueFlow:
        """Create a dialogue flow plan based on content outline."""
        total_sections = len(outline.sections)
        estimated_exchanges = max(12, total_sections * 3)  # Minimum 12 exchanges

        # Distribute exchanges across flow stages
        intro_exchanges = min(3, max(1, estimated_exchanges // 8))
        conclusion_exchanges = min(3, max(1, estimated_exchanges // 8))
        example_exchanges = min(6, max(2, estimated_exchanges // 4))
        main_exchanges = (
            estimated_exchanges
            - intro_exchanges
            - conclusion_exchanges
            - example_exchanges
        )

        return DialogueFlow(
            introduction_exchanges=intro_exchanges,
            main_discussion_exchanges=main_exchanges,
            example_exchanges=example_exchanges,
            conclusion_exchanges=conclusion_exchanges,
            transition_phrases=self._select_transition_phrases(context),
        )

    def _select_transition_phrases(self, context: ConversationContext) -> list[str]:
        """Select appropriate transition phrases for the context."""
        phrases = []

        # Add phrases based on conversation goals
        if "problem_solving" in str(context.conversation_goals):
            phrases.extend(self.transition_phrases["problem_to_solution"][:2])

        phrases.extend(self.transition_phrases["topic_change"][:2])
        phrases.extend(self.transition_phrases["example_introduction"][:1])

        return phrases

    def validate_flow(
        self, exchanges: list[DialogueExchange]
    ) -> tuple[bool, list[str]]:
        """Validate that dialogue follows good conversation flow."""
        issues = []

        if len(exchanges) < 4:
            issues.append("Dialogue too short - needs at least 4 exchanges")

        # Check for reasonable speaker alternation
        consecutive_count = 1
        current_speaker = None

        for exchange in exchanges:
            if exchange.speaker == current_speaker:
                consecutive_count += 1
                if consecutive_count > 3:
                    issues.append(
                        f"Too many consecutive exchanges by {exchange.speaker}"
                    )
                    break
            else:
                consecutive_count = 1
                current_speaker = exchange.speaker

        # Check for natural conversation patterns
        question_count = sum(1 for ex in exchanges if "?" in ex.content)
        if question_count == 0:
            issues.append("No questions found - conversation lacks natural inquiry")
        elif question_count > len(exchanges) // 2:
            issues.append(
                "Too many questions - balance with statements and explanations"
            )

        return len(issues) == 0, issues


class TechnicalContextualizer:
    """Ensures technical accuracy and context in conversations."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def contextualize_technical_discussion(
        self,
        topic: str,
        technical_details: list[TechnicalDetail],
        target_level: str = "intermediate",
    ) -> dict[str, Any]:
        """Create technical context for conversation."""
        relevant_details = [
            detail
            for detail in technical_details
            if topic.lower() in detail.concept.lower()
            or any(
                topic.lower() in tech.lower() for tech in detail.related_technologies
            )
        ]

        context = {
            "main_concept": topic,
            "complexity_level": target_level,
            "key_points": [],
            "common_pitfalls": [],
            "best_practices": [],
            "code_examples": [],
            "related_concepts": set(),
        }

        for detail in relevant_details[:5]:  # Limit to top 5 relevant details
            context["key_points"].append(detail.description)
            context["related_concepts"].update(detail.related_technologies)

            # Extract best practices from use cases
            if detail.use_cases:
                context["best_practices"].extend(detail.use_cases[:2])

        return context

    def generate_technical_examples(
        self, concept: str, code_examples: list[CodeExample]
    ) -> list[str]:
        """Generate technical examples for a concept."""
        relevant_examples = [
            example
            for example in code_examples
            if concept.lower() in example.technical_concepts
            or concept.lower() in example.explanation.lower()
        ]

        example_texts = []
        for example in relevant_examples[:2]:  # Limit to 2 examples
            example_text = f"Here's a {example.language} example:\n\n```{example.language}\n{example.code_snippet}\n```\n\n{example.explanation}"
            example_texts.append(example_text)

        return example_texts

    def validate_technical_accuracy(
        self, content: str, context: dict[str, Any]
    ) -> list[str]:
        """Validate technical accuracy of content."""
        issues = []

        # Check if technical concepts are mentioned appropriately
        mentioned_concepts = set(re.findall(r"\b\w+\b", content.lower()))
        expected_concepts = context.get("related_concepts", set())

        # Ensure some technical depth
        if len(mentioned_concepts.intersection(expected_concepts)) == 0:
            issues.append("Content lacks expected technical concepts")

        return issues


class DialogueGenerator:
    """Main dialogue generator for conversational blog content."""

    def __init__(self, agent_config):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.flow_manager = ConversationFlowManager()
        self.contextualizer = TechnicalContextualizer()
        self.agent_config = agent_config

    async def generate_dialogue_sections(
        self,
        outline: ContentOutline,
        blog_input: BlogInput,
        synthesized_knowledge: SynthesizedKnowledge,
        personas: tuple[ProblemPresenter, SolutionProvider],
    ) -> list[DialogueSection]:
        """
        Generate dialogue sections based on content outline and knowledge.

        Args:
            outline: Content structure to follow
            blog_input: Original blog requirements
            synthesized_knowledge: Research-based knowledge
            personas: Configured conversation personas

        Returns:
            List of dialogue sections
        """
        try:
            # Create conversation context
            context = self._create_conversation_context(
                outline, blog_input, synthesized_knowledge
            )

            # Generate dialogue flow plan
            flow_plan = self.flow_manager.create_flow_plan(outline, context)

            # Generate sections
            sections = []

            # Introduction section
            intro_section = await self._generate_introduction_section(
                outline, context, personas, flow_plan
            )
            sections.append(intro_section)

            # Main content sections
            for i, content_section in enumerate(outline.sections):
                dialogue_section = await self._generate_content_section(
                    content_section, context, personas, i == len(outline.sections) - 1
                )
                sections.append(dialogue_section)

            # Conclusion section
            conclusion_section = await self._generate_conclusion_section(
                outline, context, personas, flow_plan
            )
            sections.append(conclusion_section)

            self.logger.info(f"Generated {len(sections)} dialogue sections")
            return sections

        except Exception as e:
            self.logger.error(f"Dialogue generation failed: {e}")
            raise DialogueGenerationError(f"Failed to generate dialogue: {e}") from e

    def _create_conversation_context(
        self,
        outline: ContentOutline,
        blog_input: BlogInput,
        synthesized_knowledge: SynthesizedKnowledge,
    ) -> ConversationContext:
        """Create context for conversation generation."""
        return ConversationContext(
            main_topic=outline.title,
            technical_concepts=list(
                synthesized_knowledge.original_knowledge.technical_concepts
            )[:10],
            target_audience_level=blog_input.target_audience.value,
            conversation_goals=[
                "Explore practical development challenges",
                "Provide actionable solutions",
                "Share real-world examples",
                "Discuss best practices",
            ],
            available_examples=synthesized_knowledge.code_examples,
            key_insights=[
                insight.content
                for insight in synthesized_knowledge.original_knowledge.insights
            ][:8],
            problem_solution_pairs=synthesized_knowledge.problem_solution_pairs,
        )

    async def _generate_introduction_section(
        self,
        outline: ContentOutline,
        context: ConversationContext,
        personas: tuple[ProblemPresenter, SolutionProvider],
        flow_plan: DialogueFlow,
    ) -> DialogueSection:
        """Generate introduction dialogue section."""
        problem_presenter, solution_provider = personas

        exchanges = []

        # Problem presenter introduces the topic
        intro_content = f"I've been working with {context.main_topic} recently, and I'm curious about best practices. What would you say are the key things developers should know?"

        exchanges.append(
            DialogueExchange(
                speaker=problem_presenter.profile.name,
                content=intro_content,
                intent="question",
                technical_concepts=[context.main_topic.lower()],
                confidence_level=0.8,
            )
        )

        # Solution provider responds with overview
        overview_concepts = context.technical_concepts[:3]
        response_content = f"That's a great question! When it comes to {context.main_topic}, there are several important aspects to consider. The main areas I'd focus on are {', '.join(overview_concepts[:2])}. Let's dive into the practical challenges and solutions."

        exchanges.append(
            DialogueExchange(
                speaker=solution_provider.profile.name,
                content=response_content,
                intent="explanation",
                technical_concepts=overview_concepts,
                confidence_level=0.9,
            )
        )

        return DialogueSection(
            section_title="Introduction",
            exchanges=exchanges,
            technical_focus=context.main_topic,
            learning_objective="Set context and establish conversation flow",
            section_type="introduction",
        )

    async def _generate_content_section(
        self,
        content_section,
        context: ConversationContext,
        personas: tuple[ProblemPresenter, SolutionProvider],
        is_last_section: bool,
    ) -> DialogueSection:
        """Generate dialogue for a main content section."""
        problem_presenter, solution_provider = personas
        exchanges = []

        section_title = content_section.heading
        key_points = content_section.key_points

        # Technical context for this section
        self.contextualizer.contextualize_technical_discussion(
            section_title,
            [],  # We'll pass actual technical details when available
            context.target_audience_level,
        )

        # Problem presenter asks about or presents challenge related to section
        problem_patterns = [
            f"When working with {section_title}, what are the main challenges you encounter?",
            f"I've been struggling with {section_title}. What's your approach?",
            f"Can you walk me through best practices for {section_title}?",
            f"What are common mistakes people make with {section_title}?",
        ]

        problem_content = random.choice(problem_patterns)
        exchanges.append(
            DialogueExchange(
                speaker=problem_presenter.profile.name,
                content=problem_content,
                intent="question",
                technical_concepts=[section_title.lower()],
                confidence_level=0.7,
            )
        )

        # Solution provider addresses the key points
        solution_content = f"Good question! With {section_title}, the key things to keep in mind are:\n\n"

        for i, point in enumerate(key_points[:3], 1):
            solution_content += f"{i}. {point}\n"

        if context.available_examples:
            solution_content += "\nLet me show you how this looks in practice."

        exchanges.append(
            DialogueExchange(
                speaker=solution_provider.profile.name,
                content=solution_content,
                intent="explanation",
                technical_concepts=[section_title.lower()]
                + [point.lower() for point in key_points[:2]],
                confidence_level=0.9,
            )
        )

        # Add code example if available and needed
        if content_section.code_examples_needed and context.available_examples:
            example_exchange = self._generate_code_example_exchange(
                section_title,
                context.available_examples,
                solution_provider.profile.name,
            )
            if example_exchange:
                exchanges.append(example_exchange)

                # Problem presenter follow-up
                followup_content = "That's really helpful! Are there any edge cases or performance considerations I should be aware of?"
                exchanges.append(
                    DialogueExchange(
                        speaker=problem_presenter.profile.name,
                        content=followup_content,
                        intent="question",
                        technical_concepts=[],
                        confidence_level=0.8,
                    )
                )

        return DialogueSection(
            section_title=section_title,
            exchanges=exchanges,
            technical_focus=section_title,
            learning_objective=f"Understand {section_title} concepts and best practices",
            section_type="discussion",
        )

    def _generate_code_example_exchange(
        self, topic: str, code_examples: list[CodeExample], speaker_name: str
    ) -> DialogueExchange | None:
        """Generate an exchange with a code example."""
        relevant_examples = [
            ex
            for ex in code_examples
            if topic.lower() in ex.explanation.lower()
            or any(topic.lower() in concept for concept in ex.technical_concepts)
        ]

        if not relevant_examples:
            return None

        example = relevant_examples[0]

        content = f"Here's a practical {example.language} example:\n\n```{example.language}\n{example.code_snippet}\n```\n\n{example.explanation}"

        return DialogueExchange(
            speaker=speaker_name,
            content=content,
            intent="example",
            technical_concepts=example.technical_concepts,
            confidence_level=0.9,
        )

    async def _generate_conclusion_section(
        self,
        outline: ContentOutline,
        context: ConversationContext,
        personas: tuple[ProblemPresenter, SolutionProvider],
        flow_plan: DialogueFlow,
    ) -> DialogueSection:
        """Generate conclusion dialogue section."""
        problem_presenter, solution_provider = personas
        exchanges = []

        # Problem presenter summarizes learning
        summary_content = f"This has been really insightful! To summarize, the key takeaways for {context.main_topic} are the importance of {', '.join(context.technical_concepts[:2])}. What would be your top recommendation for someone just getting started?"

        exchanges.append(
            DialogueExchange(
                speaker=problem_presenter.profile.name,
                content=summary_content,
                intent="summary",
                technical_concepts=context.technical_concepts[:2],
                confidence_level=0.8,
            )
        )

        # Solution provider provides final advice
        advice_content = "Great summary! My top recommendation would be to start with the fundamentals and practice regularly. Focus on understanding the core concepts before moving to advanced topics. And remember, every expert was once a beginner - keep experimenting and learning!"

        exchanges.append(
            DialogueExchange(
                speaker=solution_provider.profile.name,
                content=advice_content,
                intent="advice",
                technical_concepts=[],
                confidence_level=0.9,
            )
        )

        return DialogueSection(
            section_title="Conclusion",
            exchanges=exchanges,
            technical_focus="general advice",
            learning_objective="Reinforce key points and provide next steps",
            section_type="conclusion",
        )

    async def format_dialogue_as_markdown(self, sections: list[DialogueSection]) -> str:
        """Format dialogue sections as markdown content."""
        markdown_content = []

        for section in sections:
            if section.section_type == "introduction":
                # Don't add a header for introduction - it flows naturally
                pass
            elif section.section_type == "conclusion":
                markdown_content.append("## Key Takeaways\n")
            else:
                markdown_content.append(f"## {section.section_title}\n")

            # Add exchanges
            for exchange in section.exchanges:
                speaker_name = exchange.speaker

                # Format as conversational markdown
                if exchange.intent == "question":
                    markdown_content.append(f"**{speaker_name}:** {exchange.content}\n")
                elif exchange.intent == "example" and "```" in exchange.content:
                    # Code examples get special formatting
                    markdown_content.append(f"**{speaker_name}:** {exchange.content}\n")
                else:
                    markdown_content.append(f"**{speaker_name}:** {exchange.content}\n")

                markdown_content.append("")  # Empty line for spacing

            markdown_content.append("")  # Section separator

        return "\n".join(markdown_content)

    async def validate_generated_dialogue(
        self,
        sections: list[DialogueSection],
        personas: tuple[ProblemPresenter, SolutionProvider],
    ) -> tuple[bool, list[str]]:
        """Validate the generated dialogue for quality and consistency."""
        all_exchanges = []
        for section in sections:
            all_exchanges.extend(section.exchanges)

        # Use flow manager to validate conversation flow
        flow_valid, flow_issues = self.flow_manager.validate_flow(all_exchanges)

        # Additional quality checks
        quality_issues = []

        # Check technical content coverage
        all_technical_concepts = set()
        for exchange in all_exchanges:
            all_technical_concepts.update(exchange.technical_concepts)

        if len(all_technical_concepts) < 3:
            quality_issues.append("Insufficient technical concepts covered")

        # Check for code examples if needed
        has_code = any("```" in ex.content for ex in all_exchanges)
        example_exchanges = [ex for ex in all_exchanges if ex.intent == "example"]

        if len(example_exchanges) == 0 and not has_code:
            quality_issues.append("No practical examples provided")

        # Check conversation balance
        speaker_counts = {}
        for exchange in all_exchanges:
            speaker_counts[exchange.speaker] = (
                speaker_counts.get(exchange.speaker, 0) + 1
            )

        if len(speaker_counts) == 2:
            counts = list(speaker_counts.values())
            if max(counts) > min(counts) * 2.5:
                quality_issues.append("Conversation is imbalanced between speakers")

        all_issues = flow_issues + quality_issues
        return len(all_issues) == 0, all_issues
