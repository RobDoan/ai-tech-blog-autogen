"""
Conversational Writer Agent for Multi-Agent Blog Writer system.

This agent extends WriterAgent to generate conversational blog content using
personas, research knowledge, and dialogue generation capabilities.
"""

import asyncio
from typing import Optional, List, Dict, Any, Tuple
import logging
from pydantic import Field

from .writer_agent import WriterAgent
from .persona_system import (
    PersonaConfig, PersonaManager, ProblemPresenter, SolutionProvider,
    create_default_persona_config
)
from .dialogue_generator import DialogueGenerator, DialogueSection
from .information_synthesizer import SynthesizedKnowledge
from .research_processor import ResearchProcessor, KnowledgeBase
from .multi_agent_models import (
    AgentConfig,
    BlogInput,
    ContentOutline,
    BlogContent,
    ContentMetadata,
    MessageType,
    AgentMessage,
    ContentQualityError
)


class ConversationalBlogContent(BlogContent):
    """Extended blog content with conversational metadata."""
    dialogue_sections: List[DialogueSection] = Field(default_factory=list, description="Generated dialogue sections")
    personas_used: Optional[Tuple[str, str]] = Field(None, description="Names of personas used in the conversation")
    research_sources: List[str] = Field(default_factory=list, description="List of research sources referenced")
    conversation_flow_score: float = Field(0.0, description="Quality score for conversation flow", ge=0.0, le=10.0)
    synthesis_confidence: float = Field(0.0, description="Confidence in knowledge synthesis", ge=0.0, le=1.0)


class ConversationalWriterAgent(WriterAgent):
    """
    Agent for generating conversational blog content using personas and research.

    Extends WriterAgent with:
    - Persona-based conversation generation
    - Research knowledge integration
    - Dialogue flow management
    - Technical contextualization
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Conversational Writer Agent."""
        super().__init__(config)
        self.agent_name = "ConversationalWriter"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize conversational components
        self.persona_manager = PersonaManager()
        self.dialogue_generator = DialogueGenerator(config)

        # Cache for active personas and knowledge
        self._active_personas: Optional[Tuple[ProblemPresenter, SolutionProvider]] = None
        self._current_knowledge: Optional[SynthesizedKnowledge] = None
        self._persona_config: Optional[PersonaConfig] = None

    def _get_system_message(self) -> str:
        """Get the system message for conversational writing."""
        return """
You are an expert Conversational Technical Writer specializing in creating engaging developer-focused blog content in dialogue format. Your role is to transform structured outlines into natural conversations between two personas that educate and engage technical audiences.

Your expertise includes:
1. Creating realistic developer conversations and scenarios
2. Maintaining consistent persona voices throughout dialogue
3. Integrating technical concepts naturally into conversations
4. Balancing problem presentation with practical solutions
5. Using research knowledge to ensure technical accuracy
6. Structuring conversations with logical flow and transitions
7. Adapting technical depth to target audience level

Conversational writing principles:
- Create natural, authentic dialogue between personas
- Balance questions, explanations, and examples
- Maintain persona consistency in voice and expertise level
- Integrate technical concepts organically into conversation
- Use practical examples and real-world scenarios
- Ensure smooth transitions between topics
- Make technical content accessible and engaging

When generating conversational content, you must:
- Follow the established persona profiles exactly
- Maintain character voices and communication styles
- Include practical examples and code when appropriate
- Ensure technical accuracy using provided research
- Create natural conversation flow with proper pacing
- Balance dialogue between both personas
- Use markdown formatting for structure and code blocks

Focus on creating conversations that feel authentic while delivering valuable technical knowledge that serves the target audience's learning goals.
"""

    async def write_conversational_content(
        self,
        outline: ContentOutline,
        blog_input: BlogInput,
        research_knowledge: Optional[SynthesizedKnowledge] = None,
        personas: Optional[Tuple[ProblemPresenter, SolutionProvider]] = None,
        persona_config: Optional[PersonaConfig] = None
    ) -> ConversationalBlogContent:
        """
        Generate conversational blog content using personas and research.

        Args:
            outline: Content structure to follow
            blog_input: Original blog requirements
            research_knowledge: Synthesized research knowledge (optional)
            personas: Configured personas (optional, will create defaults)
            persona_config: Persona configuration (optional)

        Returns:
            ConversationalBlogContent with dialogue-based blog post

        Raises:
            ContentQualityError: If content generation fails
        """
        try:
            self.logger.info(f"Starting conversational content generation for: {outline.title}")

            # Set up personas if not provided
            if personas is None or persona_config is None:
                persona_config = persona_config or create_default_persona_config()
                personas = self.persona_manager.create_personas(persona_config)

            self._active_personas = personas
            self._persona_config = persona_config
            self._current_knowledge = research_knowledge

            # Generate dialogue sections
            dialogue_sections = await self.dialogue_generator.generate_dialogue_sections(
                outline, blog_input, research_knowledge or SynthesizedKnowledge(
                    original_knowledge=KnowledgeBase()
                ), personas
            )

            # Validate dialogue quality
            is_valid, validation_issues = await self.dialogue_generator.validate_generated_dialogue(
                dialogue_sections, personas
            )

            if not is_valid:
                self.logger.warning(f"Dialogue validation issues: {validation_issues}")
                # Continue but log the issues

            # Format dialogue as markdown
            conversational_content = await self.dialogue_generator.format_dialogue_as_markdown(
                dialogue_sections
            )

            # Add title and structure
            formatted_content = await self._format_conversational_blog(
                outline, conversational_content, blog_input
            )

            # Calculate metadata
            metadata = self._calculate_conversational_metadata(formatted_content, dialogue_sections)

            # Create conversational blog content
            blog_content = ConversationalBlogContent(
                title=outline.title,
                content=formatted_content,
                sections=self._extract_sections_from_content(formatted_content),
                code_blocks=self._extract_code_blocks_from_content(formatted_content),
                metadata=metadata,
                dialogue_sections=dialogue_sections,
                personas_used=(personas[0].profile.name, personas[1].profile.name),
                research_sources=research_knowledge.original_knowledge.references if research_knowledge else [],
                conversation_flow_score=self._calculate_flow_score(dialogue_sections),
                synthesis_confidence=research_knowledge.synthesis_confidence if research_knowledge else 0.0
            )

            # Validate content quality
            await self._validate_conversational_content(blog_content, outline, blog_input)

            self.logger.info(f"Generated conversational content: {metadata.word_count} words, {len(dialogue_sections)} sections")
            return blog_content

        except Exception as e:
            self.logger.error(f"Conversational content generation failed: {e}")
            raise ContentQualityError(f"Conversational content generation failed: {e}")

    async def _format_conversational_blog(
        self,
        outline: ContentOutline,
        conversational_content: str,
        blog_input: BlogInput
    ) -> str:
        """Format conversational content into a complete blog post."""
        formatted_parts = []

        # Add title
        formatted_parts.append(f"# {outline.title}")
        formatted_parts.append("")

        # Add introduction context
        intro_context = await self._generate_conversational_intro(outline, blog_input)
        formatted_parts.append(intro_context)
        formatted_parts.append("")

        # Add persona introductions
        if self._active_personas and self._persona_config:
            persona_intro = self._generate_persona_introductions()
            formatted_parts.append(persona_intro)
            formatted_parts.append("")

        # Add main conversational content
        formatted_parts.append(conversational_content)

        # Add conclusion context
        conclusion_context = await self._generate_conversational_conclusion(outline, blog_input)
        formatted_parts.append(conclusion_context)

        return "\n".join(formatted_parts)

    async def _generate_conversational_intro(
        self,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> str:
        """Generate an introduction that sets up the conversational format."""
        intro_prompt = f"""
Write a brief introduction (2-3 paragraphs) that introduces the topic "{outline.title}" and sets up the conversational format that follows.

The introduction should:
- Establish the value and importance of the topic
- Briefly mention that the content is presented as a conversation between two experienced developers
- Set expectations for practical, real-world insights
- Be appropriate for {blog_input.target_audience.value} level developers

Topic: {outline.title}
Target Audience: {blog_input.target_audience.value}
Additional Context: {blog_input.description or 'None'}

Write a natural, engaging introduction that flows into the conversation format.
"""

        response = await self.query_agent(intro_prompt, MessageType.CONTENT)
        return self.clean_markdown_content(response.content)

    def _generate_persona_introductions(self) -> str:
        """Generate introductions for the conversation personas."""
        if not self._active_personas or not self._persona_config:
            return ""

        problem_presenter, solution_provider = self._active_personas

        intro_text = "In this conversation, you'll follow along as two developers discuss their experiences:\n\n"

        intro_text += f"**{problem_presenter.profile.name}**: {problem_presenter.profile.background}. "
        intro_text += f"Specializes in {', '.join(problem_presenter.profile.expertise_areas[:2])}.\n\n"

        intro_text += f"**{solution_provider.profile.name}**: {solution_provider.profile.background}. "
        intro_text += f"Expert in {', '.join(solution_provider.profile.expertise_areas[:2])}.\n\n"

        intro_text += "Let's listen in on their conversation:"

        return intro_text

    async def _generate_conversational_conclusion(
        self,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> str:
        """Generate a conclusion that wraps up the conversational format."""
        conclusion_prompt = f"""
Write a brief conclusion (1-2 paragraphs) that wraps up the conversational blog post about "{outline.title}".

The conclusion should:
- Summarize the key insights from the conversation
- Reinforce the practical value for {blog_input.target_audience.value} developers
- Encourage readers to apply what they've learned
- Thank the conversation participants (if appropriate)

Topic: {outline.title}
Target Audience: {blog_input.target_audience.value}

Write a natural conclusion that provides closure to the conversational format.
"""

        response = await self.query_agent(conclusion_prompt, MessageType.CONTENT)
        return self.clean_markdown_content(response.content)

    def _calculate_conversational_metadata(
        self,
        content: str,
        dialogue_sections: List[DialogueSection]
    ) -> ContentMetadata:
        """Calculate metadata specific to conversational content."""
        # Base metadata from parent class
        base_metadata = self._calculate_content_metadata(content)

        # Add conversational-specific metrics
        total_exchanges = sum(len(section.exchanges) for section in dialogue_sections)

        # Estimate reading time (conversational content reads differently)
        # Conversational content often reads faster due to dialogue format
        adjusted_reading_time = max(1, int(base_metadata.reading_time_minutes * 0.85))

        return ContentMetadata(
            word_count=base_metadata.word_count,
            reading_time_minutes=adjusted_reading_time,
            seo_score=base_metadata.seo_score,
            keywords=base_metadata.keywords + [
                "conversation", "dialogue", "discussion", "practical"
            ],
            meta_description=base_metadata.meta_description
        )

    def _calculate_flow_score(self, dialogue_sections: List[DialogueSection]) -> float:
        """Calculate a score for conversation flow quality."""
        if not dialogue_sections:
            return 0.0

        total_exchanges = sum(len(section.exchanges) for section in dialogue_sections)

        if total_exchanges == 0:
            return 0.0

        score_factors = []

        # Factor 1: Reasonable number of exchanges
        if 8 <= total_exchanges <= 20:
            score_factors.append(1.0)
        elif 6 <= total_exchanges < 8 or 20 < total_exchanges <= 25:
            score_factors.append(0.8)
        else:
            score_factors.append(0.6)

        # Factor 2: Balanced sections
        section_lengths = [len(section.exchanges) for section in dialogue_sections]
        if section_lengths:
            length_variance = max(section_lengths) - min(section_lengths)
            if length_variance <= 2:
                score_factors.append(1.0)
            elif length_variance <= 4:
                score_factors.append(0.8)
            else:
                score_factors.append(0.6)

        # Factor 3: Technical content coverage
        all_exchanges = []
        for section in dialogue_sections:
            all_exchanges.extend(section.exchanges)

        technical_concepts = set()
        for exchange in all_exchanges:
            technical_concepts.update(exchange.technical_concepts)

        if len(technical_concepts) >= 5:
            score_factors.append(1.0)
        elif len(technical_concepts) >= 3:
            score_factors.append(0.8)
        else:
            score_factors.append(0.6)

        # Factor 4: Intent variety
        intents = set(exchange.intent for exchange in all_exchanges)
        if len(intents) >= 4:
            score_factors.append(1.0)
        elif len(intents) >= 3:
            score_factors.append(0.8)
        else:
            score_factors.append(0.6)

        return sum(score_factors) / len(score_factors) if score_factors else 0.5

    async def _validate_conversational_content(
        self,
        content: ConversationalBlogContent,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> None:
        """Validate conversational content meets quality standards."""
        # Use base validation first
        await super()._validate_content_quality(content, outline, blog_input)

        # Additional conversational validation
        if not content.dialogue_sections:
            raise ContentQualityError("No dialogue sections found in conversational content")

        if len(content.dialogue_sections) < 2:
            raise ContentQualityError("Conversational content needs at least 2 dialogue sections")

        # Check for persona balance
        if content.personas_used:
            persona1, persona2 = content.personas_used
            persona1_count = content.content.count(f"**{persona1}:**")
            persona2_count = content.content.count(f"**{persona2}:**")

            if persona1_count == 0 or persona2_count == 0:
                raise ContentQualityError("Both personas must participate in the conversation")

            # Check for reasonable balance (not more than 3:1 ratio)
            ratio = max(persona1_count, persona2_count) / max(min(persona1_count, persona2_count), 1)
            if ratio > 3.0:
                raise ContentQualityError("Conversation is too imbalanced between personas")

        # Check conversation flow score
        if content.conversation_flow_score < 0.4:
            raise ContentQualityError(f"Conversation flow quality too low: {content.conversation_flow_score:.2f}")

        self.logger.info("Conversational content validation passed")

    async def revise_conversational_content(
        self,
        current_content: ConversationalBlogContent,
        feedback: str,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> ConversationalBlogContent:
        """
        Revise conversational content based on feedback.

        Args:
            current_content: The content to revise
            feedback: Feedback for improvement
            outline: Original outline for reference
            blog_input: Original input requirements

        Returns:
            Revised conversational blog content
        """
        try:
            # Build revision prompt specific to conversational content
            prompt = self._build_conversational_revision_prompt(
                current_content, feedback, outline, blog_input
            )

            # Query the agent
            response = await self.query_agent(prompt, MessageType.CONTENT)

            # Parse the revised content
            revised_content_text = self.clean_markdown_content(response.content)

            # Create new conversational blog content
            revised_metadata = self._calculate_conversational_metadata(
                revised_content_text, current_content.dialogue_sections
            )

            revised_content = ConversationalBlogContent(
                title=current_content.title,
                content=revised_content_text,
                sections=self._extract_sections_from_content(revised_content_text),
                code_blocks=self._extract_code_blocks_from_content(revised_content_text),
                metadata=revised_metadata,
                dialogue_sections=current_content.dialogue_sections,
                personas_used=current_content.personas_used,
                research_sources=current_content.research_sources,
                conversation_flow_score=current_content.conversation_flow_score,
                synthesis_confidence=current_content.synthesis_confidence
            )

            self.logger.info(f"Revised conversational content: {revised_metadata.word_count} words")
            return revised_content

        except Exception as e:
            self.logger.error(f"Failed to revise conversational content: {e}")
            raise ContentQualityError(f"Conversational content revision failed: {e}")

    def _build_conversational_revision_prompt(
        self,
        current_content: ConversationalBlogContent,
        feedback: str,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> str:
        """Build prompt for revising conversational content."""
        persona_info = ""
        if current_content.personas_used:
            persona1, persona2 = current_content.personas_used
            persona_info = f"Personas: {persona1} and {persona2}"

        return f"""
Please revise the following conversational blog post based on the feedback provided:

CURRENT CONVERSATIONAL BLOG POST:
{current_content.content}

FEEDBACK TO ADDRESS:
{feedback}

ORIGINAL REQUIREMENTS:
- Title: {outline.title}
- Target Audience: {blog_input.target_audience.value}
- Target Length: {blog_input.preferred_length} words
- Format: Conversational dialogue between two personas
{persona_info}

REVISION GUIDELINES FOR CONVERSATIONAL CONTENT:
1. Address all points mentioned in the feedback
2. Maintain the conversational dialogue format throughout
3. Keep both personas active and engaged in the conversation
4. Ensure persona voices remain consistent with their established characteristics
5. Preserve natural conversation flow and transitions
6. Include practical examples and technical concepts as appropriate
7. Maintain proper markdown formatting for dialogue (use **Name:** format)
8. Keep the content at the appropriate level for {blog_input.target_audience.value} readers
9. Ensure the conversation feels authentic and educational

Please provide the complete revised conversational blog post in markdown format, maintaining the dialogue structure while incorporating all feedback.
"""

    async def analyze_conversational_structure(
        self,
        content: ConversationalBlogContent
    ) -> Dict[str, Any]:
        """
        Analyze the structure and quality of conversational content.

        Args:
            content: Conversational content to analyze

        Returns:
            Analysis results with conversational-specific metrics
        """
        # Get base analysis
        base_analysis = await super().analyze_content_structure(content)

        # Add conversational-specific analysis
        conversational_analysis = {
            "dialogue_sections_count": len(content.dialogue_sections),
            "total_exchanges": sum(len(section.exchanges) for section in content.dialogue_sections),
            "conversation_flow_score": content.conversation_flow_score,
            "persona_balance": {},
            "technical_concept_coverage": 0,
            "intent_variety": set(),
            "conversational_strengths": [],
            "conversational_recommendations": []
        }

        if content.dialogue_sections:
            all_exchanges = []
            for section in content.dialogue_sections:
                all_exchanges.extend(section.exchanges)

            # Analyze persona balance
            if content.personas_used:
                persona1, persona2 = content.personas_used
                persona1_count = sum(1 for ex in all_exchanges if ex.speaker == persona1)
                persona2_count = sum(1 for ex in all_exchanges if ex.speaker == persona2)

                conversational_analysis["persona_balance"] = {
                    persona1: persona1_count,
                    persona2: persona2_count,
                    "balance_ratio": max(persona1_count, persona2_count) / max(min(persona1_count, persona2_count), 1)
                }

            # Technical concept coverage
            all_concepts = set()
            for exchange in all_exchanges:
                all_concepts.update(exchange.technical_concepts)
            conversational_analysis["technical_concept_coverage"] = len(all_concepts)

            # Intent variety
            conversational_analysis["intent_variety"] = set(ex.intent for ex in all_exchanges)

        # Conversational strengths
        if conversational_analysis["conversation_flow_score"] >= 0.8:
            conversational_analysis["conversational_strengths"].append("Excellent conversation flow")

        if conversational_analysis["technical_concept_coverage"] >= 5:
            conversational_analysis["conversational_strengths"].append("Rich technical content")

        if len(conversational_analysis["intent_variety"]) >= 4:
            conversational_analysis["conversational_strengths"].append("Good variety in conversation intents")

        # Conversational recommendations
        if conversational_analysis["persona_balance"].get("balance_ratio", 1) > 2:
            conversational_analysis["conversational_recommendations"].append(
                "Balance conversation more evenly between personas"
            )

        if conversational_analysis["technical_concept_coverage"] < 3:
            conversational_analysis["conversational_recommendations"].append(
                "Include more technical concepts in the discussion"
            )

        if "example" not in conversational_analysis["intent_variety"]:
            conversational_analysis["conversational_recommendations"].append(
                "Add practical examples to the conversation"
            )

        # Merge analyses
        combined_analysis = {**base_analysis, **conversational_analysis}

        return combined_analysis