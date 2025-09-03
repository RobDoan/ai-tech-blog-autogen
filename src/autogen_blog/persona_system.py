"""
Persona System for Conversational Blog Writer.

This module provides persona management, configuration, and consistency checking
for creating natural conversational blog content.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .multi_agent_models import BlogGenerationError


class PersonaError(BlogGenerationError):
    """Raised when persona operations fail."""

    pass


class PersonaConsistencyError(PersonaError):
    """Raised when persona consistency is violated."""

    pass


class CommunicationStyle(str, Enum):
    """Communication style options."""

    CASUAL = "casual"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    FORMAL = "formal"


class FormalityLevel(str, Enum):
    """Formality level options."""

    VERY_CASUAL = "very_casual"
    CASUAL = "casual"
    MODERATE = "moderate"
    PROFESSIONAL = "professional"
    FORMAL = "formal"


class TechnicalDepth(str, Enum):
    """Technical depth levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DialoguePace(str, Enum):
    """Dialogue pacing options."""

    QUICK = "quick"
    MODERATE = "moderate"
    DETAILED = "detailed"
    THOROUGH = "thorough"


@dataclass
class PersonalityTrait:
    """Represents a personality trait with intensity."""

    name: str
    intensity: float  # 0.0 to 1.0
    description: str
    behavioral_indicators: list[str] = field(default_factory=list)


class PersonaProfile(BaseModel):
    """Detailed persona profile for conversational agents."""

    name: str = Field(..., description="Persona name")
    role: str = Field(
        ..., description="Role in conversation (problem_presenter, solution_provider)"
    )
    background: str = Field(..., description="Professional background")
    expertise_areas: list[str] = Field(..., description="Areas of expertise")
    communication_style: CommunicationStyle = Field(CommunicationStyle.PROFESSIONAL)
    personality_traits: list[str] = Field(
        default_factory=list, description="Key personality traits"
    )
    typical_phrases: list[str] = Field(
        default_factory=list, description="Characteristic phrases"
    )
    technical_level: TechnicalDepth = Field(TechnicalDepth.INTERMEDIATE)
    conversation_goals: list[str] = Field(
        default_factory=list, description="What they aim to achieve"
    )
    preferred_topics: list[str] = Field(
        default_factory=list, description="Topics they gravitate toward"
    )
    speech_patterns: dict[str, Any] = Field(
        default_factory=dict, description="Speech pattern preferences"
    )

    @field_validator("personality_traits")
    def validate_traits(cls, v):
        if len(v) > 8:
            raise ValueError("Maximum 8 personality traits allowed")
        return v


class ConversationStyle(BaseModel):
    """Configuration for conversation style and flow."""

    formality_level: FormalityLevel = Field(FormalityLevel.MODERATE)
    technical_depth: TechnicalDepth = Field(TechnicalDepth.INTERMEDIATE)
    dialogue_pace: DialoguePace = Field(DialoguePace.MODERATE)
    max_exchange_length: int = Field(
        3, description="Max sentences per exchange", ge=1, le=10
    )
    include_questions: bool = Field(True, description="Include rhetorical questions")
    allow_interruptions: bool = Field(
        False, description="Allow conversation interruptions"
    )
    context_awareness: float = Field(
        0.8, description="How much context to maintain", ge=0.0, le=1.0
    )
    topic_transitions: bool = Field(True, description="Allow smooth topic transitions")


class PersonaConfig(BaseModel):
    """Complete configuration for conversational personas."""

    problem_presenter: PersonaProfile = Field(
        ..., description="Persona presenting problems"
    )
    solution_provider: PersonaProfile = Field(
        ..., description="Persona providing solutions"
    )
    conversation_style: ConversationStyle = Field(default_factory=ConversationStyle)
    domain_focus: str = Field(
        "software_development", description="Primary domain focus"
    )
    target_audience: str = Field(
        "developers", description="Target audience for conversation"
    )
    dialogue_objective: str = Field(
        "educational", description="Main objective of dialogue"
    )

    @field_validator("problem_presenter")
    def validate_problem_presenter(cls, v):
        if v.role != "problem_presenter":
            v.role = "problem_presenter"
        return v

    @field_validator("solution_provider")
    def validate_solution_provider(cls, v):
        if v.role != "solution_provider":
            v.role = "solution_provider"
        return v


@dataclass
class DialogueExchange:
    """Represents a single exchange in dialogue."""

    speaker: str  # persona name
    content: str
    intent: str  # question, explanation, example, etc.
    technical_concepts: list[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    confidence_level: float = 0.8


@dataclass
class DialogueSection:
    """Section of dialogue focused on a specific topic."""

    section_title: str
    exchanges: list[DialogueExchange] = field(default_factory=list)
    technical_focus: str = ""
    learning_objective: str = ""
    section_type: str = "discussion"  # introduction, discussion, example, conclusion


class ProblemPresenter:
    """Persona that presents development problems and challenges."""

    def __init__(self, profile: PersonaProfile):
        self.profile = profile
        self.logger = logging.getLogger(f"{__name__}.ProblemPresenter")

        # Default characteristics for problem presenter
        if not profile.conversation_goals:
            profile.conversation_goals = [
                "Present realistic development challenges",
                "Ask practical questions",
                "Share common pain points",
                "Seek actionable solutions",
            ]

        if not profile.typical_phrases:
            profile.typical_phrases = self._get_default_problem_phrases()

    def _get_default_problem_phrases(self) -> list[str]:
        """Get default phrases for problem presenter."""
        return [
            "I've been struggling with",
            "The challenge I'm facing is",
            "I'm not sure how to approach",
            "What would you recommend for",
            "I've tried several approaches but",
            "The issue seems to be",
            "I'm looking for a way to",
            "How do you handle situations where",
        ]

    def generate_problem_statement(self, topic: str, context: dict[str, Any]) -> str:
        """Generate a problem statement for the given topic."""
        problem_templates = [
            f"I've been working on {topic}, but I'm running into some challenges.",
            f"When dealing with {topic}, what's the best approach to handle complexity?",
            f"I'm trying to implement {topic} in my project, but I'm not sure where to start.",
            f"What are the common pitfalls when working with {topic}?",
        ]

        # Select template based on context or randomly
        template = problem_templates[hash(topic) % len(problem_templates)]
        return template

    def get_followup_questions(self, solution_content: str) -> list[str]:
        """Generate follow-up questions based on solution content."""
        questions = [
            "That makes sense. How would you handle edge cases?",
            "What about performance considerations?",
            "Are there any alternatives to this approach?",
            "How does this scale in larger applications?",
            "What are the potential drawbacks?",
        ]

        return questions[:2]  # Return 2 follow-up questions


class SolutionProvider:
    """Persona that provides technical solutions and explanations."""

    def __init__(self, profile: PersonaProfile):
        self.profile = profile
        self.logger = logging.getLogger(f"{__name__}.SolutionProvider")

        # Default characteristics for solution provider
        if not profile.conversation_goals:
            profile.conversation_goals = [
                "Provide practical solutions",
                "Explain technical concepts clearly",
                "Share best practices",
                "Offer multiple approaches",
            ]

        if not profile.typical_phrases:
            profile.typical_phrases = self._get_default_solution_phrases()

    def _get_default_solution_phrases(self) -> list[str]:
        """Get default phrases for solution provider."""
        return [
            "A good approach would be to",
            "Here's how I typically handle this",
            "The key insight here is",
            "Let me walk you through",
            "The best practice is to",
            "Consider this approach",
            "Here's what I recommend",
            "One effective strategy is",
        ]

    def generate_solution_response(self, problem: str, context: dict[str, Any]) -> str:
        """Generate a solution response to the given problem."""
        solution_templates = [
            "That's a common challenge. Here's how I typically approach it:",
            "I've dealt with similar issues. The key is to:",
            "Let me share a strategy that works well for this:",
            "Here's a practical solution that should help:",
        ]

        template = solution_templates[hash(problem) % len(solution_templates)]
        return template

    def provide_code_example_intro(self, concept: str) -> str:
        """Provide introduction for code examples."""
        intros = [
            f"Here's a practical example of {concept}:",
            f"Let me show you how to implement {concept}:",
            f"This is how I typically set up {concept}:",
            f"Here's a clean implementation of {concept}:",
        ]

        return intros[hash(concept) % len(intros)]


class PersonaManager:
    """Manages persona configurations and consistency checking."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PersonaManager")
        self.active_personas: dict[str, PersonaProfile] = {}
        self.conversation_history: list[DialogueExchange] = []
        self.consistency_rules: dict[str, list[str]] = {}

    def create_personas(
        self, config: PersonaConfig
    ) -> tuple[ProblemPresenter, SolutionProvider]:
        """Create persona instances from configuration."""
        try:
            problem_presenter = ProblemPresenter(config.problem_presenter)
            solution_provider = SolutionProvider(config.solution_provider)

            # Store active personas
            self.active_personas[config.problem_presenter.name] = (
                config.problem_presenter
            )
            self.active_personas[config.solution_provider.name] = (
                config.solution_provider
            )

            # Set up consistency rules
            self._setup_consistency_rules(config)

            self.logger.info(
                f"Created personas: {config.problem_presenter.name} and {config.solution_provider.name}"
            )
            return problem_presenter, solution_provider

        except Exception as e:
            self.logger.error(f"Failed to create personas: {e}")
            raise PersonaError(f"Persona creation failed: {e}") from e

    def _setup_consistency_rules(self, config: PersonaConfig) -> None:
        """Set up consistency rules for personas."""
        self.consistency_rules = {
            config.problem_presenter.name: [
                "should ask questions or present problems",
                "should show uncertainty or need for guidance",
                "should reference practical scenarios",
                f"should maintain {config.problem_presenter.communication_style.value} communication style",
            ],
            config.solution_provider.name: [
                "should provide solutions or explanations",
                "should demonstrate expertise and confidence",
                "should offer practical advice",
                f"should maintain {config.solution_provider.communication_style.value} communication style",
            ],
        }

    def validate_persona_consistency(self, dialogue: str) -> tuple[bool, list[str]]:
        """
        Validate that dialogue maintains persona consistency.

        Args:
            dialogue: The dialogue content to validate

        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []
        is_consistent = True

        try:
            # Parse dialogue into exchanges
            exchanges = self._parse_dialogue_exchanges(dialogue)

            for exchange in exchanges:
                speaker_issues = self._validate_speaker_consistency(exchange)
                if speaker_issues:
                    issues.extend(speaker_issues)
                    is_consistent = False

            # Check overall flow consistency
            flow_issues = self._validate_dialogue_flow(exchanges)
            if flow_issues:
                issues.extend(flow_issues)
                is_consistent = False

        except Exception as e:
            self.logger.error(f"Consistency validation failed: {e}")
            issues.append(f"Validation error: {e}")
            is_consistent = False

        return is_consistent, issues

    def _parse_dialogue_exchanges(self, dialogue: str) -> list[DialogueExchange]:
        """Parse dialogue text into structured exchanges."""
        exchanges = []

        # Look for speaker patterns (Name: content)
        pattern = r"^([A-Za-z\s]+):\s*(.+?)(?=^\w+:|$)"
        matches = re.finditer(pattern, dialogue, re.MULTILINE | re.DOTALL)

        for match in matches:
            speaker_name = match.group(1).strip()
            content = match.group(2).strip()

            if speaker_name in self.active_personas and content:
                exchange = DialogueExchange(
                    speaker=speaker_name,
                    content=content,
                    intent=self._infer_intent(content),
                    technical_concepts=self._extract_technical_concepts(content),
                )
                exchanges.append(exchange)

        return exchanges

    def _infer_intent(self, content: str) -> str:
        """Infer the intent of the dialogue content."""
        content_lower = content.lower()

        if "?" in content:
            return "question"
        elif any(word in content_lower for word in ["explain", "describe", "define"]):
            return "explanation"
        elif any(word in content_lower for word in ["example", "show", "demonstrate"]):
            return "example"
        elif any(word in content_lower for word in ["problem", "issue", "challenge"]):
            return "problem"
        elif any(word in content_lower for word in ["solution", "fix", "approach"]):
            return "solution"
        else:
            return "discussion"

    def _extract_technical_concepts(self, content: str) -> list[str]:
        """Extract technical concepts from content."""
        # This is a simplified version - could be enhanced with NLP
        technical_terms = re.findall(
            r"\b(?:API|SDK|JSON|XML|HTTP|HTTPS|REST|GraphQL|SQL|NoSQL|Docker|Kubernetes|React|Vue|Angular|Node\.js|Python|Java|JavaScript|TypeScript|Git|CI/CD|DevOps)\b",
            content,
            re.IGNORECASE,
        )

        return list({term.lower() for term in technical_terms})

    def _validate_speaker_consistency(self, exchange: DialogueExchange) -> list[str]:
        """Validate consistency for a single exchange."""
        issues = []

        persona = self.active_personas.get(exchange.speaker)
        if not persona:
            return [f"Unknown speaker: {exchange.speaker}"]

        self.consistency_rules.get(exchange.speaker, [])
        content_lower = exchange.content.lower()

        # Check role-specific consistency
        if persona.role == "problem_presenter":
            # Should present problems or ask questions
            if not any(
                word in content_lower
                for word in ["problem", "issue", "challenge", "?", "how", "what", "why"]
            ):
                issues.append(
                    f"{exchange.speaker} should present problems or ask questions"
                )

        elif persona.role == "solution_provider":
            # Should provide solutions or explanations
            if not any(
                word in content_lower
                for word in [
                    "solution",
                    "approach",
                    "method",
                    "way",
                    "use",
                    "implement",
                    "consider",
                ]
            ):
                issues.append(
                    f"{exchange.speaker} should provide solutions or explanations"
                )

        # Check communication style consistency
        style_issues = self._check_communication_style(exchange, persona)
        issues.extend(style_issues)

        return issues

    def _check_communication_style(
        self, exchange: DialogueExchange, persona: PersonaProfile
    ) -> list[str]:
        """Check if exchange matches persona's communication style."""
        issues = []
        content = exchange.content.lower()

        if persona.communication_style == CommunicationStyle.CASUAL:
            # Should use informal language
            formal_indicators = ["furthermore", "therefore", "consequently", "moreover"]
            if any(indicator in content for indicator in formal_indicators):
                issues.append(f"{persona.name} should use more casual language")

        elif persona.communication_style == CommunicationStyle.FORMAL:
            # Should avoid very casual language
            casual_indicators = ["yeah", "ok", "cool", "awesome", "kinda"]
            if any(indicator in content for indicator in casual_indicators):
                issues.append(f"{persona.name} should use more formal language")

        return issues

    def _validate_dialogue_flow(self, exchanges: list[DialogueExchange]) -> list[str]:
        """Validate the overall flow of dialogue."""
        issues = []

        if not exchanges:
            return ["No dialogue exchanges found"]

        # Check for reasonable alternation
        speakers = [ex.speaker for ex in exchanges]

        # Count consecutive exchanges by same speaker
        consecutive_count = 1
        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i - 1]:
                consecutive_count += 1
                if consecutive_count > 3:
                    issues.append(f"Too many consecutive exchanges by {speakers[i]}")
                    break
            else:
                consecutive_count = 1

        # Check for balanced participation
        speaker_counts = {}
        for speaker in speakers:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        if len(speaker_counts) > 1:
            counts = list(speaker_counts.values())
            max_count, min_count = max(counts), min(counts)
            if max_count > min_count * 3:  # One speaker dominates too much
                issues.append("Dialogue is imbalanced - one speaker dominates")

        return issues

    def suggest_improvements(self, dialogue: str) -> list[str]:
        """Suggest improvements for dialogue consistency."""
        suggestions = []

        is_consistent, issues = self.validate_persona_consistency(dialogue)

        if not is_consistent:
            suggestions.extend([f"Fix: {issue}" for issue in issues])

        # Additional suggestions
        exchanges = self._parse_dialogue_exchanges(dialogue)

        if len(exchanges) < 4:
            suggestions.append("Add more exchanges for richer dialogue")

        if not any("?" in ex.content for ex in exchanges):
            suggestions.append("Include some questions for more natural conversation")

        technical_concepts = set()
        for ex in exchanges:
            technical_concepts.update(ex.technical_concepts)

        if len(technical_concepts) < 2:
            suggestions.append("Include more technical concepts for educational value")

        return suggestions

    def get_persona_voice_guide(self, persona_name: str) -> dict[str, Any]:
        """Get voice and style guide for a persona."""
        persona = self.active_personas.get(persona_name)
        if not persona:
            return {}

        return {
            "name": persona.name,
            "role": persona.role,
            "communication_style": persona.communication_style.value,
            "typical_phrases": persona.typical_phrases,
            "personality_traits": persona.personality_traits,
            "conversation_goals": persona.conversation_goals,
            "technical_level": persona.technical_level.value,
            "do_use": self._get_recommended_language(persona),
            "avoid_using": self._get_language_to_avoid(persona),
        }

    def _get_recommended_language(self, persona: PersonaProfile) -> list[str]:
        """Get recommended language patterns for persona."""
        recommendations = []

        if persona.role == "problem_presenter":
            recommendations.extend(
                [
                    "Questions and uncertainty expressions",
                    "Problem-focused language",
                    "Practical scenario descriptions",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Solution-oriented language",
                    "Confident explanations",
                    "Best practice recommendations",
                ]
            )

        if persona.communication_style == CommunicationStyle.CASUAL:
            recommendations.append("Informal, conversational tone")
        elif persona.communication_style == CommunicationStyle.FORMAL:
            recommendations.append("Professional, structured language")

        return recommendations

    def _get_language_to_avoid(self, persona: PersonaProfile) -> list[str]:
        """Get language patterns to avoid for persona."""
        avoid = []

        if persona.role == "problem_presenter":
            avoid.extend(
                [
                    "Overly confident assertions",
                    "Detailed technical explanations without context",
                ]
            )
        else:
            avoid.extend(
                [
                    "Uncertain or questioning tone when providing solutions",
                    "Vague or non-specific advice",
                ]
            )

        if persona.communication_style == CommunicationStyle.CASUAL:
            avoid.append("Overly formal or academic language")
        elif persona.communication_style == CommunicationStyle.FORMAL:
            avoid.append("Very casual expressions or slang")

        return avoid


def create_default_persona_config() -> PersonaConfig:
    """Create a default persona configuration for developer conversations."""
    problem_presenter = PersonaProfile(
        name="Alex",
        role="problem_presenter",
        background="Full-stack developer with 3 years experience",
        expertise_areas=["web development", "JavaScript", "Python"],
        communication_style=CommunicationStyle.PROFESSIONAL,
        personality_traits=["curious", "practical", "detail-oriented"],
        technical_level=TechnicalDepth.INTERMEDIATE,
        conversation_goals=[
            "Present realistic development challenges",
            "Ask practical questions",
            "Seek actionable solutions",
        ],
        preferred_topics=["best practices", "troubleshooting", "tool recommendations"],
    )

    solution_provider = PersonaProfile(
        name="Jordan",
        role="solution_provider",
        background="Senior software engineer and tech lead with 8 years experience",
        expertise_areas=["software architecture", "DevOps", "mentoring"],
        communication_style=CommunicationStyle.PROFESSIONAL,
        personality_traits=["knowledgeable", "helpful", "systematic"],
        technical_level=TechnicalDepth.ADVANCED,
        conversation_goals=[
            "Provide practical solutions",
            "Explain technical concepts clearly",
            "Share industry best practices",
        ],
        preferred_topics=[
            "architecture patterns",
            "performance optimization",
            "team practices",
        ],
    )

    conversation_style = ConversationStyle(
        formality_level=FormalityLevel.PROFESSIONAL,
        technical_depth=TechnicalDepth.INTERMEDIATE,
        dialogue_pace=DialoguePace.MODERATE,
    )

    return PersonaConfig(
        problem_presenter=problem_presenter,
        solution_provider=solution_provider,
        conversation_style=conversation_style,
        domain_focus="software_development",
        target_audience="developers",
    )


def load_persona_config_from_file(config_path: Path) -> PersonaConfig:
    """Load persona configuration from JSON file."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

        return PersonaConfig(**config_data)

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load persona config: {e}")
        raise PersonaError(f"Failed to load persona configuration: {e}") from e


def save_persona_config_to_file(config: PersonaConfig, config_path: Path) -> None:
    """Save persona configuration to JSON file."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.dict(), f, indent=2, default=str)

        logging.getLogger(__name__).info(f"Saved persona config to {config_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save persona config: {e}")
        raise PersonaError(f"Failed to save persona configuration: {e}") from e
