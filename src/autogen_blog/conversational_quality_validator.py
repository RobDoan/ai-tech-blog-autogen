"""
Conversational Content Quality Validation.

This module provides quality validation specifically for conversational blog content,
including dialogue naturalness scoring, technical accuracy validation, and persona
consistency measurement.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from .conversational_writer_agent import ConversationalBlogContent
from .information_synthesizer import SynthesizedKnowledge
from .persona_system import (
    DialogueExchange,
    DialogueSection,
    PersonaManager,
    PersonaProfile,
)


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a quality validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    location: str = ""
    suggestion: str = ""
    confidence: float = 1.0


@dataclass
class DialogueNaturalnessScore:
    """Dialogue naturalness assessment."""
    overall_score: float = 0.0  # 0.0 to 1.0
    conversation_flow_score: float = 0.0
    persona_consistency_score: float = 0.0
    technical_integration_score: float = 0.0
    engagement_score: float = 0.0
    issues: list[ValidationIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


@dataclass
class TechnicalAccuracyScore:
    """Technical accuracy assessment."""
    overall_score: float = 0.0  # 0.0 to 1.0
    concept_accuracy_score: float = 0.0
    code_quality_score: float = 0.0
    research_alignment_score: float = 0.0
    depth_appropriateness_score: float = 0.0
    issues: list[ValidationIssue] = field(default_factory=list)
    validated_concepts: list[str] = field(default_factory=list)


@dataclass
class PersonaConsistencyScore:
    """Persona consistency assessment."""
    overall_score: float = 0.0  # 0.0 to 1.0
    voice_consistency_score: float = 0.0
    role_adherence_score: float = 0.0
    expertise_level_score: float = 0.0
    communication_style_score: float = 0.0
    issues: list[ValidationIssue] = field(default_factory=list)
    persona_violations: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ConversationalQualityReport:
    """Comprehensive quality assessment report."""
    overall_quality_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.FAIR
    dialogue_naturalness: DialogueNaturalnessScore = field(default_factory=DialogueNaturalnessScore)
    technical_accuracy: TechnicalAccuracyScore = field(default_factory=TechnicalAccuracyScore)
    persona_consistency: PersonaConsistencyScore = field(default_factory=PersonaConsistencyScore)

    # Aggregated results
    all_issues: list[ValidationIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validation_timestamp: str = ""

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report."""
        self.all_issues.append(issue)

        # Also add to specific score objects based on category
        if issue.category in ['flow', 'engagement', 'naturalness']:
            self.dialogue_naturalness.issues.append(issue)
        elif issue.category in ['technical', 'accuracy', 'concepts']:
            self.technical_accuracy.issues.append(issue)
        elif issue.category in ['persona', 'consistency', 'voice']:
            self.persona_consistency.issues.append(issue)

    def get_quality_level(self) -> QualityLevel:
        """Determine quality level based on overall score."""
        if self.overall_quality_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.overall_quality_score >= 0.7:
            return QualityLevel.GOOD
        elif self.overall_quality_score >= 0.5:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR


class DialogueNaturalnessValidator:
    """Validates dialogue naturalness and flow."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate_naturalness(
        self,
        dialogue_sections: list[DialogueSection],
        personas: tuple[PersonaProfile, PersonaProfile] | None = None
    ) -> DialogueNaturalnessScore:
        """Validate dialogue naturalness and flow."""
        score = DialogueNaturalnessScore()

        if not dialogue_sections:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="naturalness",
                message="No dialogue sections found",
                suggestion="Ensure dialogue content is properly structured"
            ))
            return score

        all_exchanges = []
        for section in dialogue_sections:
            all_exchanges.extend(section.exchanges)

        if not all_exchanges:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="naturalness",
                message="No dialogue exchanges found",
                suggestion="Add conversational exchanges between personas"
            ))
            return score

        # Validate conversation flow
        score.conversation_flow_score = await self._validate_conversation_flow(all_exchanges, score)

        # Validate engagement level
        score.engagement_score = await self._validate_engagement_level(all_exchanges, score)

        # Validate technical integration
        score.technical_integration_score = await self._validate_technical_integration(all_exchanges, score)

        # Calculate overall naturalness score
        score.overall_score = (
            score.conversation_flow_score * 0.4 +
            score.engagement_score * 0.3 +
            score.technical_integration_score * 0.3
        )

        # Add strengths based on high scores
        if score.conversation_flow_score >= 0.8:
            score.strengths.append("Excellent conversation flow and transitions")

        if score.engagement_score >= 0.8:
            score.strengths.append("High engagement with questions and interactions")

        if score.technical_integration_score >= 0.8:
            score.strengths.append("Natural integration of technical concepts")

        return score

    async def _validate_conversation_flow(self, exchanges: list[DialogueExchange], score: DialogueNaturalnessScore) -> float:
        """Validate conversation flow and transitions."""
        flow_score = 1.0

        if len(exchanges) < 4:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow",
                message="Very short conversation - consider adding more exchanges",
                suggestion="Aim for at least 6-8 exchanges for natural flow"
            ))
            flow_score -= 0.2

        # Check speaker alternation
        consecutive_speakers = 1
        current_speaker = None
        max_consecutive = 1

        for exchange in exchanges:
            if exchange.speaker == current_speaker:
                consecutive_speakers += 1
                max_consecutive = max(max_consecutive, consecutive_speakers)
            else:
                consecutive_speakers = 1
                current_speaker = exchange.speaker

        if max_consecutive > 3:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow",
                message=f"One speaker has {max_consecutive} consecutive exchanges",
                suggestion="Balance the conversation more evenly between speakers"
            ))
            flow_score -= 0.3

        # Check for natural transitions
        transition_quality = self._assess_transition_quality(exchanges)
        flow_score *= transition_quality

        if transition_quality < 0.6:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="flow",
                message="Some transitions between topics seem abrupt",
                suggestion="Add smoother transitions between different topics"
            ))

        return max(0.0, flow_score)

    def _assess_transition_quality(self, exchanges: list[DialogueExchange]) -> float:
        """Assess quality of transitions between exchanges."""
        if len(exchanges) < 2:
            return 1.0

        good_transitions = 0
        total_transitions = len(exchanges) - 1

        transition_phrases = [
            'that brings up', 'speaking of', 'related to', 'building on',
            'exactly', 'right', 'good point', 'interesting', 'let me',
            'what about', 'how about', 'another', 'also'
        ]

        for i in range(1, len(exchanges)):
            current_content = exchanges[i].content.lower()

            # Check if exchange references previous content or uses transition phrases
            has_transition = any(phrase in current_content for phrase in transition_phrases)

            # Check if it's a natural response (questions followed by answers, etc.)
            prev_has_question = '?' in exchanges[i-1].content
            current_provides_answer = any(word in current_content for word in [
                'yes', 'no', 'well', 'actually', 'here', 'let me', 'the answer', 'you can'
            ])

            if has_transition or (prev_has_question and current_provides_answer):
                good_transitions += 1

        return good_transitions / total_transitions if total_transitions > 0 else 1.0

    async def _validate_engagement_level(self, exchanges: list[DialogueExchange], score: DialogueNaturalnessScore) -> float:
        """Validate engagement level of the conversation."""
        engagement_score = 1.0

        # Count questions
        questions = [ex for ex in exchanges if '?' in ex.content]
        question_ratio = len(questions) / len(exchanges)

        if question_ratio == 0:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="engagement",
                message="No questions found in conversation",
                suggestion="Add questions to make dialogue more interactive"
            ))
            engagement_score -= 0.4
        elif question_ratio > 0.7:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="engagement",
                message="Very high question ratio - might feel like an interview",
                suggestion="Balance questions with explanations and statements"
            ))
            engagement_score -= 0.2

        # Check for examples and illustrations
        example_indicators = ['example', 'instance', 'like this', 'such as', 'for instance']
        examples_count = sum(1 for ex in exchanges
                           if any(indicator in ex.content.lower() for indicator in example_indicators))

        if examples_count == 0:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="engagement",
                message="No examples or illustrations mentioned",
                suggestion="Include practical examples to improve engagement"
            ))
            engagement_score -= 0.2

        # Check for emotional/conversational markers
        conversational_markers = ['interesting', 'great', 'exactly', 'right', 'cool', 'nice', 'wow']
        marker_count = sum(1 for ex in exchanges
                          if any(marker in ex.content.lower() for marker in conversational_markers))

        if marker_count == 0:
            engagement_score -= 0.1

        return max(0.0, engagement_score)

    async def _validate_technical_integration(self, exchanges: list[DialogueExchange], score: DialogueNaturalnessScore) -> float:
        """Validate how naturally technical concepts are integrated."""
        integration_score = 1.0

        technical_exchanges = [ex for ex in exchanges if ex.technical_concepts]

        if not technical_exchanges:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message="No technical concepts identified in conversation",
                suggestion="Ensure technical terminology is properly tagged"
            ))
            return 0.3

        # Check if technical terms are explained or contextualized
        explained_count = 0
        for exchange in technical_exchanges:
            content_lower = exchange.content.lower()

            # Look for explanation patterns
            explanation_patterns = [
                'is a', 'means', 'refers to', 'basically', 'essentially',
                'in other words', 'think of it as', 'like'
            ]

            has_explanation = any(pattern in content_lower for pattern in explanation_patterns)
            if has_explanation:
                explained_count += 1

        explanation_ratio = explained_count / len(technical_exchanges)
        if explanation_ratio < 0.3:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="technical",
                message="Technical concepts could use more explanation",
                suggestion="Add brief explanations for technical terms"
            ))
            integration_score -= 0.2

        return max(0.0, integration_score)


class TechnicalAccuracyValidator:
    """Validates technical accuracy of conversational content."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate_accuracy(
        self,
        content: ConversationalBlogContent,
        research_knowledge: SynthesizedKnowledge | None = None
    ) -> TechnicalAccuracyScore:
        """Validate technical accuracy of conversational content."""
        score = TechnicalAccuracyScore()

        # Extract technical concepts from content
        all_exchanges = []
        for section in content.dialogue_sections:
            all_exchanges.extend(section.exchanges)

        # Validate concept accuracy
        score.concept_accuracy_score = await self._validate_concept_accuracy(all_exchanges, score)

        # Validate code quality
        score.code_quality_score = await self._validate_code_quality(content.code_blocks, score)

        # Validate research alignment if available
        if research_knowledge:
            score.research_alignment_score = await self._validate_research_alignment(
                all_exchanges, research_knowledge, score
            )
        else:
            score.research_alignment_score = 0.7  # Default when no research available

        # Validate depth appropriateness
        score.depth_appropriateness_score = await self._validate_depth_appropriateness(all_exchanges, score)

        # Calculate overall accuracy score
        weights = [0.3, 0.2, 0.3, 0.2]  # concept, code, research, depth
        scores = [
            score.concept_accuracy_score,
            score.code_quality_score,
            score.research_alignment_score,
            score.depth_appropriateness_score
        ]

        score.overall_score = sum(w * s for w, s in zip(weights, scores, strict=False))

        return score

    async def _validate_concept_accuracy(self, exchanges: list[DialogueExchange], score: TechnicalAccuracyScore) -> float:
        """Validate accuracy of technical concepts mentioned."""
        # This is a simplified validation - in a real implementation,
        # you might use a knowledge base or external APIs

        all_concepts = set()
        for exchange in exchanges:
            all_concepts.update(exchange.technical_concepts)

        score.validated_concepts = list(all_concepts)

        # Basic heuristic validation
        concept_score = 1.0

        # Check for common misspellings or incorrect usage
        problematic_patterns = [
            (r'\bjavascript\b.*\bmultithreading\b', "JavaScript doesn't have traditional multithreading"),
            (r'\bpython\b.*\bcompiled\b', "Python is interpreted, not compiled"),
        ]

        all_content = ' '.join(ex.content.lower() for ex in exchanges)

        for pattern, issue_msg in problematic_patterns:
            if re.search(pattern, all_content):
                score.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="accuracy",
                    message=f"Potential technical inaccuracy: {issue_msg}",
                    suggestion="Verify technical claims for accuracy"
                ))
                concept_score -= 0.1

        return max(0.0, concept_score)

    async def _validate_code_quality(self, code_blocks, score: TechnicalAccuracyScore) -> float:
        """Validate quality of code examples."""
        if not code_blocks:
            return 0.8  # Neutral score when no code

        code_score = 1.0

        for i, code_block in enumerate(code_blocks):
            # Check for common code issues
            if not code_block.code.strip():
                score.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="code",
                    message=f"Empty code block {i+1}",
                    suggestion="Ensure all code blocks contain meaningful code"
                ))
                code_score -= 0.2
                continue

            # Check for syntax errors (basic)
            if code_block.language.lower() == 'python':
                if not self._basic_python_syntax_check(code_block.code):
                    score.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="code",
                        message=f"Potential Python syntax issues in code block {i+1}",
                        suggestion="Review Python code for syntax errors"
                    ))
                    code_score -= 0.1

        return max(0.0, code_score)

    def _basic_python_syntax_check(self, code: str) -> bool:
        """Basic Python syntax validation."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
        except:
            return True  # Other errors don't indicate syntax problems

    async def _validate_research_alignment(
        self,
        exchanges: list[DialogueExchange],
        research_knowledge: SynthesizedKnowledge,
        score: TechnicalAccuracyScore
    ) -> float:
        """Validate alignment with research knowledge."""
        alignment_score = 1.0

        # Check if key insights from research are reflected
        research_concepts = set()
        for insight in research_knowledge.original_knowledge.insights:
            # Extract technical terms from insights
            words = re.findall(r'\b[A-Za-z]{3,}\b', insight.content.lower())
            research_concepts.update(words)

        mentioned_concepts = set()
        for exchange in exchanges:
            content_words = re.findall(r'\b[A-Za-z]{3,}\b', exchange.content.lower())
            mentioned_concepts.update(content_words)

        # Calculate overlap
        common_concepts = research_concepts.intersection(mentioned_concepts)
        if research_concepts:
            overlap_ratio = len(common_concepts) / len(research_concepts)

            if overlap_ratio < 0.2:
                score.issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="research",
                    message="Limited use of research concepts in conversation",
                    suggestion="Incorporate more insights from research materials"
                ))
                alignment_score -= 0.3

        return max(0.0, alignment_score)

    async def _validate_depth_appropriateness(self, exchanges: list[DialogueExchange], score: TechnicalAccuracyScore) -> float:
        """Validate that technical depth is appropriate."""
        # This is a simplified heuristic-based approach
        depth_score = 1.0

        # Count advanced vs basic terminology
        advanced_terms = ['architecture', 'optimization', 'scalability', 'performance', 'algorithm']
        basic_terms = ['variable', 'function', 'loop', 'if', 'print']

        all_content = ' '.join(ex.content.lower() for ex in exchanges)

        advanced_count = sum(1 for term in advanced_terms if term in all_content)
        basic_count = sum(1 for term in basic_terms if term in all_content)

        # Check balance
        if advanced_count > 5 and basic_count == 0:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="technical",
                message="Content may be too advanced without foundational explanation",
                suggestion="Include some basic concepts for context"
            ))
            depth_score -= 0.2

        return max(0.0, depth_score)


class PersonaConsistencyValidator:
    """Validates persona consistency throughout conversation."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.persona_manager = PersonaManager()

    async def validate_consistency(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile] | None = None
    ) -> PersonaConsistencyScore:
        """Validate persona consistency in conversational content."""
        score = PersonaConsistencyScore()

        if not persona_profiles:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="persona",
                message="No persona profiles provided for validation",
                suggestion="Provide persona profiles for consistency checking"
            ))
            return score

        # Use persona manager for validation
        is_consistent, consistency_issues = self.persona_manager.validate_persona_consistency(content.content)

        # Convert consistency issues to validation issues
        for issue in consistency_issues:
            score.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="consistency",
                message=issue,
                suggestion="Review persona adherence in dialogue"
            ))

        # Calculate component scores
        score.voice_consistency_score = await self._validate_voice_consistency(content, persona_profiles, score)
        score.role_adherence_score = await self._validate_role_adherence(content, persona_profiles, score)
        score.expertise_level_score = await self._validate_expertise_level(content, persona_profiles, score)
        score.communication_style_score = await self._validate_communication_style(content, persona_profiles, score)

        # Calculate overall consistency score
        score.overall_score = (
            score.voice_consistency_score * 0.3 +
            score.role_adherence_score * 0.3 +
            score.expertise_level_score * 0.2 +
            score.communication_style_score * 0.2
        )

        return score

    async def _validate_voice_consistency(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile],
        score: PersonaConsistencyScore
    ) -> float:
        """Validate consistency of persona voices."""
        # This is a simplified implementation
        voice_score = 0.8  # Default score

        # Check if typical phrases are used appropriately
        problem_presenter, solution_provider = persona_profiles

        for section in content.dialogue_sections:
            for exchange in section.exchanges:
                if exchange.speaker == problem_presenter.name:
                    # Should use problem-oriented language
                    if not any(word in exchange.content.lower()
                             for word in ['how', 'what', 'why', 'challenge', 'problem', 'issue']):
                        voice_score -= 0.05

                elif exchange.speaker == solution_provider.name:
                    # Should use solution-oriented language
                    if not any(word in exchange.content.lower()
                             for word in ['solution', 'approach', 'way', 'method', 'can', 'use']):
                        voice_score -= 0.05

        return max(0.0, voice_score)

    async def _validate_role_adherence(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile],
        score: PersonaConsistencyScore
    ) -> float:
        """Validate adherence to persona roles."""
        return 0.8  # Simplified - would need more sophisticated analysis

    async def _validate_expertise_level(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile],
        score: PersonaConsistencyScore
    ) -> float:
        """Validate that personas maintain appropriate expertise levels."""
        return 0.8  # Simplified - would analyze technical depth per persona

    async def _validate_communication_style(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile],
        score: PersonaConsistencyScore
    ) -> float:
        """Validate communication style consistency."""
        return 0.8  # Simplified - would analyze formality, tone, etc.


class ConversationalQualityValidator:
    """Main quality validator for conversational blog content."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dialogue_validator = DialogueNaturalnessValidator()
        self.technical_validator = TechnicalAccuracyValidator()
        self.persona_validator = PersonaConsistencyValidator()

    async def validate_quality(
        self,
        content: ConversationalBlogContent,
        persona_profiles: tuple[PersonaProfile, PersonaProfile] | None = None,
        research_knowledge: SynthesizedKnowledge | None = None
    ) -> ConversationalQualityReport:
        """
        Perform comprehensive quality validation of conversational content.

        Args:
            content: Conversational blog content to validate
            persona_profiles: Persona profiles used in conversation
            research_knowledge: Research knowledge base used

        Returns:
            Comprehensive quality report
        """
        from datetime import datetime

        self.logger.info("Starting conversational quality validation")

        report = ConversationalQualityReport()
        report.validation_timestamp = datetime.now().isoformat()

        try:
            # Validate dialogue naturalness
            report.dialogue_naturalness = await self.dialogue_validator.validate_naturalness(
                content.dialogue_sections, persona_profiles
            )

            # Validate technical accuracy
            report.technical_accuracy = await self.technical_validator.validate_accuracy(
                content, research_knowledge
            )

            # Validate persona consistency
            report.persona_consistency = await self.persona_validator.validate_consistency(
                content, persona_profiles
            )

            # Calculate overall quality score
            report.overall_quality_score = (
                report.dialogue_naturalness.overall_score * 0.4 +
                report.technical_accuracy.overall_score * 0.3 +
                report.persona_consistency.overall_score * 0.3
            )

            # Determine quality level
            report.quality_level = report.get_quality_level()

            # Aggregate all issues
            report.all_issues = (
                report.dialogue_naturalness.issues +
                report.technical_accuracy.issues +
                report.persona_consistency.issues
            )

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            self.logger.info(f"Quality validation completed. Score: {report.overall_quality_score:.2f}")

        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="validation",
                message=f"Validation process failed: {e}",
                suggestion="Review content structure and retry validation"
            ))

        return report

    def _generate_recommendations(self, report: ConversationalQualityReport) -> list[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Dialogue naturalness recommendations
        if report.dialogue_naturalness.overall_score < 0.7:
            recommendations.append("Improve conversation flow with better transitions and speaker balance")

        if report.dialogue_naturalness.engagement_score < 0.6:
            recommendations.append("Add more questions and interactive elements to increase engagement")

        # Technical accuracy recommendations
        if report.technical_accuracy.overall_score < 0.7:
            recommendations.append("Review technical content for accuracy and appropriate depth")

        if not report.technical_accuracy.validated_concepts:
            recommendations.append("Include more technical concepts and ensure they are properly explained")

        # Persona consistency recommendations
        if report.persona_consistency.overall_score < 0.7:
            recommendations.append("Improve persona consistency by better adhering to character profiles")

        # Overall recommendations
        critical_issues = [issue for issue in report.all_issues
                         if issue.severity == ValidationSeverity.CRITICAL]

        if critical_issues:
            recommendations.append("Address critical issues before publishing content")

        if report.overall_quality_score < 0.5:
            recommendations.append("Consider significant revision - quality score indicates major issues")

        return recommendations[:8]  # Limit to top 8 recommendations
