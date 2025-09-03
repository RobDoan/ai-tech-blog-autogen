"""
Information Synthesis Components for Conversational Blog Writer.

This module provides advanced information extraction and synthesis capabilities
for generating actionable knowledge from research materials.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .multi_agent_models import BlogGenerationError
from .research_processor import Insight, KnowledgeBase, ResearchFile


class InformationSynthesisError(BlogGenerationError):
    """Raised when information synthesis fails."""

    pass


@dataclass
class TechnicalConcept:
    """Represents a technical concept with context."""

    name: str
    frequency: int
    contexts: list[str]
    related_concepts: set[str]
    importance_score: float


@dataclass
class CodeExample:
    """Represents a code example with metadata."""

    language: str
    code_snippet: str
    explanation: str
    source_file: str
    technical_concepts: list[str]
    complexity_level: str  # beginner, intermediate, advanced


class TechnicalDetail(BaseModel):
    """Detailed technical information extracted from research."""

    concept: str = Field(..., description="Technical concept name")
    description: str = Field(..., description="Detailed description")
    use_cases: list[str] = Field(
        default_factory=list, description="Practical use cases"
    )
    code_examples: list[str] = Field(
        default_factory=list, description="Related code snippets"
    )
    related_technologies: list[str] = Field(
        default_factory=list, description="Related technologies"
    )
    difficulty_level: str = Field("intermediate", description="Difficulty level")
    importance_score: float = Field(0.5, description="Importance score", ge=0.0, le=1.0)
    confidence_score: float = Field(0.7, description="Confidence score", ge=0.0, le=1.0)


class Reference(BaseModel):
    """Reference to source material."""

    title: str = Field(..., description="Reference title")
    source_file: str = Field(..., description="Source file path")
    content_preview: str = Field(..., description="Preview of referenced content")
    relevance_score: float = Field(0.5, description="Relevance score", ge=0.0, le=1.0)
    reference_type: str = Field("general", description="Type of reference")


class SynthesizedKnowledge(BaseModel):
    """Enhanced knowledge base with synthesized information."""

    original_knowledge: KnowledgeBase = Field(
        ..., description="Original knowledge base"
    )
    technical_details: list[TechnicalDetail] = Field(default_factory=list)
    code_examples: list[CodeExample] = Field(default_factory=list)
    key_themes: list[str] = Field(
        default_factory=list, description="Main themes identified"
    )
    problem_solution_pairs: list[dict[str, Any]] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)
    synthesis_confidence: float = Field(0.0, description="Overall synthesis confidence")

    class Config:
        arbitrary_types_allowed = True


class InformationExtractor:
    """Extract detailed information from research content."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Technical concept patterns
        self.tech_patterns = {
            "frameworks": r"\b(React|Vue|Angular|Django|Flask|Express|Spring|Laravel|Rails)\b",
            "languages": r"\b(Python|JavaScript|TypeScript|Java|C\+\+|C#|Go|Rust|Swift|Kotlin)\b",
            "databases": r"\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB)\b",
            "cloud_services": r"\b(AWS|Azure|GCP|Docker|Kubernetes|Lambda|S3|EC2)\b",
            "tools": r"\b(Git|Jenkins|CircleCI|Terraform|Ansible|Webpack|Vite|npm|pip)\b",
            "concepts": r"\b(API|REST|GraphQL|Microservices|DevOps|CI/CD|TDD|BDD|SOLID)\b",
        }

        # Problem-solution indicators
        self.problem_indicators = [
            "issue",
            "problem",
            "challenge",
            "difficulty",
            "error",
            "bug",
            "limitation",
            "drawback",
            "bottleneck",
            "pain point",
        ]

        self.solution_indicators = [
            "solution",
            "fix",
            "resolve",
            "approach",
            "method",
            "technique",
            "workaround",
            "implementation",
            "best practice",
            "recommendation",
        ]

    async def extract_technical_concepts(
        self, research_files: list[ResearchFile]
    ) -> list[TechnicalConcept]:
        """Extract and analyze technical concepts from research files."""
        concept_data = defaultdict(
            lambda: {"frequency": 0, "contexts": [], "files": set(), "related": set()}
        )

        for file_data in research_files:
            content = file_data.content.lower()

            # Extract technical concepts using patterns
            for category, pattern in self.tech_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)

                for match in matches:
                    concept_name = match.lower()
                    concept_data[concept_name]["frequency"] += 1
                    concept_data[concept_name]["files"].add(str(file_data.path))

                    # Extract context around the concept
                    context_pattern = rf".{{0,50}}{re.escape(match)}.{{0,50}}"
                    contexts = re.findall(
                        context_pattern, content, re.IGNORECASE | re.DOTALL
                    )
                    concept_data[concept_name]["contexts"].extend(
                        contexts[:3]
                    )  # Limit contexts

        # Convert to TechnicalConcept objects
        technical_concepts = []
        for concept, data in concept_data.items():
            if data["frequency"] >= 2:  # Only concepts mentioned multiple times
                importance = min(1.0, data["frequency"] / 10.0)

                tech_concept = TechnicalConcept(
                    name=concept,
                    frequency=data["frequency"],
                    contexts=data["contexts"][:5],  # Top 5 contexts
                    related_concepts=set(),  # Will be populated later
                    importance_score=importance,
                )
                technical_concepts.append(tech_concept)

        # Sort by importance
        technical_concepts.sort(key=lambda x: x.importance_score, reverse=True)

        self.logger.info(f"Extracted {len(technical_concepts)} technical concepts")
        return technical_concepts

    async def extract_problem_solution_pairs(
        self, research_files: list[ResearchFile]
    ) -> list[dict[str, Any]]:
        """Extract problem-solution pairs from research content."""
        pairs = []

        for file_data in research_files:
            content = file_data.content

            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            # Look for problem-solution patterns
            for i, paragraph in enumerate(paragraphs):
                paragraph_lower = paragraph.lower()

                # Check if paragraph contains problem indicators
                has_problem = any(
                    indicator in paragraph_lower
                    for indicator in self.problem_indicators
                )

                if has_problem:
                    # Look for solution in the same or next few paragraphs
                    solution_candidates = paragraphs[
                        i : i + 3
                    ]  # Current + next 2 paragraphs

                    for candidate in solution_candidates:
                        candidate_lower = candidate.lower()
                        has_solution = any(
                            indicator in candidate_lower
                            for indicator in self.solution_indicators
                        )

                        if has_solution and candidate != paragraph:
                            pairs.append(
                                {
                                    "problem": paragraph[
                                        :300
                                    ],  # Truncate for readability
                                    "solution": candidate[:300],
                                    "source_file": str(file_data.path),
                                    "confidence": 0.7,
                                }
                            )
                            break

        self.logger.info(f"Extracted {len(pairs)} problem-solution pairs")
        return pairs[:10]  # Limit to top 10 pairs

    async def extract_code_examples(
        self, research_files: list[ResearchFile]
    ) -> list[CodeExample]:
        """Extract and analyze code examples from research files."""
        code_examples = []

        for file_data in research_files:
            if file_data.file_type == "markdown":
                # Extract code blocks from markdown
                code_blocks = self._extract_markdown_code_blocks(
                    file_data.content, str(file_data.path)
                )
                code_examples.extend(code_blocks)

        self.logger.info(f"Extracted {len(code_examples)} code examples")
        return code_examples

    def _extract_markdown_code_blocks(
        self, content: str, source_file: str
    ) -> list[CodeExample]:
        """Extract code blocks from markdown content."""
        code_examples = []

        # Find code blocks with language specification
        pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()

            if not code or len(code) < 10:  # Skip very short snippets
                continue

            # Try to find explanation near the code block
            explanation = self._find_code_explanation(
                content, match.start(), match.end()
            )

            # Determine complexity based on code characteristics
            complexity = self._assess_code_complexity(code, language)

            # Extract technical concepts from code
            tech_concepts = self._extract_concepts_from_code(code, language)

            code_example = CodeExample(
                language=language,
                code_snippet=code,
                explanation=explanation,
                source_file=source_file,
                technical_concepts=tech_concepts,
                complexity_level=complexity,
            )
            code_examples.append(code_example)

        return code_examples

    def _find_code_explanation(
        self, content: str, code_start: int, code_end: int
    ) -> str:
        """Find explanation text near a code block."""
        # Look for text before the code block
        before_text = content[:code_start].strip()
        after_text = content[code_end:].strip()

        # Get the last paragraph before code
        before_paragraphs = before_text.split("\n\n")
        if before_paragraphs:
            before_candidate = before_paragraphs[-1].strip()
            if len(before_candidate) > 20 and not before_candidate.startswith("#"):
                return before_candidate

        # Get the first paragraph after code
        after_paragraphs = after_text.split("\n\n")
        if after_paragraphs:
            after_candidate = after_paragraphs[0].strip()
            if len(after_candidate) > 20 and not after_candidate.startswith("#"):
                return after_candidate

        return "Code example from research materials"

    def _assess_code_complexity(self, code: str, language: str) -> str:
        """Assess the complexity level of code."""
        complexity_indicators = {
            "beginner": ["print", "console.log", "hello", "simple", "basic"],
            "advanced": [
                "async",
                "await",
                "class",
                "interface",
                "decorator",
                "lambda",
                "generator",
            ],
        }

        code_lower = code.lower()

        # Count advanced patterns
        advanced_score = sum(
            1 for pattern in complexity_indicators["advanced"] if pattern in code_lower
        )

        # Consider length and structure
        lines = len(code.split("\n"))

        if advanced_score >= 2 or lines > 20:
            return "advanced"
        elif advanced_score >= 1 or lines > 10:
            return "intermediate"
        else:
            return "beginner"

    def _extract_concepts_from_code(self, code: str, language: str) -> list[str]:
        """Extract technical concepts from code content."""
        concepts = [language]

        # Language-specific concept extraction
        if language.lower() in ["python"]:
            concepts.extend(self._extract_python_concepts(code))
        elif language.lower() in ["javascript", "typescript"]:
            concepts.extend(self._extract_js_concepts(code))

        return list(set(concepts))

    def _extract_python_concepts(self, code: str) -> list[str]:
        """Extract Python-specific concepts."""
        concepts = []

        patterns = {
            "async/await": r"\b(async|await)\b",
            "decorators": r"@\w+",
            "list_comprehension": r"\[[^\]]+for\s+\w+\s+in\s+[^\]]+\]",
            "lambda": r"\blambda\b",
            "classes": r"\bclass\s+\w+",
            "imports": r"\b(import|from)\s+\w+",
        }

        for concept, pattern in patterns.items():
            if re.search(pattern, code):
                concepts.append(concept)

        return concepts

    def _extract_js_concepts(self, code: str) -> list[str]:
        """Extract JavaScript/TypeScript-specific concepts."""
        concepts = []

        patterns = {
            "arrow_functions": r"=>",
            "promises": r"\b(Promise|then|catch)\b",
            "async/await": r"\b(async|await)\b",
            "destructuring": r"\{[^}]+\}\s*=",
            "spread_operator": r"\.\.\.",
            "template_literals": r"`[^`]*\$\{[^}]*\}[^`]*`",
        }

        for concept, pattern in patterns.items():
            if re.search(pattern, code):
                concepts.append(concept)

        return concepts


class InformationSynthesizer:
    """Synthesize information from multiple sources into actionable knowledge."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.extractor = InformationExtractor()

    async def synthesize_knowledge(
        self, knowledge_base: KnowledgeBase, research_files: list[ResearchFile]
    ) -> SynthesizedKnowledge:
        """
        Synthesize comprehensive knowledge from research materials.

        Args:
            knowledge_base: Basic knowledge base from research processor
            research_files: Original research files for detailed analysis

        Returns:
            Enhanced knowledge base with synthesized information
        """
        try:
            self.logger.info("Starting knowledge synthesis process")

            # Extract technical concepts
            technical_concepts = await self.extractor.extract_technical_concepts(
                research_files
            )

            # Extract problem-solution pairs
            problem_solutions = await self.extractor.extract_problem_solution_pairs(
                research_files
            )

            # Extract code examples
            code_examples = await self.extractor.extract_code_examples(research_files)

            # Generate technical details
            technical_details = await self._generate_technical_details(
                technical_concepts, research_files
            )

            # Identify key themes
            key_themes = await self._identify_key_themes(
                knowledge_base.insights, technical_concepts
            )

            # Extract best practices
            best_practices = await self._extract_best_practices(research_files)

            # Calculate synthesis confidence
            confidence = self._calculate_synthesis_confidence(
                technical_concepts, problem_solutions, code_examples, technical_details
            )

            synthesized = SynthesizedKnowledge(
                original_knowledge=knowledge_base,
                technical_details=technical_details,
                code_examples=code_examples,
                key_themes=key_themes,
                problem_solution_pairs=problem_solutions,
                best_practices=best_practices,
                synthesis_confidence=confidence,
            )

            self.logger.info(
                f"Knowledge synthesis completed with confidence: {confidence:.2f}"
            )
            return synthesized

        except Exception as e:
            self.logger.error(f"Knowledge synthesis failed: {e}")
            raise InformationSynthesisError(f"Failed to synthesize knowledge: {e}")

    async def _generate_technical_details(
        self, concepts: list[TechnicalConcept], research_files: list[ResearchFile]
    ) -> list[TechnicalDetail]:
        """Generate detailed technical information."""
        details = []

        for concept in concepts[:15]:  # Limit to top 15 concepts
            # Find relevant content for this concept
            relevant_content = []
            for file_data in research_files:
                if concept.name.lower() in file_data.content.lower():
                    relevant_content.append(file_data.content)

            if relevant_content:
                # Extract description from contexts
                description = self._extract_concept_description(
                    concept, relevant_content
                )

                # Determine difficulty level
                difficulty = self._assess_concept_difficulty(concept.name)

                technical_detail = TechnicalDetail(
                    concept=concept.name,
                    description=description,
                    use_cases=concept.contexts[:3],  # Use contexts as use cases
                    related_technologies=list(concept.related_concepts)[:5],
                    difficulty_level=difficulty,
                    importance_score=concept.importance_score,
                    confidence_score=min(1.0, concept.frequency / 5.0),
                )
                details.append(technical_detail)

        return details

    def _extract_concept_description(
        self, concept: TechnicalConcept, content_list: list[str]
    ) -> str:
        """Extract description for a technical concept."""
        # Find sentences that mention the concept and seem explanatory
        descriptions = []

        for content in content_list[:3]:  # Limit to first 3 relevant files
            sentences = re.split(r"[.!?]+", content)

            for sentence in sentences:
                if (
                    concept.name.lower() in sentence.lower()
                    and len(sentence.split()) > 5
                ):
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if clean_sentence and len(clean_sentence) < 200:
                        descriptions.append(clean_sentence)
                        break  # One description per file

        if descriptions:
            return ". ".join(descriptions[:2])  # Combine top 2 descriptions
        else:
            return f"{concept.name} - technical concept found in research materials."

    def _assess_concept_difficulty(self, concept_name: str) -> str:
        """Assess the difficulty level of a concept."""
        beginner_concepts = ["html", "css", "javascript", "python", "git", "api"]
        advanced_concepts = [
            "kubernetes",
            "microservices",
            "tensorflow",
            "blockchain",
            "machine learning",
        ]

        concept_lower = concept_name.lower()

        if any(bc in concept_lower for bc in beginner_concepts):
            return "beginner"
        elif any(ac in concept_lower for ac in advanced_concepts):
            return "advanced"
        else:
            return "intermediate"

    async def _identify_key_themes(
        self, insights: list[Insight], concepts: list[TechnicalConcept]
    ) -> list[str]:
        """Identify key themes from insights and concepts."""
        theme_words = []

        # Extract themes from insights
        for insight in insights:
            if insight.category in ["technology", "solution"]:
                words = re.findall(r"\b\w{4,}\b", insight.content.lower())
                theme_words.extend(words)

        # Add concept names
        theme_words.extend([concept.name for concept in concepts[:10]])

        # Count and identify top themes
        word_counts = Counter(theme_words)
        common_words = {
            "this",
            "that",
            "with",
            "from",
            "they",
            "have",
            "been",
            "will",
            "when",
            "where",
        }

        themes = [
            word
            for word, count in word_counts.most_common(10)
            if count >= 2 and word not in common_words and len(word) > 3
        ]

        return themes

    async def _extract_best_practices(
        self, research_files: list[ResearchFile]
    ) -> list[str]:
        """Extract best practices from research content."""
        practices = []

        practice_indicators = [
            "best practice",
            "recommendation",
            "should",
            "avoid",
            "consider",
            "tip:",
            "note:",
            "important:",
            "warning:",
            "remember",
        ]

        for file_data in research_files:
            sentences = re.split(r"[.!?]+", file_data.content)

            for sentence in sentences:
                sentence_lower = sentence.lower().strip()

                if any(
                    indicator in sentence_lower for indicator in practice_indicators
                ):
                    if 20 < len(sentence) < 150:  # Reasonable length
                        practices.append(sentence.strip())

        # Remove duplicates and limit
        unique_practices = list(dict.fromkeys(practices))[:8]

        return unique_practices

    def _calculate_synthesis_confidence(
        self,
        concepts: list[TechnicalConcept],
        problem_solutions: list[dict[str, str]],
        code_examples: list[CodeExample],
        technical_details: list[TechnicalDetail],
    ) -> float:
        """Calculate overall confidence in the synthesis process."""
        factors = []

        # Concept quality factor
        if concepts:
            avg_concept_importance = sum(c.importance_score for c in concepts) / len(
                concepts
            )
            factors.append(min(1.0, avg_concept_importance * 2))

        # Content richness factor
        content_richness = min(
            1.0, (len(concepts) + len(problem_solutions) + len(code_examples)) / 20
        )
        factors.append(content_richness)

        # Technical detail quality factor
        if technical_details:
            avg_detail_confidence = sum(
                td.confidence_score for td in technical_details
            ) / len(technical_details)
            factors.append(avg_detail_confidence)

        # Problem-solution coverage
        ps_coverage = min(
            1.0, len(problem_solutions) / 5
        )  # Ideal: 5+ problem-solution pairs
        factors.append(ps_coverage)

        return sum(factors) / len(factors) if factors else 0.5
