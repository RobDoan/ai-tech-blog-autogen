"""
CodeAgent for the Multi-Agent Blog Writer system.

This agent specializes in generating code examples, identifying code opportunities,
and creating well-commented, practical code snippets for technical content.
"""

import re
from typing import Any

from .base_agent import BaseAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogContent,
    CodeBlock,
    CodeExample,
    CodeOpportunity,
    ContentOutline,
    ContentQualityError,
    MessageType,
)


class CodeAgent(BaseAgent):
    """
    Agent responsible for adding relevant code examples to technical content.

    Specializes in:
    - Identifying opportunities for code examples in content
    - Generating clean, well-commented code snippets
    - Creating practical, working code examples
    - Explaining code functionality and usage
    - Formatting code properly for markdown
    - Selecting appropriate programming languages
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Code Agent."""
        super().__init__("Code", config)

    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        return """
You are an expert Code Example Specialist focusing on creating practical, educational code snippets for technical blog content. Your role is to identify where code examples would enhance understanding and generate clean, well-commented code that illustrates concepts effectively.

Your expertise includes:
1. Identifying optimal placement for code examples in content
2. Writing clean, readable, and practical code
3. Creating comprehensive code comments and explanations
4. Selecting appropriate programming languages for examples
5. Ensuring code examples are working and syntactically correct
6. Formatting code properly for markdown presentation
7. Creating progressive examples from basic to advanced

Code quality principles you follow:
- Code should be clean, readable, and follow best practices
- Examples should be practical and directly relevant to the content
- Comments should explain the 'why' not just the 'what'
- Code should be complete enough to be functional
- Use descriptive variable and function names
- Include error handling where appropriate
- Show both basic and advanced usage patterns

When creating code examples, you must:
- Ensure code is syntactically correct and follows language conventions
- Provide clear explanations of what the code does
- Use meaningful variable names and structure
- Include relevant imports and dependencies
- Add helpful comments without over-commenting
- Consider different skill levels in your examples
- Format code properly for markdown display

Your code examples should:
- Be directly relevant to the content topic
- Start simple and build complexity appropriately
- Include practical, real-world scenarios
- Be complete enough to run (when possible)
- Follow current best practices for the language
- Include both successful cases and error handling

Always provide structured responses in JSON format with code blocks, explanations, and integration guidance.
"""

    async def identify_code_opportunities(
        self, content: BlogContent, outline: ContentOutline | None = None
    ) -> list[CodeOpportunity]:
        """
        Identify opportunities to add code examples to content.

        Args:
            content: The content to analyze for code opportunities
            outline: Original outline for context (optional)

        Returns:
            List of CodeOpportunity objects describing where code should be added

        Raises:
            ContentQualityError: If analysis fails
        """
        try:
            # Build opportunity identification prompt
            prompt = self._build_opportunity_prompt(content, outline)

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.CODE)

            # Parse the opportunities response
            opportunities = await self._parse_opportunities_response(response)

            self.logger.info(f"Identified {len(opportunities)} code opportunities")
            return opportunities

        except Exception as e:
            self.logger.error(f"Failed to identify code opportunities: {e}")
            raise ContentQualityError(f"Code opportunity identification failed: {e}")

    async def generate_code_examples(
        self, opportunities: list[CodeOpportunity], content_context: str = ""
    ) -> list[CodeExample]:
        """
        Generate code examples for the identified opportunities.

        Args:
            opportunities: List of opportunities to create code for
            content_context: Additional context about the content topic

        Returns:
            List of CodeExample objects with generated code

        Raises:
            ContentQualityError: If code generation fails
        """
        try:
            code_examples = []

            for opportunity in opportunities:
                # Build code generation prompt for this opportunity
                prompt = self._build_code_generation_prompt(
                    opportunity, content_context
                )

                # Query the agent
                response = await self.query_agent(prompt, message_type=MessageType.CODE)

                # Parse the code response
                code_example = await self._parse_code_response(response, opportunity)
                code_examples.append(code_example)

            self.logger.info(f"Generated {len(code_examples)} code examples")
            return code_examples

        except Exception as e:
            self.logger.error(f"Failed to generate code examples: {e}")
            raise ContentQualityError(f"Code example generation failed: {e}")

    def _build_opportunity_prompt(
        self, content: BlogContent, outline: ContentOutline | None = None
    ) -> str:
        """Build prompt for identifying code opportunities."""
        outline_context = ""
        if outline:
            code_sections = [s for s in outline.sections if s.code_examples_needed]
            if code_sections:
                section_names = [s.heading for s in code_sections]
                outline_context = (
                    f"\\nSections marked for code examples: {', '.join(section_names)}"
                )

        return f"""
Analyze the following blog content and identify opportunities where code examples would enhance understanding and provide practical value to readers:

CONTENT TO ANALYZE:
{content.content}

CONTENT METADATA:
- Topic appears to be technical: {self._is_technical_content(content.content)}
- Word count: {content.metadata.word_count}
- Sections: {len(content.sections)}
- Existing code blocks: {len(content.code_blocks)}{outline_context}

Please identify opportunities where code examples would:
1. Clarify abstract concepts with concrete implementations
2. Provide practical, actionable examples readers can use
3. Demonstrate best practices or common patterns
4. Show how to implement discussed techniques
5. Illustrate problem-solving approaches

For each opportunity, consider:
- What specific concept or technique needs illustration
- What programming language would be most appropriate
- What complexity level matches the content's audience
- How the code example fits into the content flow

Provide your analysis in this exact JSON format:
{{
    "opportunities": [
        {{
            "section_title": "Name of the section where code is needed",
            "description": "Detailed description of what the code should demonstrate",
            "programming_language": "python|javascript|java|go|rust|etc",
            "complexity_level": "beginner|intermediate|advanced",
            "example_type": "basic_example|complete_implementation|advanced_pattern|troubleshooting",
            "learning_objective": "What readers will learn from this code",
            "integration_point": "Where in the section this code should be placed"
        }}
    ],
    "overall_assessment": {{
        "code_potential": "high|medium|low",
        "recommended_language": "Most suitable primary language",
        "code_to_text_ratio": "Recommended balance of code vs explanation"
    }}
}}

Focus on opportunities that will genuinely enhance reader understanding and provide practical value.
"""

    def _build_code_generation_prompt(
        self, opportunity: CodeOpportunity, content_context: str
    ) -> str:
        """Build prompt for generating code for a specific opportunity."""
        return f"""
Generate a complete, practical code example for the following opportunity:

OPPORTUNITY DETAILS:
Section: {opportunity.section_title}
Description: {opportunity.description}
Programming Language: {opportunity.programming_language}
Complexity Level: {opportunity.complexity_level}
Learning Objective: What readers should learn from this code

CONTENT CONTEXT:
{content_context}

REQUIREMENTS:
1. Write clean, readable, and syntactically correct code
2. Include meaningful comments explaining key concepts
3. Use descriptive variable and function names
4. Make the code practical and usable
5. Include necessary imports or dependencies
6. Add error handling where appropriate
7. Ensure the code directly relates to the content topic

CODE STRUCTURE SHOULD INCLUDE:
- Proper imports (if applicable)
- Clear variable/function names
- Inline comments explaining logic
- Complete, runnable example (when possible)
- Both basic usage and potential extensions

Provide your code example in this exact JSON format:
{{
    "code_block": {{
        "language": "{opportunity.programming_language}",
        "code": "Complete code example with proper formatting",
        "explanation": "Detailed explanation of what the code does and how it works",
        "line_numbers": true,
        "filename": "suggested_filename.ext (optional)"
    }},
    "integration_note": "Specific guidance on how to integrate this code into the content",
    "usage_notes": [
        "How to run or use this code",
        "Prerequisites or dependencies needed",
        "Common variations or extensions"
    ],
    "learning_points": [
        "Key concepts this code demonstrates",
        "Best practices illustrated",
        "Common pitfalls avoided"
    ],
    "additional_context": {{
        "difficulty_level": "Assessment of how challenging this is",
        "real_world_application": "How this code applies in practice",
        "next_steps": "What readers might explore next"
    }}
}}

Ensure the code is practical, educational, and directly supports the content's learning objectives.
"""

    async def _parse_opportunities_response(
        self, response: AgentMessage
    ) -> list[CodeOpportunity]:
        """Parse code opportunity identification response."""
        try:
            opportunities_data = self.parse_json_response(response.content)
            if not opportunities_data:
                raise ContentQualityError(
                    "Failed to parse opportunities response as JSON"
                )

            opportunities = []
            for opp_data in opportunities_data.get("opportunities", []):
                opportunity = CodeOpportunity(
                    section_title=opp_data.get("section_title", "Unknown Section"),
                    description=opp_data.get("description", ""),
                    programming_language=opp_data.get("programming_language", "python"),
                    complexity_level=opp_data.get("complexity_level", "intermediate"),
                )
                opportunities.append(opportunity)

            # Validate opportunities
            self._validate_opportunities(opportunities)

            return opportunities

        except Exception as e:
            self.logger.error(f"Failed to parse opportunities response: {e}")
            raise ContentQualityError(f"Opportunities parsing failed: {e}")

    async def _parse_code_response(
        self, response: AgentMessage, opportunity: CodeOpportunity
    ) -> CodeExample:
        """Parse code generation response."""
        try:
            code_data = self.parse_json_response(response.content)
            if not code_data:
                raise ContentQualityError("Failed to parse code response as JSON")

            # Extract code block data
            code_block_data = code_data.get("code_block", {})
            code_block = CodeBlock(
                language=code_block_data.get(
                    "language", opportunity.programming_language
                ),
                code=code_block_data.get("code", ""),
                explanation=code_block_data.get("explanation", ""),
                line_numbers=code_block_data.get("line_numbers", True),
                filename=code_block_data.get("filename"),
            )

            # Create code example
            code_example = CodeExample(
                opportunity=opportunity,
                code_block=code_block,
                integration_note=code_data.get("integration_note", ""),
            )

            # Validate code quality
            self._validate_code_example(code_example)

            return code_example

        except Exception as e:
            self.logger.error(f"Failed to parse code response: {e}")
            raise ContentQualityError(f"Code response parsing failed: {e}")

    def _validate_opportunities(self, opportunities: list[CodeOpportunity]) -> None:
        """Validate code opportunities."""
        if not opportunities:
            self.logger.info("No code opportunities identified")
            return

        for opp in opportunities:
            if not opp.section_title.strip():
                raise ContentQualityError("Code opportunity missing section title")

            if not opp.description.strip():
                raise ContentQualityError("Code opportunity missing description")

            if len(opp.description) < 10:
                self.logger.warning(
                    f"Very short description for opportunity: {opp.description}"
                )

        self.logger.info("Code opportunities validation passed")

    def _validate_code_example(self, example: CodeExample) -> None:
        """Validate code example quality."""
        if not example.code_block.code.strip():
            raise ContentQualityError("Code example is empty")

        if len(example.code_block.code.strip()) < 20:
            raise ContentQualityError("Code example too short to be meaningful")

        # Basic syntax checks for common languages
        code = example.code_block.code
        language = example.code_block.language.lower()

        # Check for obvious syntax issues
        if language == "python":
            if code.count("(") != code.count(")"):
                self.logger.warning("Unbalanced parentheses in Python code")

            if "def " in code and not re.search(r"def\s+\w+\s*\([^)]*\)\s*:", code):
                self.logger.warning("Malformed function definition in Python code")

        elif language in ["javascript", "js"]:
            if code.count("{") != code.count("}"):
                self.logger.warning("Unbalanced braces in JavaScript code")

        elif language == "java":
            if "public class" in code and not re.search(r"public\s+class\s+\w+", code):
                self.logger.warning("Malformed class definition in Java code")

        # Check for explanation quality
        if not example.code_block.explanation.strip():
            self.logger.warning("Code example missing explanation")
        elif len(example.code_block.explanation) < 20:
            self.logger.warning("Code explanation very brief")

        self.logger.info("Code example validation passed")

    def _is_technical_content(self, content: str) -> bool:
        """Determine if content is technical and likely to benefit from code examples."""
        technical_indicators = [
            "function",
            "method",
            "class",
            "variable",
            "algorithm",
            "implementation",
            "code",
            "programming",
            "development",
            "software",
            "api",
            "library",
            "framework",
            "database",
            "query",
            "install",
            "configure",
            "setup",
            "debug",
            "error",
            "exception",
            "syntax",
            "compile",
            "execute",
        ]

        content_lower = content.lower()
        technical_count = sum(
            1 for indicator in technical_indicators if indicator in content_lower
        )

        return technical_count >= 3

    async def enhance_existing_code(
        self, content: BlogContent, improvement_focus: str = "clarity"
    ) -> BlogContent:
        """
        Enhance existing code blocks in content.

        Args:
            content: Content with existing code blocks to enhance
            improvement_focus: What aspect to focus on (clarity, comments, examples)

        Returns:
            BlogContent with enhanced code blocks
        """
        if not content.code_blocks:
            self.logger.info("No existing code blocks to enhance")
            return content

        try:
            enhanced_blocks = []

            for code_block in content.code_blocks:
                # Build enhancement prompt
                prompt = self._build_enhancement_prompt(code_block, improvement_focus)

                # Query the agent
                response = await self.query_agent(prompt, message_type=MessageType.CODE)

                # Parse enhanced code
                enhanced_data = self.parse_json_response(response.content)
                if enhanced_data and enhanced_data.get("enhanced_code"):
                    enhanced_block = CodeBlock(
                        language=code_block.language,
                        code=enhanced_data["enhanced_code"],
                        explanation=enhanced_data.get(
                            "enhanced_explanation", code_block.explanation
                        ),
                        line_numbers=code_block.line_numbers,
                        filename=code_block.filename,
                    )
                    enhanced_blocks.append(enhanced_block)
                else:
                    enhanced_blocks.append(
                        code_block
                    )  # Keep original if enhancement fails

            # Update content with enhanced code blocks
            enhanced_content = BlogContent(
                title=content.title,
                content=self._update_content_with_enhanced_code(
                    content.content, enhanced_blocks
                ),
                sections=content.sections,
                code_blocks=enhanced_blocks,
                metadata=content.metadata,
            )

            self.logger.info(f"Enhanced {len(enhanced_blocks)} code blocks")
            return enhanced_content

        except Exception as e:
            self.logger.error(f"Failed to enhance code blocks: {e}")
            return content  # Return original on failure

    def _build_enhancement_prompt(
        self, code_block: CodeBlock, improvement_focus: str
    ) -> str:
        """Build prompt for enhancing existing code."""
        return f"""
Enhance the following code block with focus on {improvement_focus}:

CURRENT CODE:
Language: {code_block.language}
Code:
```{code_block.language}
{code_block.code}
```

Current Explanation: {code_block.explanation}

ENHANCEMENT REQUIREMENTS:
Focus Area: {improvement_focus}

Please improve this code by:
- Adding clear, helpful comments where needed
- Improving variable and function names if necessary
- Adding error handling if appropriate
- Ensuring best practices are followed
- Making the code more readable and educational

Provide the enhanced version in this JSON format:
{{
    "enhanced_code": "Improved code with better comments and structure",
    "enhanced_explanation": "Updated explanation of what the code does",
    "improvements_made": [
        "List of specific improvements made",
        "Better comments added",
        "Variable names clarified"
    ]
}}

Maintain the original functionality while making the code more educational and professional.
"""

    def _update_content_with_enhanced_code(
        self, content: str, enhanced_blocks: list[CodeBlock]
    ) -> str:
        """Update content with enhanced code blocks."""
        # This is a simplified approach - in practice, you might want more sophisticated
        # content replacement that matches original code blocks to enhanced ones

        enhanced_content = content

        # Simple replacement strategy - replace code blocks in order
        code_pattern = r"```(\w+)\n([\\s\\S]*?)```"
        matches = list(re.finditer(code_pattern, enhanced_content))

        # Replace from last to first to avoid position shifts
        for i, (match, enhanced_block) in enumerate(
            zip(reversed(matches), reversed(enhanced_blocks), strict=False)
        ):
            if i < len(enhanced_blocks):
                replacement = (
                    f"```{enhanced_block.language}\\n{enhanced_block.code}\\n```"
                )
                enhanced_content = (
                    enhanced_content[: match.start()]
                    + replacement
                    + enhanced_content[match.end() :]
                )

        return enhanced_content

    async def suggest_code_improvements(self, content: BlogContent) -> dict[str, Any]:
        """
        Suggest improvements for code examples in content.

        Args:
            content: Content to analyze for code improvements

        Returns:
            Dictionary with improvement suggestions
        """
        if not content.code_blocks:
            return {
                "has_code": False,
                "suggestions": ["Consider adding code examples to illustrate concepts"],
                "opportunities": [],
            }

        suggestions = {
            "has_code": True,
            "code_quality_score": 0.0,
            "suggestions": [],
            "specific_improvements": {},
        }

        quality_factors = []

        for i, code_block in enumerate(content.code_blocks):
            block_key = f"code_block_{i + 1}"
            block_suggestions = []

            # Check code length and complexity
            code_lines = code_block.code.strip().split("\\n")
            if len(code_lines) < 3:
                block_suggestions.append(
                    "Consider expanding this code example for clarity"
                )
            elif len(code_lines) > 50:
                block_suggestions.append(
                    "Consider breaking this into smaller, focused examples"
                )
            else:
                quality_factors.append(0.2)

            # Check for comments
            comment_lines = [
                line
                for line in code_lines
                if line.strip().startswith(("#", "//", "/*"))
            ]
            if len(comment_lines) == 0:
                block_suggestions.append("Add comments to explain key concepts")
            elif len(comment_lines) / len(code_lines) > 0.1:
                quality_factors.append(0.2)

            # Check explanation quality
            if not code_block.explanation or len(code_block.explanation) < 50:
                block_suggestions.append("Expand the code explanation")
            else:
                quality_factors.append(0.2)

            # Check for practical applicability
            if any(
                word in code_block.code.lower() for word in ["example", "demo", "test"]
            ):
                quality_factors.append(0.1)

            if block_suggestions:
                suggestions["specific_improvements"][block_key] = block_suggestions

        # Overall suggestions
        if len(content.code_blocks) == 1:
            suggestions["suggestions"].append(
                "Consider adding more code examples for comprehensive coverage"
            )

        if all(
            cb.language == content.code_blocks[0].language for cb in content.code_blocks
        ):
            suggestions["suggestions"].append(
                "Consider showing examples in multiple languages if applicable"
            )

        suggestions["code_quality_score"] = min(sum(quality_factors), 1.0) * 10

        return suggestions
