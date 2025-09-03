"""
WriterAgent for the Multi-Agent Blog Writer system.

This agent specializes in technical writing, transforming structured outlines
into comprehensive, well-written blog content with consistent tone and style.
"""

import re
from typing import Any

from .base_agent import BaseAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogContent,
    BlogInput,
    CodeBlock,
    ContentMetadata,
    ContentOutline,
    ContentQualityError,
    MessageType,
)


class WriterAgent(BaseAgent):
    """
    Agent responsible for generating blog content from structured outlines.

    Specializes in:
    - Converting outlines to comprehensive blog content
    - Maintaining consistent tone and writing style
    - Creating engaging introductions and conclusions
    - Formatting content in proper markdown
    - Incorporating technical concepts accessibly
    - Revising content based on feedback
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Writer Agent."""
        super().__init__("Writer", config)

    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        return """
You are an expert Technical Writer specializing in creating comprehensive, engaging blog content. Your role is to transform structured outlines into well-written, accessible technical articles that educate and engage readers.

Your expertise includes:
1. Technical writing with clear explanations and logical flow
2. Adapting complexity level to target audience needs
3. Creating engaging introductions that hook readers
4. Writing informative, practical content sections
5. Crafting conclusions that reinforce value and encourage action
6. Proper markdown formatting and structure
7. Incorporating technical concepts in an accessible way
8. Maintaining consistent tone and style throughout

Writing principles you follow:
- Start with compelling hooks that establish value
- Use clear, concise language appropriate for the target audience
- Structure content with logical flow and smooth transitions
- Include practical examples and real-world applications
- Break up text with headers, lists, and formatting for readability
- End with actionable conclusions that reinforce learning

When writing content, you must:
- Follow the provided outline structure exactly
- Maintain consistent voice and tone throughout
- Use proper markdown formatting (headers, lists, code blocks, links)
- Write at the appropriate technical level for the audience
- Include engaging introductions and valuable conclusions
- Ensure content is comprehensive yet accessible
- Format all content as clean, valid markdown

Focus on creating content that is both technically accurate and highly readable, serving the needs of your target audience while establishing authority and providing genuine value.
"""

    async def write_content(
        self, outline: ContentOutline, blog_input: BlogInput
    ) -> BlogContent:
        """
        Generate comprehensive blog content from an outline.

        Args:
            outline: Structured outline to follow
            blog_input: Original input requirements

        Returns:
            BlogContent with the generated blog post

        Raises:
            ContentQualityError: If content generation fails or produces low-quality results
        """
        try:
            # Build prompt for content generation
            prompt = self._build_content_prompt(outline, blog_input)

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.CONTENT)

            # Parse and validate the response
            blog_content = await self._parse_content_response(
                response, outline, blog_input
            )

            self.logger.info(
                f"Generated blog content: {blog_content.metadata.word_count} words"
            )
            return blog_content

        except Exception as e:
            self.logger.error(f"Failed to generate content: {e}")
            raise ContentQualityError(f"Content generation failed: {e}")

    async def revise_content(
        self,
        current_content: BlogContent,
        feedback: str,
        outline: ContentOutline,
        blog_input: BlogInput,
    ) -> BlogContent:
        """
        Revise existing content based on feedback.

        Args:
            current_content: The content to revise
            feedback: Feedback for improvement
            outline: Original outline for reference
            blog_input: Original input requirements

        Returns:
            Revised BlogContent

        Raises:
            ContentQualityError: If revision fails
        """
        try:
            # Build revision prompt
            prompt = self._build_revision_prompt(
                current_content, feedback, outline, blog_input
            )

            # Query the agent
            response = await self.query_agent(prompt, message_type=MessageType.CONTENT)

            # Parse the revised content
            revised_content = await self._parse_content_response(
                response, outline, blog_input
            )

            self.logger.info(
                f"Revised content: {revised_content.metadata.word_count} words"
            )
            return revised_content

        except Exception as e:
            self.logger.error(f"Failed to revise content: {e}")
            raise ContentQualityError(f"Content revision failed: {e}")

    def _build_content_prompt(
        self, outline: ContentOutline, blog_input: BlogInput
    ) -> str:
        """Build the prompt for generating blog content."""
        sections_text = self._format_outline_for_prompt(outline)

        context_section = ""
        if blog_input.description:
            context_section = f"\\nAdditional Context: {blog_input.description}"
        if blog_input.book_reference:
            context_section += f"\\nReference Material: {blog_input.book_reference}"

        return f"""
Write a comprehensive blog post based on the following approved outline:

Title: {outline.title}
Target Audience: {blog_input.target_audience.value}
Target Length: {blog_input.preferred_length} words{context_section}

OUTLINE TO FOLLOW:
{sections_text}

WRITING REQUIREMENTS:
1. Write in markdown format with proper headers, lists, and formatting
2. Start with an engaging introduction that hooks the reader and establishes value
3. Follow the outline structure exactly - cover all sections and key points
4. Maintain consistent tone appropriate for {blog_input.target_audience.value} level readers
5. Use clear, concise language with smooth transitions between sections
6. Include practical examples and real-world applications where relevant
7. End with a strong conclusion that reinforces value and next steps
8. Use proper markdown formatting throughout (# ## ### for headers, - for lists, etc.)
9. Write approximately {blog_input.preferred_length} words total

STRUCTURE YOUR RESPONSE AS:
- Start with the title as an H1 header (# Title)
- Write an engaging introduction (2-3 paragraphs)
- Create each section as specified in the outline using H2 headers (## Section Name)
- Include subsections with H3 headers where appropriate (### Subsection)
- End with a compelling conclusion
- Use markdown formatting throughout

Focus on creating valuable, comprehensive content that serves the reader's needs while being highly readable and engaging. Ensure the content flows logically and provides actionable insights.
"""

    def _build_revision_prompt(
        self,
        current_content: BlogContent,
        feedback: str,
        outline: ContentOutline,
        blog_input: BlogInput,
    ) -> str:
        """Build the prompt for revising blog content based on feedback."""
        return f"""
Please revise the following blog post based on the feedback provided:

CURRENT BLOG POST:
{current_content.content}

FEEDBACK TO ADDRESS:
{feedback}

ORIGINAL REQUIREMENTS:
- Title: {outline.title}
- Target Audience: {blog_input.target_audience.value}
- Target Length: {blog_input.preferred_length} words
- Additional Context: {blog_input.description or "None"}

REVISION GUIDELINES:
1. Address all points mentioned in the feedback
2. Maintain the overall structure and flow of the blog post
3. Keep the content at the appropriate level for {blog_input.target_audience.value} readers
4. Ensure proper markdown formatting throughout
5. Maintain approximately {blog_input.preferred_length} words
6. Preserve the engaging tone and readability
7. Make sure all sections from the original outline are still covered

Please provide the complete revised blog post in markdown format, incorporating all the feedback while maintaining high quality and readability.
"""

    def _format_outline_for_prompt(self, outline: ContentOutline) -> str:
        """Format the outline for inclusion in prompts."""
        sections_text = f"Introduction: {outline.introduction}\\n\\n"

        for i, section in enumerate(outline.sections, 1):
            sections_text += f"Section {i}: {section.heading}\\n"
            sections_text += "Key points to cover:\\n"
            for point in section.key_points:
                sections_text += f"  - {point}\\n"
            if section.code_examples_needed:
                sections_text += "  - Include code examples as needed\\n"
            sections_text += f"Estimated words: {section.estimated_words}\\n\\n"

        sections_text += f"Conclusion: {outline.conclusion}"

        return sections_text

    async def _parse_content_response(
        self, response: AgentMessage, outline: ContentOutline, blog_input: BlogInput
    ) -> BlogContent:
        """
        Parse the agent's response into a BlogContent object.

        Args:
            response: Response from the agent
            outline: Original outline for validation
            blog_input: Original input for validation

        Returns:
            BlogContent object

        Raises:
            ContentQualityError: If parsing fails or content is invalid
        """
        try:
            content = self.clean_markdown_content(response.content)

            # Extract title from content
            title = self._extract_title_from_content(content, outline.title)

            # Extract sections
            sections = self._extract_sections_from_content(content)

            # Extract code blocks
            code_blocks = self._extract_code_blocks_from_content(content)

            # Calculate metadata
            metadata = self._calculate_content_metadata(content)

            # Create BlogContent object
            blog_content = BlogContent(
                title=title,
                content=content,
                sections=sections,
                code_blocks=code_blocks,
                metadata=metadata,
            )

            # Validate content quality
            await self._validate_content_quality(blog_content, outline, blog_input)

            return blog_content

        except Exception as e:
            self.logger.error(f"Failed to parse content response: {e}")
            raise ContentQualityError(f"Content parsing failed: {e}")

    def clean_markdown_content(self, content: str) -> str:
        """Clean and normalize markdown content."""
        # Remove excessive whitespace
        content = re.sub(r"\\n\\s*\\n\\s*\\n", "\\n\\n", content)

        # Ensure proper header spacing
        content = re.sub(r"\\n(#{1,6}\\s+)", "\\n\\n\\1", content)
        content = re.sub(r"(#{1,6}\\s+[^\\n]+)\\n([^\\n#])", "\\1\\n\\n\\2", content)

        # Clean up list formatting
        content = re.sub(r"\\n([-*+]\\s+)", "\\n\\1", content)

        # Strip leading/trailing whitespace
        content = content.strip()

        return content

    def _extract_title_from_content(self, content: str, fallback_title: str) -> str:
        """Extract the title from markdown content."""
        lines = content.split("\\n")
        for line in lines:
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
        return fallback_title

    def _extract_sections_from_content(self, content: str) -> list[str]:
        """Extract section headings from markdown content."""
        sections = []
        lines = content.split("\\n")

        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                sections.append(line[3:].strip())

        return sections

    def _extract_code_blocks_from_content(self, content: str) -> list[CodeBlock]:
        """Extract code blocks from markdown content."""
        code_blocks = []

        # Find code blocks with language specification
        pattern = r"```(\\w+)?\\n([\\s\\S]*?)```"
        matches = re.finditer(pattern, content)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()

            # Find any explanation before or after the code block
            explanation = ""
            # This is a simple heuristic - in practice you might want more sophisticated extraction

            code_block = CodeBlock(
                language=language, code=code, explanation=explanation, line_numbers=True
            )
            code_blocks.append(code_block)

        return code_blocks

    def _calculate_content_metadata(self, content: str) -> ContentMetadata:
        """Calculate metadata for the content."""
        # Count words (approximate)
        word_count = len(content.split())

        # Estimate reading time (average 200-250 words per minute)
        reading_time_minutes = max(1, word_count // 225)

        # Extract potential keywords (simple heuristic)
        keywords = []
        # Remove markdown formatting for keyword extraction
        text_content = re.sub(r"[#*`_\\[\\](){}]", " ", content.lower())
        # This is a simplified approach - you might want more sophisticated keyword extraction

        return ContentMetadata(
            word_count=word_count,
            reading_time_minutes=reading_time_minutes,
            seo_score=None,  # Will be set by SEO agent
            keywords=keywords,
            meta_description=None,  # Will be set by SEO agent
        )

    async def _validate_content_quality(
        self, content: BlogContent, outline: ContentOutline, blog_input: BlogInput
    ) -> None:
        """
        Validate that the content meets quality standards.

        Args:
            content: The content to validate
            outline: Original outline for comparison
            blog_input: Original requirements

        Raises:
            ContentQualityError: If content doesn't meet quality standards
        """
        # Check minimum word count
        if content.metadata.word_count < 200:
            raise ContentQualityError("Content too short - minimum 200 words required")

        # Check if content is extremely long
        max_words = blog_input.preferred_length * 2
        if content.metadata.word_count > max_words:
            raise ContentQualityError(f"Content too long - maximum {max_words} words")

        # Check if major sections are present
        content_lower = content.content.lower()

        # Look for introduction (should be near the beginning)
        intro_found = False
        lines = content.content.split("\\n")[:10]  # Check first 10 lines
        for line in lines:
            if len(line.split()) > 10:  # Substantial content line
                intro_found = True
                break

        if not intro_found:
            raise ContentQualityError("No substantial introduction found")

        # Check for headers (markdown structure)
        if "##" not in content.content:
            raise ContentQualityError(
                "Content lacks proper section structure (missing H2 headers)"
            )

        # Check if outline sections are covered
        missing_sections = []
        for section in outline.sections:
            section_heading_lower = section.heading.lower()
            # Check if section heading or key concept appears in content
            if not any(word in content_lower for word in section_heading_lower.split()):
                missing_sections.append(section.heading)

        if len(missing_sections) > len(outline.sections) // 2:
            self.logger.warning(
                f"Many outline sections seem missing: {missing_sections}"
            )

        # Check for conclusion (should be near the end)
        conclusion_indicators = [
            "conclusion",
            "summary",
            "final",
            "wrap up",
            "in summary",
        ]
        last_quarter = content.content[-len(content.content) // 4 :].lower()
        conclusion_found = any(
            indicator in last_quarter for indicator in conclusion_indicators
        )

        if not conclusion_found:
            self.logger.warning("No clear conclusion section found")

        # Validate markdown formatting
        if not content.content.startswith("# "):
            raise ContentQualityError("Content must start with H1 title")

        self.logger.info("Content validation passed")

    async def enhance_content_readability(self, content: BlogContent) -> BlogContent:
        """
        Enhance content readability with formatting improvements.

        Args:
            content: Content to enhance

        Returns:
            Enhanced BlogContent
        """
        enhanced_content = content.content

        # Add line breaks before headers if missing
        enhanced_content = re.sub(
            r"([^\\n])\\n(#{2,6}\\s+)", "\\1\\n\\n\\2", enhanced_content
        )

        # Ensure proper spacing after headers
        enhanced_content = re.sub(
            r"(#{1,6}\\s+[^\\n]+)\\n([^\\n#-\\*])", "\\1\\n\\n\\2", enhanced_content
        )

        # Improve list formatting
        enhanced_content = re.sub(
            r"([^\\n])\\n([-\\*+]\\s+)", "\\1\\n\\n\\2", enhanced_content
        )

        # Clean up excessive line breaks
        enhanced_content = re.sub(r"\\n{4,}", "\\n\\n\\n", enhanced_content)

        # Update the content
        enhanced_blog_content = BlogContent(
            title=content.title,
            content=enhanced_content,
            sections=content.sections,
            code_blocks=content.code_blocks,
            metadata=content.metadata,
        )

        return enhanced_blog_content

    async def analyze_content_structure(self, content: BlogContent) -> dict[str, Any]:
        """
        Analyze the structure and quality of written content.

        Args:
            content: Content to analyze

        Returns:
            Analysis results with structure metrics and recommendations
        """
        analysis = {
            "structure_score": 0.0,
            "readability_score": 0.0,
            "completeness_score": 0.0,
            "recommendations": [],
            "strengths": [],
        }

        content_text = content.content

        # Structure analysis
        structure_factors = []

        # Check header hierarchy
        h1_count = len(re.findall(r"^# ", content_text, re.MULTILINE))
        h2_count = len(re.findall(r"^## ", content_text, re.MULTILINE))
        h3_count = len(re.findall(r"^### ", content_text, re.MULTILINE))

        if h1_count == 1:
            structure_factors.append(0.2)
            analysis["strengths"].append("Proper single H1 title")
        else:
            analysis["recommendations"].append(
                "Use exactly one H1 header for the title"
            )

        if h2_count >= 3:
            structure_factors.append(0.3)
            analysis["strengths"].append(
                "Good section structure with multiple H2 headers"
            )
        else:
            analysis["recommendations"].append("Add more main sections (H2 headers)")

        # Check for lists and formatting
        list_count = len(re.findall(r"^[-\\*+] ", content_text, re.MULTILINE))
        if list_count >= 3:
            structure_factors.append(0.2)
            analysis["strengths"].append("Good use of lists for readability")

        # Check for code blocks
        code_block_count = len(re.findall(r"```", content_text)) // 2
        if code_block_count > 0:
            structure_factors.append(0.15)
            analysis["strengths"].append("Includes code examples")

        # Check paragraph length (approximate)
        paragraphs = [
            p.strip()
            for p in content_text.split("\\n\\n")
            if p.strip() and not p.strip().startswith("#")
        ]
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(
                paragraphs
            )
            if 30 <= avg_paragraph_length <= 80:
                structure_factors.append(0.15)
                analysis["strengths"].append("Well-balanced paragraph lengths")
            else:
                if avg_paragraph_length < 30:
                    analysis["recommendations"].append(
                        "Expand paragraphs for more detail"
                    )
                else:
                    analysis["recommendations"].append(
                        "Break up long paragraphs for readability"
                    )

        analysis["structure_score"] = sum(structure_factors)

        # Readability analysis
        readability_factors = []

        # Sentence length analysis (approximate)
        sentences = re.split(r"[.!?]+", content_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(
                sentences
            )
            if 15 <= avg_sentence_length <= 25:
                readability_factors.append(0.3)
                analysis["strengths"].append("Good average sentence length")
            else:
                if avg_sentence_length > 25:
                    analysis["recommendations"].append(
                        "Break up long sentences for clarity"
                    )
                else:
                    analysis["recommendations"].append(
                        "Expand sentences for more detail"
                    )

        # Check for transition words and phrases
        transitions = [
            "however",
            "therefore",
            "furthermore",
            "additionally",
            "meanwhile",
            "consequently",
        ]
        transition_count = sum(
            content_text.lower().count(trans) for trans in transitions
        )
        if transition_count >= 3:
            readability_factors.append(0.2)
            analysis["strengths"].append("Good use of transition words")

        # Check for examples and illustrations
        example_indicators = ["example", "instance", "case", "illustration", "consider"]
        example_count = sum(
            content_text.lower().count(indicator) for indicator in example_indicators
        )
        if example_count >= 2:
            readability_factors.append(0.2)
            analysis["strengths"].append("Includes examples and illustrations")

        # Check introduction and conclusion
        intro_quality = (
            len(content_text.split("\\n\\n")[1].split())
            if len(content_text.split("\\n\\n")) > 1
            else 0
        )
        if intro_quality >= 50:
            readability_factors.append(0.15)
            analysis["strengths"].append("Substantial introduction")

        # Look for conclusion section
        conclusion_found = any(
            word in content_text.lower() for word in ["conclusion", "summary", "final"]
        )
        if conclusion_found:
            readability_factors.append(0.15)
            analysis["strengths"].append("Has conclusion section")
        else:
            analysis["recommendations"].append("Add a clear conclusion section")

        analysis["readability_score"] = sum(readability_factors)

        # Completeness analysis
        completeness_factors = []

        # Word count appropriateness
        word_count = content.metadata.word_count
        if word_count >= 800:
            completeness_factors.append(0.3)
            analysis["strengths"].append("Comprehensive content length")
        elif word_count >= 400:
            completeness_factors.append(0.2)
        else:
            analysis["recommendations"].append(
                "Expand content for more comprehensive coverage"
            )

        # Section coverage
        if len(content.sections) >= 4:
            completeness_factors.append(0.3)
            analysis["strengths"].append("Covers multiple topics/sections")

        # Practical content indicators
        practical_words = [
            "how",
            "step",
            "method",
            "approach",
            "technique",
            "way",
            "process",
        ]
        practical_count = sum(
            content_text.lower().count(word) for word in practical_words
        )
        if practical_count >= 5:
            completeness_factors.append(0.2)
            analysis["strengths"].append("Practical, actionable content")

        # Technical depth (for technical content)
        technical_indicators = ["implement", "configure", "setup", "install", "deploy"]
        technical_count = sum(
            content_text.lower().count(word) for word in technical_indicators
        )
        if technical_count >= 3:
            completeness_factors.append(0.2)
            analysis["strengths"].append("Good technical depth")

        analysis["completeness_score"] = sum(completeness_factors)

        return analysis
