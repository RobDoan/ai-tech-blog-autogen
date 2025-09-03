"""
BlogWriterOrchestrator for the Multi-Agent Blog Writer system.

This orchestrator coordinates all specialized agents to transform a topic and context
into a comprehensive, SEO-optimized blog post through a structured workflow.
"""

import logging
from datetime import datetime
from typing import Any

from .base_agent import AgentState
from .code_agent import CodeAgent
from .content_planner_agent import ContentPlannerAgent
from .critic_agent import CriticAgent
from .multi_agent_models import (
    AgentConfig,
    AgentMessage,
    BlogContent,
    BlogGenerationError,
    BlogInput,
    BlogResult,
    ContentOutline,
    MessageType,
    WorkflowConfig,
)
from .seo_agent import SEOAgent
from .writer_agent import WriterAgent


class BlogWriterOrchestrator:
    """
    Main orchestrator that coordinates all agents in the blog generation workflow.

    Manages the complete process:
    1. Content planning and outline creation
    2. Initial content generation
    3. Content review and refinement
    4. SEO optimization
    5. Code example integration
    6. Final quality assurance
    """

    def __init__(self, agent_config: AgentConfig, workflow_config: WorkflowConfig):
        """
        Initialize the orchestrator with all specialized agents.

        Args:
            agent_config: Configuration for all agents
            workflow_config: Workflow behavior configuration
        """
        self.agent_config = agent_config
        self.workflow_config = workflow_config
        self.logger = self._setup_logger()

        # Initialize all specialized agents
        self.content_planner = ContentPlannerAgent(agent_config)
        self.writer = WriterAgent(agent_config)
        self.critic = CriticAgent(agent_config)
        self.seo_agent = SEOAgent(agent_config) if workflow_config.enable_seo_agent else None
        self.code_agent = CodeAgent(agent_config) if workflow_config.enable_code_agent else None

        # Initialize workflow state
        self.agent_state = AgentState()

        self.logger.info("BlogWriterOrchestrator initialized with all agents")

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Orchestrator - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def generate_blog(
        self,
        topic: str,
        description: str | None = None,
        book_reference: str | None = None
    ) -> BlogResult:
        """
        Generate a complete blog post through the multi-agent workflow.

        Args:
            topic: Main topic for the blog post
            description: Optional additional context
            book_reference: Optional reference material

        Returns:
            BlogResult with the generated content and metadata
        """
        start_time = datetime.now()

        try:
            # Create blog input
            blog_input = BlogInput(
                topic=topic,
                description=description,
                book_reference=book_reference,
                target_audience=self.workflow_config.__dict__.get('target_audience', 'intermediate'),
                preferred_length=self.workflow_config.__dict__.get('preferred_length', 1500)
            )

            self.logger.info(f"Starting blog generation for topic: {topic}")

            # Step 1: Content Planning
            outline = await self._create_content_outline(blog_input)
            self.agent_state.set_workflow_data("outline", outline)

            # Step 2: Initial Content Generation
            initial_content = await self._generate_initial_content(outline, blog_input)
            self.agent_state.set_workflow_data("initial_content", initial_content)

            # Step 3: Content Review and Refinement
            reviewed_content = await self._review_and_refine_content(
                initial_content, outline, blog_input
            )

            # Step 4: SEO Optimization (if enabled)
            if self.seo_agent:
                seo_optimized_content = await self._apply_seo_optimization(
                    reviewed_content, outline
                )
            else:
                seo_optimized_content = reviewed_content

            # Step 5: Code Example Integration (if enabled)
            if self.code_agent:
                final_content = await self._integrate_code_examples(
                    seo_optimized_content, outline
                )
            else:
                final_content = seo_optimized_content

            # Step 6: Final Quality Check
            final_blog_content = await self._perform_final_quality_check(
                final_content, blog_input
            )

            # Generate final result
            generation_time = (datetime.now() - start_time).total_seconds()

            result = BlogResult(
                content=final_blog_content.content,
                metadata={
                    "title": final_blog_content.title,
                    "word_count": final_blog_content.metadata.word_count,
                    "reading_time_minutes": final_blog_content.metadata.reading_time_minutes,
                    "seo_score": final_blog_content.metadata.seo_score,
                    "code_blocks_count": len(final_blog_content.code_blocks),
                    "sections_count": len(final_blog_content.sections)
                },
                generation_log=self.agent_state.conversation_history,
                success=True,
                generation_time_seconds=generation_time
            )

            self.logger.info(
                f"Blog generation completed successfully in {generation_time:.2f}s. "
                f"Generated {final_blog_content.metadata.word_count} words."
            )

            return result

        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()

            self.logger.error(f"Blog generation failed: {e}")

            # Create error result with partial content if available
            partial_content = self._extract_partial_content()

            return BlogResult(
                content=partial_content,
                metadata={
                    "error_occurred": True,
                    "partial_content": len(partial_content) > 0
                },
                generation_log=self.agent_state.conversation_history,
                success=False,
                error_message=str(e),
                generation_time_seconds=generation_time
            )

    async def _create_content_outline(self, blog_input: BlogInput) -> ContentOutline:
        """Create initial content outline using the content planner."""
        self.logger.info("Creating content outline...")

        try:
            outline = await self.content_planner.create_outline(blog_input)

            # Log the planning message
            planning_message = AgentMessage(
                agent_name=self.content_planner.name,
                message_type=MessageType.OUTLINE,
                content=f"Created outline with {len(outline.sections)} sections",
                timestamp=datetime.now(),
                metadata={"outline_title": outline.title}
            )
            self.agent_state.add_message(planning_message)

            self.logger.info(f"Outline created: '{outline.title}' with {len(outline.sections)} sections")
            return outline

        except Exception as e:
            self.logger.error(f"Failed to create outline: {e}")
            raise BlogGenerationError(f"Content outline creation failed: {e}")

    async def _generate_initial_content(
        self,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> BlogContent:
        """Generate initial blog content using the writer agent."""
        self.logger.info("Generating initial content...")

        try:
            content = await self.writer.write_content(outline, blog_input)

            # Log the writing message
            writing_message = AgentMessage(
                agent_name=self.writer.name,
                message_type=MessageType.CONTENT,
                content=f"Generated content with {content.metadata.word_count} words",
                timestamp=datetime.now(),
                metadata={"word_count": content.metadata.word_count}
            )
            self.agent_state.add_message(writing_message)

            self.logger.info(f"Initial content generated: {content.metadata.word_count} words")
            return content

        except Exception as e:
            self.logger.error(f"Failed to generate initial content: {e}")
            raise BlogGenerationError(f"Initial content generation failed: {e}")

    async def _review_and_refine_content(
        self,
        content: BlogContent,
        outline: ContentOutline,
        blog_input: BlogInput
    ) -> BlogContent:
        """Review content and refine based on feedback."""
        self.logger.info("Reviewing and refining content...")

        current_content = content
        max_iterations = self.workflow_config.max_iterations

        for iteration in range(max_iterations):
            self.logger.info(f"Review iteration {iteration + 1}/{max_iterations}")

            try:
                # Get review feedback
                feedback = await self.critic.review_content(
                    current_content,
                    outline,
                    self.workflow_config.quality_threshold
                )

                # Log the review
                review_message = AgentMessage(
                    agent_name=self.critic.name,
                    message_type=MessageType.FEEDBACK,
                    content=f"Review score: {feedback.overall_score}/10, Approved: {feedback.approved}",
                    timestamp=datetime.now(),
                    metadata={"score": feedback.overall_score, "approved": feedback.approved}
                )
                self.agent_state.add_message(review_message)

                # Check if content is approved
                if await self.critic.approve_content(current_content, feedback):
                    self.logger.info(f"Content approved after {iteration + 1} iterations")
                    return current_content

                # If not approved and not final iteration, refine content
                if iteration < max_iterations - 1:
                    self.logger.info("Refining content based on feedback...")

                    # Create feedback summary for revision
                    feedback_summary = f"Score: {feedback.overall_score}/10\\n"
                    feedback_summary += "Issues to address:\\n"
                    for improvement in feedback.improvements:
                        feedback_summary += f"- {improvement}\\n"

                    # Revise content
                    current_content = await self.writer.revise_content(
                        current_content,
                        feedback_summary,
                        outline,
                        blog_input
                    )

                    # Log revision
                    revision_message = AgentMessage(
                        agent_name=self.writer.name,
                        message_type=MessageType.CONTENT,
                        content=f"Revised content based on feedback (iteration {iteration + 1})",
                        timestamp=datetime.now(),
                        metadata={"iteration": iteration + 1}
                    )
                    self.agent_state.add_message(revision_message)

            except Exception as e:
                self.logger.error(f"Error in review iteration {iteration + 1}: {e}")
                if iteration == 0:
                    # If first iteration fails, we can't proceed
                    raise BlogGenerationError(f"Content review failed: {e}")
                else:
                    # Return the last working version
                    self.logger.warning("Using content from previous iteration due to review error")
                    break

        self.logger.info("Content refinement completed")
        return current_content

    async def _apply_seo_optimization(
        self,
        content: BlogContent,
        outline: ContentOutline
    ) -> BlogContent:
        """Apply SEO optimization to the content."""
        if not self.seo_agent:
            return content

        self.logger.info("Applying SEO optimization...")

        try:
            # Analyze keywords
            keyword_analysis = await self.seo_agent.analyze_keywords(
                topic=outline.title,
                content=content,
                outline=outline
            )

            # Optimize content
            seo_optimized = await self.seo_agent.optimize_content(
                content,
                keyword_analysis
            )

            # Update content with SEO improvements
            optimized_content = BlogContent(
                title=seo_optimized.optimized_title,
                content=seo_optimized.optimized_content,
                sections=content.sections,
                code_blocks=content.code_blocks,
                metadata=content.metadata
            )

            # Update metadata with SEO information
            optimized_content.metadata.seo_score = seo_optimized.seo_score
            optimized_content.metadata.meta_description = seo_optimized.meta_description
            optimized_content.metadata.keywords = seo_optimized.keywords_used

            # Log SEO optimization
            seo_message = AgentMessage(
                agent_name=self.seo_agent.name,
                message_type=MessageType.SEO_ANALYSIS,
                content=f"SEO optimization completed, score: {seo_optimized.seo_score}/100",
                timestamp=datetime.now(),
                metadata={"seo_score": seo_optimized.seo_score}
            )
            self.agent_state.add_message(seo_message)

            self.logger.info(f"SEO optimization completed with score: {seo_optimized.seo_score}/100")
            return optimized_content

        except Exception as e:
            self.logger.error(f"SEO optimization failed: {e}")
            # Return original content if SEO fails (graceful degradation)
            self.logger.warning("Continuing without SEO optimization")
            return content

    async def _integrate_code_examples(
        self,
        content: BlogContent,
        outline: ContentOutline
    ) -> BlogContent:
        """Integrate code examples into the content."""
        if not self.code_agent:
            return content

        self.logger.info("Integrating code examples...")

        try:
            # Identify code opportunities
            opportunities = await self.code_agent.identify_code_opportunities(content, outline)

            if not opportunities:
                self.logger.info("No code opportunities identified")
                return content

            # Generate code examples
            code_examples = await self.code_agent.generate_code_examples(
                opportunities,
                content_context=f"Topic: {content.title}"
            )

            if not code_examples:
                self.logger.info("No code examples generated")
                return content

            # Integrate code examples into content
            updated_content = self._integrate_code_into_content(content, code_examples)

            # Log code integration
            code_message = AgentMessage(
                agent_name=self.code_agent.name,
                message_type=MessageType.CODE,
                content=f"Integrated {len(code_examples)} code examples",
                timestamp=datetime.now(),
                metadata={"code_examples_count": len(code_examples)}
            )
            self.agent_state.add_message(code_message)

            self.logger.info(f"Code integration completed with {len(code_examples)} examples")
            return updated_content

        except Exception as e:
            self.logger.error(f"Code integration failed: {e}")
            # Return original content if code integration fails (graceful degradation)
            self.logger.warning("Continuing without code examples")
            return content

    def _integrate_code_into_content(
        self,
        content: BlogContent,
        code_examples: list[Any]
    ) -> BlogContent:
        """Integrate generated code examples into the content."""
        # This is a simplified integration approach
        # In practice, you might want more sophisticated content placement

        updated_content = content.content
        new_code_blocks = list(content.code_blocks)

        for code_example in code_examples:
            code_block = code_example.code_block
            section_title = code_example.opportunity.section_title

            # Find the section and add code after it
            section_pattern = f"## {section_title}"
            if section_pattern in updated_content:
                # Add code block after the section
                code_markdown = f"\\n\\n```{code_block.language}\\n{code_block.code}\\n```\\n\\n"
                if code_block.explanation:
                    code_markdown += f"{code_block.explanation}\\n\\n"

                updated_content = updated_content.replace(
                    section_pattern,
                    section_pattern + code_markdown,
                    1  # Replace only first occurrence
                )

                new_code_blocks.append(code_block)

        # Update word count
        new_word_count = len(updated_content.split())
        content.metadata.word_count = new_word_count
        content.metadata.reading_time_minutes = max(1, new_word_count // 225)

        return BlogContent(
            title=content.title,
            content=updated_content,
            sections=content.sections,
            code_blocks=new_code_blocks,
            metadata=content.metadata
        )

    async def _perform_final_quality_check(
        self,
        content: BlogContent,
        blog_input: BlogInput
    ) -> BlogContent:
        """Perform final quality check on the generated content."""
        self.logger.info("Performing final quality check...")

        try:
            # Basic quality checks
            quality_issues = []

            # Check word count alignment
            target_length = blog_input.preferred_length
            actual_length = content.metadata.word_count
            length_diff = abs(actual_length - target_length) / target_length

            if length_diff > 0.5:  # More than 50% difference
                quality_issues.append(
                    f"Content length ({actual_length}) differs significantly from target ({target_length})"
                )

            # Check structure
            if len(content.sections) < 2:
                quality_issues.append("Content has too few sections")

            if not content.content.startswith('# '):
                quality_issues.append("Content should start with H1 title")

            # Check for conclusion
            content_lower = content.content.lower()
            conclusion_indicators = ['conclusion', 'summary', 'final', 'wrap up']
            has_conclusion = any(indicator in content_lower for indicator in conclusion_indicators)

            if not has_conclusion:
                quality_issues.append("Content appears to lack a conclusion")

            # Log quality check results
            quality_message = AgentMessage(
                agent_name="Orchestrator",
                message_type=MessageType.FEEDBACK,
                content=f"Final quality check: {len(quality_issues)} issues found",
                timestamp=datetime.now(),
                metadata={"quality_issues": quality_issues}
            )
            self.agent_state.add_message(quality_message)

            if quality_issues:
                self.logger.warning(f"Quality issues identified: {quality_issues}")
            else:
                self.logger.info("Final quality check passed")

            return content

        except Exception as e:
            self.logger.error(f"Final quality check failed: {e}")
            return content  # Return content even if quality check fails

    def _extract_partial_content(self) -> str:
        """Extract any partial content generated during workflow."""
        try:
            # Try to get content from workflow state
            initial_content = self.agent_state.get_workflow_data("initial_content")
            if initial_content and hasattr(initial_content, 'content'):
                return initial_content.content

            # Try to get outline
            outline = self.agent_state.get_workflow_data("outline")
            if outline:
                partial = f"# {outline.title}\\n\\n"
                partial += f"## Introduction\\n{outline.introduction}\\n\\n"

                for section in outline.sections:
                    partial += f"## {section.heading}\\n"
                    for point in section.key_points:
                        partial += f"- {point}\\n"
                    partial += "\\n"

                partial += f"## Conclusion\\n{outline.conclusion}\\n"
                return partial

            return ""

        except Exception as e:
            self.logger.error(f"Failed to extract partial content: {e}")
            return ""

    async def regenerate_with_feedback(
        self,
        original_result: BlogResult,
        user_feedback: str
    ) -> BlogResult:
        """
        Regenerate blog content incorporating user feedback.

        Args:
            original_result: Previous generation result
            user_feedback: User's feedback for improvement

        Returns:
            New BlogResult with improvements based on feedback
        """
        self.logger.info("Regenerating content with user feedback...")

        try:
            # Extract original content
            if not original_result.success:
                raise BlogGenerationError("Cannot regenerate from failed result")

            # Create a new agent state for regeneration
            self.agent_state = AgentState()

            # Get original outline if available
            outline = self.agent_state.get_workflow_data("outline")
            if not outline:
                # If no outline, we need to extract topic and recreate
                raise BlogGenerationError("Original outline not available for regeneration")

            # Create blog content from original result
            original_content = BlogContent(
                title=original_result.metadata.get("title", "Untitled"),
                content=original_result.content,
                sections=original_result.metadata.get("sections", []),
                code_blocks=[],  # Simplified - in practice you'd extract these properly
                metadata=None  # Simplified
            )

            # Use critic to provide structured feedback
            feedback_analysis = await self.critic.review_content(original_content, outline)

            # Combine user feedback with critic feedback
            combined_feedback = f"User Feedback: {user_feedback}\\n\\n"
            combined_feedback += f"Current Issues: {', '.join(feedback_analysis.improvements)}"

            # Revise content based on combined feedback
            blog_input = BlogInput(
                topic=original_content.title,
                description=user_feedback
            )

            revised_content = await self.writer.revise_content(
                original_content,
                combined_feedback,
                outline,
                blog_input
            )

            # Create result
            result = BlogResult(
                content=revised_content.content,
                metadata={
                    "title": revised_content.title,
                    "regenerated": True,
                    "user_feedback_incorporated": True
                },
                generation_log=self.agent_state.conversation_history,
                success=True
            )

            self.logger.info("Content regeneration completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Content regeneration failed: {e}")
            return BlogResult(
                content="",
                metadata={"regeneration_failed": True},
                generation_log=self.agent_state.conversation_history,
                success=False,
                error_message=str(e)
            )
