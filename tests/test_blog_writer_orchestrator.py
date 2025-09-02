"""
Unit tests for BlogWriterOrchestrator.

Tests the main workflow coordination, agent collaboration, error handling,
and end-to-end blog generation process.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.autogen_blog.blog_writer_orchestrator import BlogWriterOrchestrator
from src.autogen_blog.multi_agent_models import (
    AgentConfig,
    WorkflowConfig,
    BlogInput,
    BlogResult,
    ContentOutline,
    Section,
    BlogContent,
    CodeBlock,
    TargetAudience,
    MessageType,
    BlogGenerationError,
    ContentQualityError
)


class TestBlogWriterOrchestrator:
    """Test BlogWriterOrchestrator workflow coordination."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key="sk-test-key-123456789012345678901234567890123456789012345678901234",
            timeout_seconds=120
        )

    @pytest.fixture
    def workflow_config(self):
        """Create test workflow configuration."""
        return WorkflowConfig(
            max_iterations=2,
            enable_code_agent=True,
            enable_seo_agent=True,
            quality_threshold=7.0,
            parallel_processing=False
        )

    @pytest.fixture
    def orchestrator(self, agent_config, workflow_config):
        """Create BlogWriterOrchestrator instance."""
        with patch('src.autogen_blog.content_planner_agent.ContentPlannerAgent'), \
             patch('src.autogen_blog.writer_agent.WriterAgent'), \
             patch('src.autogen_blog.critic_agent.CriticAgent'), \
             patch('src.autogen_blog.seo_agent.SEOAgent'), \
             patch('src.autogen_blog.code_agent.CodeAgent'):
            return BlogWriterOrchestrator(agent_config, workflow_config)

    @pytest.fixture
    def sample_outline(self):
        """Create sample ContentOutline for testing."""
        return ContentOutline(
            title="Introduction to FastAPI",
            introduction="This guide introduces FastAPI fundamentals",
            sections=[
                Section(
                    heading="What is FastAPI?",
                    key_points=["Modern framework", "High performance", "Easy to use"],
                    estimated_words=400
                ),
                Section(
                    heading="Getting Started",
                    key_points=["Installation", "First app", "Running server"],
                    estimated_words=600
                ),
                Section(
                    heading="Advanced Features",
                    key_points=["Dependency injection", "Authentication", "Testing"],
                    estimated_words=500
                )
            ],
            conclusion="FastAPI is a powerful framework for building APIs",
            target_keywords=["FastAPI", "Python", "web framework", "REST API"],
            estimated_word_count=1500
        )

    @pytest.fixture
    def sample_blog_content(self):
        """Create sample BlogContent for testing."""
        from src.autogen_blog.multi_agent_models import ContentMetadata
        
        metadata = ContentMetadata(
            word_count=1480,
            reading_time_minutes=7,
            keywords=["fastapi", "python", "web"]
        )
        
        return BlogContent(
            title="Introduction to FastAPI",
            content="# Introduction to FastAPI\n\n## What is FastAPI?\n\nFastAPI is a modern...",
            sections=["What is FastAPI?", "Getting Started", "Advanced Features"],
            code_blocks=[
                CodeBlock(
                    language="python",
                    code="from fastapi import FastAPI\napp = FastAPI()",
                    explanation="Basic FastAPI application setup"
                )
            ],
            metadata=metadata
        )

    def test_orchestrator_initialization(self, orchestrator, agent_config, workflow_config):
        """Test orchestrator initialization."""
        assert orchestrator.agent_config == agent_config
        assert orchestrator.workflow_config == workflow_config
        assert orchestrator.planner is not None
        assert orchestrator.writer is not None
        assert orchestrator.critic is not None
        assert orchestrator.seo_agent is not None
        assert orchestrator.code_agent is not None

    @pytest.mark.asyncio
    async def test_successful_blog_generation(self, orchestrator, sample_outline, sample_blog_content):
        """Test successful end-to-end blog generation."""
        # Mock agent responses
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={
            "overall_score": 8.5,
            "approved": True,
            "feedback": "Excellent content quality"
        })
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        # Generate blog
        result = await orchestrator.generate_blog(
            topic="Introduction to FastAPI",
            description="A beginner's guide"
        )

        assert isinstance(result, BlogResult)
        assert result.success is True
        assert result.error_message is None
        assert "# Introduction to FastAPI" in result.content
        assert result.metadata.get("word_count", 0) > 0
        assert result.generation_time_seconds > 0

        # Verify agent calls
        orchestrator.planner.create_outline.assert_called_once()
        orchestrator.writer.write_content.assert_called_once()
        orchestrator.critic.review_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_blog_generation_with_book_reference(self, orchestrator, sample_outline, sample_blog_content):
        """Test blog generation with book reference."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={"overall_score": 8.0, "approved": True})
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(
            topic="Python Design Patterns",
            description="Common patterns",
            book_reference="Design Patterns by Gang of Four"
        )

        assert result.success is True

        # Check that book reference was passed to planner
        call_args = orchestrator.planner.create_outline.call_args[0][0]
        assert call_args.book_reference == "Design Patterns by Gang of Four"

    @pytest.mark.asyncio
    async def test_content_revision_workflow(self, orchestrator, sample_outline, sample_blog_content):
        """Test content revision when quality is initially below threshold."""
        # Mock initial content with low quality
        low_quality_review = {
            "overall_score": 6.0,  # Below threshold
            "approved": False,
            "feedback": "Content needs more technical depth"
        }

        high_quality_review = {
            "overall_score": 8.5,  # Above threshold
            "approved": True,
            "feedback": "Much improved content"
        }

        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.writer.revise_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(side_effect=[low_quality_review, high_quality_review])
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="FastAPI Testing")

        assert result.success is True

        # Verify revision was called
        orchestrator.writer.revise_content.assert_called_once()
        orchestrator.critic.review_content.assert_called()
        assert orchestrator.critic.review_content.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, orchestrator, sample_outline, sample_blog_content):
        """Test handling when max iterations are reached without approval."""
        # Mock consistently low quality reviews
        low_quality_review = {
            "overall_score": 5.0,
            "approved": False,
            "feedback": "Needs improvement"
        }

        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.writer.revise_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value=low_quality_review)
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="Complex Topic")

        # Should still succeed but with warning
        assert result.success is True
        assert "maximum iterations" in result.metadata.get("warnings", "").lower()

        # Verify max iterations were attempted
        assert orchestrator.critic.review_content.call_count == orchestrator.workflow_config.max_iterations

    @pytest.mark.asyncio
    async def test_seo_agent_disabled(self, orchestrator, sample_outline, sample_blog_content):
        """Test workflow when SEO agent is disabled."""
        orchestrator.workflow_config.enable_seo_agent = False

        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={"overall_score": 8.0, "approved": True})
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="Test Topic")

        assert result.success is True

        # SEO agent should not be called
        orchestrator.seo_agent.optimize_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_agent_disabled(self, orchestrator, sample_outline, sample_blog_content):
        """Test workflow when code agent is disabled."""
        orchestrator.workflow_config.enable_code_agent = False

        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={"overall_score": 8.0, "approved": True})
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)

        result = await orchestrator.generate_blog(topic="Non-Technical Topic")

        assert result.success is True

        # Code agent should not be called
        orchestrator.code_agent.enhance_existing_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_in_planning_stage(self, orchestrator):
        """Test error handling when planning stage fails."""
        orchestrator.planner.create_outline = AsyncMock(side_effect=BlogGenerationError("Planning failed"))

        result = await orchestrator.generate_blog(topic="Test Topic")

        assert result.success is False
        assert "Planning failed" in result.error_message
        assert result.content == ""
        assert result.generation_time_seconds > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_writing_stage(self, orchestrator, sample_outline):
        """Test error handling when writing stage fails."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(side_effect=BlogGenerationError("Writing failed"))

        result = await orchestrator.generate_blog(topic="Test Topic")

        assert result.success is False
        assert "Writing failed" in result.error_message

    @pytest.mark.asyncio
    async def test_partial_success_with_optional_agent_failure(self, orchestrator, sample_outline, sample_blog_content):
        """Test partial success when optional agents fail."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={"overall_score": 8.0, "approved": True})
        orchestrator.seo_agent.optimize_content = AsyncMock(side_effect=Exception("SEO service unavailable"))
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="Test Topic")

        # Should still succeed despite SEO failure
        assert result.success is True
        assert "warnings" in result.metadata
        assert "seo" in result.metadata["warnings"].lower()

    @pytest.mark.asyncio
    async def test_metadata_collection(self, orchestrator, sample_outline, sample_blog_content):
        """Test that metadata is properly collected throughout workflow."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={
            "overall_score": 8.5,
            "approved": True,
            "feedback": "Great content"
        })
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="Metadata Test")

        assert result.success is True
        assert "word_count" in result.metadata
        assert "quality_score" in result.metadata
        assert "target_keywords" in result.metadata
        assert "code_blocks_count" in result.metadata
        assert "word_count" in result.metadata
        assert result.metadata["quality_score"] == 8.5

    def test_create_blog_input(self, orchestrator):
        """Test creation of BlogInput from parameters."""
        blog_input = orchestrator._create_blog_input(
            topic="Test Topic",
            description="Test Description",
            book_reference="Test Book"
        )

        assert isinstance(blog_input, BlogInput)
        assert blog_input.topic == "Test Topic"
        assert blog_input.description == "Test Description"
        assert blog_input.book_reference == "Test Book"
        assert blog_input.target_audience == TargetAudience.INTERMEDIATE  # Default
        assert blog_input.preferred_length == 1500  # Default

    def test_format_final_content(self, orchestrator, sample_blog_content):
        """Test final content formatting with code blocks."""
        code_blocks = [
            CodeBlock(
                language="python",
                code="print('Hello World')",
                explanation="Simple print statement"
            ),
            CodeBlock(
                language="bash",
                code="pip install fastapi",
                explanation="Install FastAPI"
            )
        ]

        formatted_content = orchestrator._format_final_content(
            sample_blog_content.content,
            code_blocks
        )

        assert "```python" in formatted_content
        assert "```bash" in formatted_content
        assert "print('Hello World')" in formatted_content
        assert "pip install fastapi" in formatted_content
        assert "Simple print statement" in formatted_content
        assert "Install FastAPI" in formatted_content

    @pytest.mark.asyncio
    async def test_conversation_logging(self, orchestrator, sample_outline, sample_blog_content):
        """Test that all agent interactions are logged properly."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.critic.review_content = AsyncMock(return_value={"overall_score": 8.0, "approved": True})
        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="Logging Test")

        assert result.success is True
        assert len(result.generation_log) > 0

        # Check for different message types in log
        message_types = {msg.message_type for msg in result.generation_log}
        assert MessageType.OUTLINE in message_types
        assert MessageType.CONTENT in message_types
        assert MessageType.FEEDBACK in message_types

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator, sample_outline):
        """Test handling of agent timeouts."""
        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)

        # Mock writer with long delay
        async def slow_writer(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate long operation
            return sample_blog_content

        orchestrator.writer.write_content = slow_writer

        # This should timeout based on agent config
        with pytest.raises((asyncio.TimeoutError, BlogGenerationError)):
            await asyncio.wait_for(
                orchestrator.generate_blog(topic="Timeout Test"),
                timeout=1.0  # Short timeout for test
            )

    @pytest.mark.asyncio
    async def test_quality_threshold_configuration(self, orchestrator, sample_outline, sample_blog_content):
        """Test that quality threshold is properly applied."""
        # Set high quality threshold
        orchestrator.workflow_config.quality_threshold = 9.0

        orchestrator.planner.create_outline = AsyncMock(return_value=sample_outline)
        orchestrator.writer.write_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.writer.revise_content = AsyncMock(return_value=sample_blog_content)

        # Mock reviews that don't meet threshold
        orchestrator.critic.review_content = AsyncMock(return_value={
            "overall_score": 8.5,  # Below 9.0 threshold
            "approved": False,
            "feedback": "Good but not excellent"
        })

        orchestrator.seo_agent.optimize_content = AsyncMock(return_value=sample_blog_content)
        orchestrator.code_agent.enhance_existing_code = AsyncMock(return_value=sample_blog_content.code_blocks)

        result = await orchestrator.generate_blog(topic="High Quality Test")

        # Should attempt revisions due to high threshold
        orchestrator.writer.revise_content.assert_called()
        assert orchestrator.critic.review_content.call_count >= 2