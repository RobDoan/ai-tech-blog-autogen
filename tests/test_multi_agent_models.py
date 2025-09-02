"""
Unit tests for multi-agent blog writer data models.

Tests Pydantic models, validation, configuration, and data structures
used throughout the multi-agent blog writer system.
"""

import pytest

pytestmark = pytest.mark.unit
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock

from src.autogen_blog.multi_agent_models import (
    BlogInput,
    ContentOutline,
    Section,
    BlogContent,
    CodeBlock,
    BlogResult,
    AgentMessage,
    MessageType,
    TargetAudience,
    AgentConfig,
    WorkflowConfig,
    BlogGenerationError,
    AgentCommunicationError,
    ContentQualityError,
    SEOServiceError
)


class TestBlogInput:
    """Test BlogInput data model and validation."""
    
    def test_valid_blog_input_creation(self):
        """Test creating a valid BlogInput instance."""
        blog_input = BlogInput(
            topic="FastAPI Best Practices",
            description="Comprehensive guide for building production APIs",
            book_reference="FastAPI in Production by Jane Doe",
            target_audience=TargetAudience.ADVANCED,
            preferred_length=2500
        )
        
        assert blog_input.topic == "FastAPI Best Practices"
        assert blog_input.description == "Comprehensive guide for building production APIs"
        assert blog_input.book_reference == "FastAPI in Production by Jane Doe"
        assert blog_input.target_audience == TargetAudience.ADVANCED
        assert blog_input.preferred_length == 2500
    
    def test_minimal_blog_input_creation(self):
        """Test creating BlogInput with minimal required fields."""
        blog_input = BlogInput(topic="Python Testing")
        
        assert blog_input.topic == "Python Testing"
        assert blog_input.description is None
        assert blog_input.book_reference is None
        assert blog_input.target_audience == TargetAudience.INTERMEDIATE
        assert blog_input.preferred_length == 1500
    
    def test_empty_topic_validation(self):
        """Test that empty topic raises validation error."""
        with pytest.raises(ValueError):
            BlogInput(topic="")
    
    def test_whitespace_topic_validation(self):
        """Test that whitespace-only topic raises validation error."""
        with pytest.raises(ValueError):
            BlogInput(topic="   ")
    
    def test_length_bounds_validation(self):
        """Test preferred_length validation bounds."""
        # Test minimum bound
        with pytest.raises(ValueError):
            BlogInput(topic="Test", preferred_length=400)
        
        # Test maximum bound
        with pytest.raises(ValueError):
            BlogInput(topic="Test", preferred_length=6000)
        
        # Test valid bounds
        blog_input = BlogInput(topic="Test", preferred_length=500)
        assert blog_input.preferred_length == 500
        
        blog_input = BlogInput(topic="Test", preferred_length=5000)
        assert blog_input.preferred_length == 5000


class TestContentOutline:
    """Test ContentOutline data model and structure."""
    
    def test_valid_content_outline_creation(self):
        """Test creating a valid ContentOutline."""
        sections = [
            Section(
                heading="Introduction",
                key_points=["Overview", "Problem statement", "Solution preview"],
                estimated_words=300
            ),
            Section(
                heading="Core Concepts",
                key_points=["Concept A", "Concept B", "Implementation"],
                estimated_words=800
            )
        ]
        
        outline = ContentOutline(
            title="Complete Guide to FastAPI",
            introduction="This guide covers FastAPI fundamentals",
            sections=sections,
            conclusion="FastAPI is powerful and easy to use",
            target_keywords=["FastAPI", "Python", "API", "REST"],
            estimated_word_count=1100
        )
        
        assert outline.title == "Complete Guide to FastAPI"
        assert len(outline.sections) == 2
        assert outline.sections[0].heading == "Introduction"
        assert outline.target_keywords == ["FastAPI", "Python", "API", "REST"]
        assert outline.estimated_word_count == 1100
    
    def test_empty_sections_validation(self):
        """Test that outline requires at least one section."""
        with pytest.raises(ValueError):
            ContentOutline(
                title="Test Title",
                sections=[],
                target_keywords=["test"],
                estimated_word_count=100
            )


class TestSection:
    """Test Section data model."""
    
    def test_valid_section_creation(self):
        """Test creating a valid Section."""
        section = Section(
            heading="Advanced Features",
            key_points=["Feature A", "Feature B", "Best practices"],
            estimated_words=600
        )
        
        assert section.heading == "Advanced Features"
        assert len(section.key_points) == 3
        assert section.estimated_words == 600
    
    def test_empty_key_points_validation(self):
        """Test that section requires at least one key point."""
        # Section allows empty key_points, so this test should pass
        section = Section(
            heading="Test Section",
            key_points=[],
            estimated_words=100
        )
        assert section.heading == "Test Section"


class TestBlogContent:
    """Test BlogContent data model."""
    
    def test_valid_blog_content_creation(self):
        """Test creating valid BlogContent."""
        code_blocks = [
            CodeBlock(
                language="python",
                code="def hello():\n    return 'Hello World'",
                explanation="Simple hello world function"
            )
        ]
        
        from src.autogen_blog.multi_agent_models import ContentMetadata
        
        metadata = ContentMetadata(
            word_count=1200,
            keywords=["python", "basics"]
        )
        
        content = BlogContent(
            title="Python Basics",
            content="# Python Basics\n\nThis is content...",
            sections=["Introduction", "Core Concepts", "Examples"],
            code_blocks=code_blocks,
            metadata=metadata
        )
        
        assert content.title == "Python Basics"
        assert "# Python Basics" in content.content
        assert content.metadata.word_count == 1200
        assert len(content.code_blocks) == 1
        assert "python" in content.metadata.keywords


class TestCodeBlock:
    """Test CodeBlock data model."""
    
    def test_valid_code_block_creation(self):
        """Test creating valid CodeBlock."""
        code_block = CodeBlock(
            language="javascript",
            code="const greeting = () => 'Hello World';",
            explanation="Arrow function example"
        )
        
        assert code_block.language == "javascript"
        assert "const greeting" in code_block.code
        assert code_block.explanation == "Arrow function example"
    
    def test_code_block_with_explanation(self):
        """Test creating CodeBlock with explanation (required field)."""
        code_block = CodeBlock(
            language="python",
            code="print('Hello')",
            explanation="Simple print statement"
        )
        
        assert code_block.language == "python"
        assert code_block.code == "print('Hello')"
        assert code_block.explanation == "Simple print statement"


class TestAgentMessage:
    """Test AgentMessage data model."""
    
    def test_valid_agent_message_creation(self):
        """Test creating valid AgentMessage."""
        timestamp = datetime.now()
        metadata = {"confidence": 0.95, "tokens_used": 150}
        
        message = AgentMessage(
            agent_name="ContentPlanner",
            message_type=MessageType.OUTLINE,
            content="Generated outline content",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert message.agent_name == "ContentPlanner"
        assert message.message_type == MessageType.OUTLINE
        assert message.content == "Generated outline content"
        assert message.timestamp == timestamp
        assert message.metadata["confidence"] == 0.95


class TestBlogResult:
    """Test BlogResult data model."""
    
    def test_successful_blog_result(self):
        """Test creating successful BlogResult."""
        generation_log = [
            AgentMessage(
                agent_name="Writer",
                message_type=MessageType.CONTENT,
                content="Content generated",
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        result = BlogResult(
            content="# Blog Title\n\nContent here...",
            metadata={"word_count": 1500, "seo_score": 85},
            generation_log=generation_log,
            success=True,
            error_message=None,
            generation_time_seconds=45.2
        )
        
        assert result.success is True
        assert result.error_message is None
        assert result.metadata["word_count"] == 1500
        assert result.generation_time_seconds == 45.2
        assert len(result.generation_log) == 1
    
    def test_failed_blog_result(self):
        """Test creating failed BlogResult."""
        result = BlogResult(
            content="",
            metadata={},
            generation_log=[],
            success=False,
            error_message="OpenAI API rate limit exceeded",
            generation_time_seconds=5.0
        )
        
        assert result.success is False
        assert result.error_message == "OpenAI API rate limit exceeded"
        assert result.content == ""


class TestAgentConfig:
    """Test AgentConfig data model and validation."""
    
    def test_valid_agent_config(self):
        """Test creating valid AgentConfig."""
        config = AgentConfig(
            model="gpt-4",
            temperature=0.8,
            max_tokens=4000,
            openai_api_key="sk-test-key",
            timeout_seconds=60
        )
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 4000
        assert config.openai_api_key == "sk-test-key"
        assert config.timeout_seconds == 60
    
    def test_temperature_bounds_validation(self):
        """Test temperature validation bounds."""
        # Test minimum bound
        with pytest.raises(ValueError):
            AgentConfig(
                model="gpt-4",
                temperature=-0.1,
                openai_api_key="sk-test-key"
            )
        
        # Test maximum bound
        with pytest.raises(ValueError):
            AgentConfig(
                model="gpt-4",
                temperature=2.1,
                openai_api_key="sk-test-key"
            )
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            AgentConfig(
                model="gpt-4",
                max_tokens=0,
                openai_api_key="sk-test-key"
            )


class TestWorkflowConfig:
    """Test WorkflowConfig data model."""
    
    def test_valid_workflow_config(self):
        """Test creating valid WorkflowConfig."""
        config = WorkflowConfig(
            max_iterations=5,
            enable_code_agent=False,
            enable_seo_agent=True,
            quality_threshold=8.5,
            parallel_processing=True
        )
        
        assert config.max_iterations == 5
        assert config.enable_code_agent is False
        assert config.enable_seo_agent is True
        assert config.quality_threshold == 8.5
        assert config.parallel_processing is True
    
    def test_max_iterations_validation(self):
        """Test max_iterations validation."""
        with pytest.raises(ValueError):
            WorkflowConfig(max_iterations=0)
        
        with pytest.raises(ValueError):
            WorkflowConfig(max_iterations=11)
    
    def test_quality_threshold_validation(self):
        """Test quality_threshold validation."""
        with pytest.raises(ValueError):
            WorkflowConfig(quality_threshold=-1)
        
        with pytest.raises(ValueError):
            WorkflowConfig(quality_threshold=11)


class TestExceptionClasses:
    """Test custom exception classes."""
    
    def test_blog_generation_error(self):
        """Test BlogGenerationError exception."""
        error = BlogGenerationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_agent_communication_error(self):
        """Test AgentCommunicationError inheritance."""
        error = AgentCommunicationError("Communication failed")
        assert str(error) == "Communication failed"
        assert isinstance(error, BlogGenerationError)
        assert isinstance(error, Exception)
    
    def test_content_quality_error(self):
        """Test ContentQualityError inheritance."""
        error = ContentQualityError("Content quality below threshold")
        assert isinstance(error, BlogGenerationError)
    
    def test_seo_service_error(self):
        """Test SEOServiceError inheritance."""
        error = SEOServiceError("SEO analysis failed")
        assert isinstance(error, BlogGenerationError)


class TestMessageTypeEnum:
    """Test MessageType enum."""
    
    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.OUTLINE.value == "outline"
        assert MessageType.CONTENT.value == "content"
        assert MessageType.FEEDBACK.value == "feedback"
        assert MessageType.CODE.value == "code"
        assert MessageType.SEO_ANALYSIS.value == "seo_analysis"
        assert MessageType.APPROVAL.value == "approval"
        assert MessageType.ERROR.value == "error"


class TestTargetAudienceEnum:
    """Test TargetAudience enum."""
    
    def test_target_audience_values(self):
        """Test TargetAudience enum values."""
        assert TargetAudience.BEGINNER.value == "beginner"
        assert TargetAudience.INTERMEDIATE.value == "intermediate"
        assert TargetAudience.ADVANCED.value == "advanced"
        assert TargetAudience.EXPERT.value == "expert"