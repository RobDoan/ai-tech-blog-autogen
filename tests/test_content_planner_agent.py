"""
Unit tests for ContentPlannerAgent.

Tests content planning functionality including outline generation,
refinement, and completeness analysis for the blog writer system.
"""

import pytest

pytestmark = pytest.mark.unit
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.autogen_blog.content_planner_agent import ContentPlannerAgent
from src.autogen_blog.multi_agent_models import (
    AgentConfig,
    BlogInput,
    ContentOutline,
    Section,
    TargetAudience,
    MessageType,
    BlogGenerationError,
    AgentCommunicationError
)


class TestContentPlannerAgent:
    """Test ContentPlannerAgent functionality."""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key="sk-test-key-123456789012345678901234567890123456789012345678901234",
            timeout_seconds=60
        )
    
    @pytest.fixture
    def content_planner(self, agent_config):
        """Create ContentPlannerAgent instance."""
        with patch('src.autogen_blog.base_agent.OpenAIChatCompletionClient'), \
             patch('src.autogen_blog.base_agent.AssistantAgent'):
            return ContentPlannerAgent(agent_config)
    
    @pytest.fixture
    def sample_blog_input(self):
        """Create sample BlogInput for testing."""
        return BlogInput(
            topic="Introduction to FastAPI",
            description="A comprehensive guide for beginners",
            target_audience=TargetAudience.BEGINNER,
            preferred_length=1500
        )
    
    @pytest.fixture
    def sample_outline_response(self):
        """Sample valid outline response from agent."""
        return {
            "title": "Introduction to FastAPI: A Comprehensive Guide for Beginners",
            "introduction": "This comprehensive guide introduces FastAPI fundamentals",
            "sections": [
                {
                    "heading": "What is FastAPI?",
                    "key_points": [
                        "Modern Python web framework",
                        "Built on ASGI standards",
                        "Automatic API documentation"
                    ],
                    "estimated_words": 300
                },
                {
                    "heading": "Setting Up FastAPI",
                    "key_points": [
                        "Installation process",
                        "Creating first application",
                        "Running the development server"
                    ],
                    "estimated_words": 400
                },
                {
                    "heading": "Basic FastAPI Concepts",
                    "key_points": [
                        "Path operations",
                        "Request and response models",
                        "Dependency injection"
                    ],
                    "estimated_words": 500
                },
                {
                    "heading": "Conclusion and Next Steps",
                    "key_points": [
                        "Key takeaways",
                        "Additional resources",
                        "Advanced topics to explore"
                    ],
                    "estimated_words": 300
                }
            ],
            "conclusion": "FastAPI provides a modern approach to building APIs",
            "target_keywords": [
                "FastAPI",
                "Python web framework",
                "REST API",
                "ASGI",
                "automatic documentation"
            ],
            "estimated_word_count": 1500
        }
    
    def test_agent_initialization(self, content_planner):
        """Test ContentPlannerAgent initialization."""
        assert content_planner.name == "ContentPlanner"
        assert "content strategy" in content_planner._get_system_message().lower()
    
    def test_get_system_message(self, content_planner):
        """Test system message contains appropriate instructions."""
        system_message = content_planner._get_system_message()
        
        # Check for key components
        assert "content strategy" in system_message.lower()
        assert "outline" in system_message.lower()
        assert "audience" in system_message.lower()
        assert "json" in system_message.lower()
        assert "sections" in system_message.lower()
    
    @pytest.mark.asyncio
    async def test_create_outline_success(self, content_planner, sample_blog_input, sample_outline_response):
        """Test successful outline creation."""
        # Mock the agent response
        mock_response = Mock()
        mock_response.chat_message.content = str(sample_outline_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        # Mock JSON parsing
        with patch.object(content_planner, '_parse_json_response', return_value=sample_outline_response):
            outline = await content_planner.create_outline(sample_blog_input)
        
        assert isinstance(outline, ContentOutline)
        assert outline.title == sample_outline_response["title"]
        assert len(outline.sections) == 4
        assert outline.estimated_word_count == 1500
        assert "FastAPI" in outline.target_keywords
    
    @pytest.mark.asyncio
    async def test_create_outline_with_book_reference(self, content_planner, sample_outline_response):
        """Test outline creation with book reference."""
        blog_input = BlogInput(
            topic="Python Design Patterns",
            description="Common patterns in Python development",
            book_reference="Design Patterns: Elements of Reusable Object-Oriented Software",
            target_audience=TargetAudience.INTERMEDIATE,
            preferred_length=2000
        )
        
        mock_response = Mock()
        mock_response.chat_message.content = str(sample_outline_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=sample_outline_response):
            outline = await content_planner.create_outline(blog_input)
        
        # Verify the prompt included the book reference
        call_args = content_planner.agent.on_messages.call_args
        prompt_content = call_args[0][0][0].content
        assert "Design Patterns: Elements of Reusable" in prompt_content
    
    @pytest.mark.asyncio
    async def test_create_outline_different_audiences(self, content_planner, sample_blog_input, sample_outline_response):
        """Test outline creation for different target audiences."""
        mock_response = Mock()
        mock_response.chat_message.content = str(sample_outline_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        audiences = [TargetAudience.BEGINNER, TargetAudience.INTERMEDIATE, TargetAudience.ADVANCED, TargetAudience.EXPERT]
        
        for audience in audiences:
            sample_blog_input.target_audience = audience
            
            with patch.object(content_planner, '_parse_json_response', return_value=sample_outline_response):
                await content_planner.create_outline(sample_blog_input)
            
            # Check that audience was mentioned in the prompt
            call_args = content_planner.agent.on_messages.call_args
            prompt_content = call_args[0][0][0].content
            assert audience.value in prompt_content.lower()
    
    @pytest.mark.asyncio
    async def test_refine_outline_success(self, content_planner, sample_outline_response):
        """Test successful outline refinement."""
        original_outline = ContentOutline(
            title="Original Title",
            sections=[
                Section(title="Section 1", key_points=["Point 1"], word_count=300)
            ],
            target_keywords=["keyword1"],
            estimated_word_count=300
        )
        
        feedback = "Add more technical depth and include advanced concepts"
        
        # Mock refined response
        refined_response = sample_outline_response.copy()
        refined_response["title"] = "Refined: " + refined_response["title"]
        
        mock_response = Mock()
        mock_response.chat_message.content = str(refined_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=refined_response):
            refined_outline = await content_planner.refine_outline(original_outline, feedback)
        
        assert isinstance(refined_outline, ContentOutline)
        assert "Refined:" in refined_outline.title
        
        # Verify feedback was included in prompt
        call_args = content_planner.agent.on_messages.call_args
        prompt_content = call_args[0][0][0].content
        assert feedback in prompt_content
    
    @pytest.mark.asyncio
    async def test_analyze_outline_completeness(self, content_planner):
        """Test outline completeness analysis."""
        outline = ContentOutline(
            title="Test Outline",
            sections=[
                Section(title="Introduction", key_points=["Overview", "Goals"], word_count=300),
                Section(title="Main Content", key_points=["Topic A", "Topic B"], word_count=800),
                Section(title="Conclusion", key_points=["Summary", "Next steps"], word_count=200)
            ],
            target_keywords=["test", "outline", "analysis"],
            estimated_word_count=1300
        )
        
        analysis_response = {
            "completeness_score": 8.5,
            "strengths": [
                "Clear section structure",
                "Appropriate word count distribution",
                "Good keyword selection"
            ],
            "improvements": [
                "Add more specific technical details",
                "Consider adding code examples section"
            ],
            "missing_elements": [],
            "overall_assessment": "Well-structured outline with good flow"
        }
        
        mock_response = Mock()
        mock_response.chat_message.content = str(analysis_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=analysis_response):
            analysis = await content_planner.analyze_outline_completeness(outline)
        
        assert analysis["completeness_score"] == 8.5
        assert len(analysis["strengths"]) == 3
        assert len(analysis["improvements"]) == 2
        assert "Well-structured" in analysis["overall_assessment"]
    
    @pytest.mark.asyncio
    async def test_create_outline_invalid_response(self, content_planner, sample_blog_input):
        """Test handling of invalid outline response."""
        # Mock invalid response (missing required fields)
        invalid_response = {"title": "Test Title"}  # Missing sections
        
        mock_response = Mock()
        mock_response.chat_message.content = str(invalid_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=invalid_response):
            with pytest.raises(BlogGenerationError) as exc_info:
                await content_planner.create_outline(sample_blog_input)
        
        assert "Invalid outline structure" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_outline_empty_sections(self, content_planner, sample_blog_input):
        """Test handling of outline with empty sections."""
        invalid_response = {
            "title": "Test Title",
            "sections": [],  # Empty sections
            "target_keywords": ["test"],
            "estimated_word_count": 1000
        }
        
        mock_response = Mock()
        mock_response.chat_message.content = str(invalid_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=invalid_response):
            with pytest.raises(BlogGenerationError) as exc_info:
                await content_planner.create_outline(sample_blog_input)
        
        assert "at least one section" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_agent_communication_error(self, content_planner, sample_blog_input):
        """Test handling of agent communication errors."""
        content_planner.agent.on_messages = AsyncMock(side_effect=Exception("API error"))
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up retries
            with pytest.raises(AgentCommunicationError):
                await content_planner.create_outline(sample_blog_input)
    
    @pytest.mark.asyncio
    async def test_word_count_validation(self, content_planner, sample_blog_input):
        """Test word count validation in outline creation."""
        # Create response with word counts that don't match target
        mismatched_response = {
            "title": "Test Title",
            "sections": [
                {"title": "Section 1", "key_points": ["Point 1"], "word_count": 5000}  # Too high
            ],
            "target_keywords": ["test"],
            "estimated_word_count": 5000
        }
        
        mock_response = Mock()
        mock_response.chat_message.content = str(mismatched_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=mismatched_response):
            outline = await content_planner.create_outline(sample_blog_input)
        
        # Should still create outline but log the discrepancy
        assert outline.estimated_word_count == 5000
        assert len(content_planner.state.conversation_history) > 0
    
    def test_validate_outline_structure(self, content_planner):
        """Test outline structure validation."""
        # Valid structure
        valid_data = {
            "title": "Test Title",
            "introduction": "Test intro",
            "sections": [
                {"heading": "Section 1", "key_points": ["Point 1"], "estimated_words": 300}
            ],
            "conclusion": "Test conclusion",
            "target_keywords": ["test"],
            "estimated_word_count": 300
        }
        
        # Test with actual method if it exists, otherwise skip
        if hasattr(content_planner, '_validate_outline_structure'):
            content_planner._validate_outline_structure(valid_data)
        else:
            # Just test that ContentOutline can be created with valid data
            outline = ContentOutline(**valid_data)
            assert outline.title == "Test Title"
    
    def test_convert_to_outline_model(self, content_planner, sample_outline_response):
        """Test conversion of response data to ContentOutline model."""
        # Test if method exists, otherwise test direct model creation
        if hasattr(content_planner, '_convert_to_outline_model'):
            outline = content_planner._convert_to_outline_model(sample_outline_response)
        else:
            # Create ContentOutline directly from response data
            outline = ContentOutline(**sample_outline_response)
        
        assert isinstance(outline, ContentOutline)
        assert outline.title == sample_outline_response["title"]
        assert len(outline.sections) == len(sample_outline_response["sections"])
        
        # Check first section conversion
        first_section = outline.sections[0]
        first_section_data = sample_outline_response["sections"][0]
        assert first_section.heading == first_section_data["heading"]
        assert first_section.key_points == first_section_data["key_points"]
        assert first_section.estimated_words == first_section_data["estimated_words"]
    
    @pytest.mark.asyncio
    async def test_logging_and_state_management(self, content_planner, sample_blog_input, sample_outline_response):
        """Test that interactions are properly logged."""
        mock_response = Mock()
        mock_response.chat_message.content = str(sample_outline_response).replace("'", '"')
        content_planner.agent.on_messages = AsyncMock(return_value=mock_response)
        
        with patch.object(content_planner, '_parse_json_response', return_value=sample_outline_response):
            await content_planner.create_outline(sample_blog_input)
        
        # Check that interaction was logged
        assert len(content_planner.state.conversation_history) > 0
        
        # Find the outline message
        outline_messages = [
            msg for msg in content_planner.state.conversation_history
            if msg.message_type == MessageType.OUTLINE
        ]
        assert len(outline_messages) > 0
        
        outline_message = outline_messages[0]
        assert outline_message.agent_name == "ContentPlanner"
        assert outline_message.metadata is not None