"""
Pytest configuration and shared fixtures for multi-agent blog writer tests.

Provides common fixtures, test utilities, and configuration for all test modules.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.autogen_blog.multi_agent_models import (
    AgentConfig,
    WorkflowConfig,
    BlogInput,
    ContentOutline,
    Section,
    BlogContent,
    CodeBlock,
    TargetAudience
)


# Configure async testing
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_api_key():
    """Provide a mock OpenAI API key for testing."""
    return "sk-test-key-123456789012345678901234567890123456789012345678901234"


@pytest.fixture
def basic_agent_config(mock_openai_api_key):
    """Create a basic AgentConfig for testing."""
    return AgentConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        openai_api_key=mock_openai_api_key,
        timeout_seconds=60
    )


@pytest.fixture
def basic_workflow_config():
    """Create a basic WorkflowConfig for testing."""
    return WorkflowConfig(
        max_iterations=3,
        enable_code_agent=True,
        enable_seo_agent=True,
        quality_threshold=7.0,
        parallel_processing=False
    )


@pytest.fixture
def sample_blog_input():
    """Create a sample BlogInput for testing."""
    return BlogInput(
        topic="Introduction to FastAPI",
        description="A comprehensive beginner's guide to building REST APIs with FastAPI",
        book_reference="FastAPI Documentation",
        target_audience=TargetAudience.INTERMEDIATE,
        preferred_length=1500
    )


@pytest.fixture
def sample_content_outline():
    """Create a sample ContentOutline for testing."""
    return ContentOutline(
        title="Introduction to FastAPI: Building Modern Python APIs",
        sections=[
            Section(
                title="What is FastAPI?",
                key_points=[
                    "Modern Python web framework",
                    "Built on ASGI and Starlette",
                    "Automatic API documentation",
                    "Type hints integration"
                ],
                word_count=350
            ),
            Section(
                title="Setting Up Your Environment",
                key_points=[
                    "Installing FastAPI and dependencies",
                    "Setting up virtual environment",
                    "IDE configuration and plugins"
                ],
                word_count=300
            ),
            Section(
                title="Creating Your First API",
                key_points=[
                    "Basic application structure",
                    "Defining path operations",
                    "Request and response models",
                    "Running the development server"
                ],
                word_count=500
            ),
            Section(
                title="Advanced Features",
                key_points=[
                    "Dependency injection",
                    "Authentication and authorization",
                    "Database integration",
                    "Testing strategies"
                ],
                word_count=400
            )
        ],
        target_keywords=[
            "FastAPI",
            "Python web framework",
            "REST API",
            "ASGI",
            "automatic documentation",
            "type hints"
        ],
        estimated_word_count=1550
    )


@pytest.fixture
def sample_blog_content():
    """Create sample BlogContent for testing."""
    return BlogContent(
        title="Introduction to FastAPI: Building Modern Python APIs",
        content="""# Introduction to FastAPI: Building Modern Python APIs

## What is FastAPI?

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints...

## Setting Up Your Environment

Before we start building with FastAPI, let's set up our development environment properly...

## Creating Your First API

Now that we have our environment ready, let's create our first FastAPI application...

## Advanced Features

Once you're comfortable with the basics, FastAPI offers many advanced features...""",
        word_count=1485,
        code_blocks=[
            CodeBlock(
                language="python",
                code="""from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}""",
                explanation="Basic FastAPI application with a simple GET endpoint"
            ),
            CodeBlock(
                language="bash",
                code="pip install fastapi uvicorn",
                explanation="Installing FastAPI and the ASGI server"
            )
        ],
        metadata={
            "generated_at": datetime.now().isoformat(),
            "author": "AI Blog Writer",
            "tags": ["fastapi", "python", "web development", "api"]
        }
    )


@pytest.fixture
def mock_agent_response():
    """Create a mock agent response for testing."""
    mock_response = Mock()
    mock_response.chat_message.content = '{"result": "success", "content": "Generated content"}'
    return mock_response


@pytest.fixture
def mock_outline_response():
    """Create a mock outline response structure."""
    return {
        "title": "Test Blog Title",
        "sections": [
            {
                "title": "Introduction",
                "key_points": ["Overview", "Importance", "Scope"],
                "word_count": 300
            },
            {
                "title": "Main Content",
                "key_points": ["Core concepts", "Implementation", "Examples"],
                "word_count": 800
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary", "Next steps", "Resources"],
                "word_count": 200
            }
        ],
        "target_keywords": ["test", "blog", "example"],
        "estimated_word_count": 1300
    }


@pytest.fixture
def mock_content_response():
    """Create a mock content response structure."""
    return {
        "title": "Test Blog Title",
        "content": "# Test Blog Title\n\n## Introduction\n\nContent here...",
        "word_count": 1250,
        "code_blocks": [],
        "metadata": {"generated_at": datetime.now().isoformat()}
    }


@pytest.fixture
def mock_review_response():
    """Create a mock review response structure."""
    return {
        "overall_score": 8.5,
        "approved": True,
        "strengths": [
            "Clear structure and flow",
            "Good use of examples",
            "Appropriate technical depth"
        ],
        "improvements": [
            "Could add more code examples",
            "Consider expanding the conclusion"
        ],
        "specific_feedback": "Well-written content with good technical accuracy"
    }


@pytest.fixture
def mock_seo_response():
    """Create a mock SEO response structure."""
    return {
        "primary_keywords": ["fastapi", "python"],
        "secondary_keywords": ["web framework", "rest api"],
        "long_tail_keywords": ["fastapi tutorial", "python web development"],
        "meta_description": "Learn FastAPI, the modern Python web framework for building high-performance APIs with automatic documentation.",
        "seo_score": 85,
        "recommendations": [
            "Include more long-tail keywords",
            "Optimize heading structure",
            "Add internal links"
        ]
    }


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_mock_async_agent(response_data):
        """Create a mock async agent with specified response."""
        mock_agent = Mock()
        mock_agent.on_messages = AsyncMock(return_value=Mock(
            chat_message=Mock(content=str(response_data).replace("'", '"'))
        ))
        return mock_agent
    
    @staticmethod
    def assert_valid_blog_result(result):
        """Assert that a BlogResult has valid structure."""
        assert hasattr(result, 'success')
        assert hasattr(result, 'content')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'generation_log')
        assert hasattr(result, 'generation_time_seconds')
        
        if result.success:
            assert result.error_message is None
            assert len(result.content) > 0
            assert isinstance(result.metadata, dict)
            assert isinstance(result.generation_log, list)
            assert result.generation_time_seconds > 0
        else:
            assert result.error_message is not None


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Test markers and skip conditions
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring real API access"
    )


def pytest_runtest_setup(item):
    """Set up each test run."""
    # Skip API tests if no OpenAI key is available
    if "api" in item.keywords and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("API tests require OPENAI_API_KEY environment variable")
    
    # Set test timeout for long-running tests
    if "slow" in item.keywords:
        item.config.option.timeout = 600  # 10 minutes for slow tests


# Session-level fixtures for expensive operations
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Ensure test environment variables
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "WARNING")  # Reduce log noise in tests
    
    yield
    
    # Cleanup after all tests
    pass