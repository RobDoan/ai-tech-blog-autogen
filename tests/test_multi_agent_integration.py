"""
Integration tests for the Multi-Agent Blog Writer system.

Tests end-to-end workflows, agent communication, real API integration
(when API keys are available), and complete blog generation scenarios.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.autogen_blog.blog_writer_orchestrator import BlogWriterOrchestrator
from src.autogen_blog.multi_agent_models import (
    AgentConfig,
    WorkflowConfig,
    BlogInput,
    TargetAudience,
    MessageType
)


@pytest.mark.integration
class TestMultiAgentIntegration:
    """Integration tests for multi-agent blog generation."""
    
    @pytest.fixture
    def real_agent_config(self):
        """Create agent configuration for real API integration."""
        api_key = os.getenv("OPENAI_API_KEY")
        return AgentConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key=api_key or "sk-test-key-123456789012345678901234567890123456789012345678901234",
            timeout_seconds=180
        )
    
    @pytest.fixture
    def integration_workflow_config(self):
        """Create workflow configuration for integration tests."""
        return WorkflowConfig(
            max_iterations=2,
            enable_code_agent=True,
            enable_seo_agent=True,
            quality_threshold=6.0,  # Lower threshold for testing
            parallel_processing=False
        )
    
    @pytest.fixture
    def orchestrator(self, real_agent_config, integration_workflow_config):
        """Create orchestrator for integration tests."""
        return BlogWriterOrchestrator(real_agent_config, integration_workflow_config)
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable for real API integration"
    )
    @pytest.mark.asyncio
    async def test_real_api_blog_generation(self, orchestrator):
        """Test blog generation with real OpenAI API integration."""
        # This test only runs when OPENAI_API_KEY is available
        result = await orchestrator.generate_blog(
            topic="Python Type Hints",
            description="A practical guide to using type hints effectively"
        )
        
        assert result.success is True
        assert result.error_message is None
        assert len(result.content) > 500  # Substantial content
        assert "python" in result.content.lower()
        assert "type" in result.content.lower()
        
        # Check metadata
        assert result.metadata["word_count"] > 0
        assert "quality_score" in result.metadata
        assert len(result.generation_log) > 0
        
        # Verify different agents contributed
        agent_names = {msg.agent_name for msg in result.generation_log}
        assert "ContentPlanner" in agent_names
        assert "Writer" in agent_names
        assert "Critic" in agent_names
    
    @pytest.mark.asyncio
    async def test_mocked_end_to_end_workflow(self, orchestrator):
        """Test complete workflow with mocked agent responses."""
        
        # Mock realistic agent responses
        mock_outline = {
            "title": "Understanding Python Decorators: A Complete Guide",
            "sections": [
                {
                    "title": "What are Decorators?",
                    "key_points": [
                        "Function wrapper concept",
                        "Syntax with @ symbol",
                        "Common use cases"
                    ],
                    "word_count": 400
                },
                {
                    "title": "Creating Custom Decorators",
                    "key_points": [
                        "Basic decorator function",
                        "Decorators with arguments",
                        "Class-based decorators"
                    ],
                    "word_count": 600
                },
                {
                    "title": "Advanced Decorator Patterns",
                    "key_points": [
                        "Preserving metadata",
                        "Chaining decorators",
                        "Performance considerations"
                    ],
                    "word_count": 500
                }
            ],
            "target_keywords": ["Python", "decorators", "functions", "wrapper", "syntax"],
            "estimated_word_count": 1500
        }
        
        mock_content = {
            "title": "Understanding Python Decorators: A Complete Guide",
            "content": """# Understanding Python Decorators: A Complete Guide

## What are Decorators?

Python decorators are a powerful feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code...

## Creating Custom Decorators

To create your own decorator, you need to understand the wrapper function pattern...

## Advanced Decorator Patterns

When working with decorators in production code, there are several advanced patterns...""",
            "word_count": 1450,
            "code_blocks": [],
            "metadata": {"generated_at": datetime.now().isoformat()}
        }
        
        mock_review = {
            "overall_score": 8.2,
            "approved": True,
            "strengths": [
                "Clear explanation of concepts",
                "Good progression from basic to advanced",
                "Practical examples"
            ],
            "improvements": [
                "Could use more code examples",
                "Add performance benchmarks"
            ],
            "specific_feedback": "Excellent structure and flow"
        }
        
        mock_seo_content = mock_content.copy()
        mock_seo_content["metadata"]["seo_score"] = 85
        mock_seo_content["metadata"]["meta_description"] = "Learn Python decorators from basics to advanced patterns with practical examples and best practices."
        
        mock_code_blocks = [
            {
                "language": "python",
                "code": """def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f\"Calling {func.__name__}\")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    return f\"Hello, {name}!\"""",
                "explanation": "Basic decorator example showing function wrapping"
            }
        ]
        
        # Mock all agent responses
        with patch.multiple(
            orchestrator,
            _create_content_outline=AsyncMock(return_value=mock_outline),
            _generate_initial_content=AsyncMock(return_value=mock_content),
            _review_and_refine_content=AsyncMock(return_value=mock_content),
            _optimize_for_seo=AsyncMock(return_value=mock_seo_content),
            _enhance_with_code_examples=AsyncMock(return_value=mock_code_blocks)
        ):
            result = await orchestrator.generate_blog(
                topic="Python Decorators",
                description="Complete guide from basics to advanced"
            )
        
        assert result.success is True
        assert "Understanding Python Decorators" in result.content
        assert "def my_decorator" in result.content
        assert result.metadata["word_count"] == 1450
        assert result.metadata["seo_score"] == 85
    
    @pytest.mark.asyncio
    async def test_agent_communication_flow(self, orchestrator):
        """Test the communication flow between agents."""
        conversation_log = []
        
        def log_interaction(agent_name, message_type, content):
            conversation_log.append({
                "agent": agent_name,
                "type": message_type,
                "content": content,
                "timestamp": datetime.now()
            })
        
        # Mock agents to log their interactions
        async def mock_planner_outline(*args, **kwargs):
            log_interaction("ContentPlanner", MessageType.OUTLINE, "Outline created")
            return {
                "title": "Test Blog",
                "sections": [{"title": "Section 1", "key_points": ["Point 1"], "word_count": 500}],
                "target_keywords": ["test"],
                "estimated_word_count": 500
            }
        
        async def mock_writer_content(*args, **kwargs):
            log_interaction("Writer", MessageType.CONTENT, "Content generated")
            return {
                "title": "Test Blog",
                "content": "# Test Blog\n\nContent here...",
                "word_count": 480,
                "code_blocks": [],
                "metadata": {}
            }
        
        async def mock_critic_review(*args, **kwargs):
            log_interaction("Critic", MessageType.FEEDBACK, "Content reviewed")
            return {"overall_score": 8.0, "approved": True, "feedback": "Good content"}
        
        async def mock_seo_optimize(*args, **kwargs):
            log_interaction("SEOAgent", MessageType.SEO_ANALYSIS, "SEO optimization applied")
            return kwargs.get("content", {})
        
        async def mock_code_enhance(*args, **kwargs):
            log_interaction("CodeAgent", MessageType.CODE_EXAMPLE, "Code examples added")
            return []
        
        with patch.multiple(
            orchestrator,
            _create_content_outline=mock_planner_outline,
            _generate_initial_content=mock_writer_content,
            _review_and_refine_content=mock_writer_content,
            _optimize_for_seo=mock_seo_optimize,
            _enhance_with_code_examples=mock_code_enhance
        ):
            result = await orchestrator.generate_blog(topic="Test Communication Flow")
        
        assert result.success is True
        assert len(conversation_log) >= 4  # At least planner, writer, critic, seo
        
        # Verify agent sequence
        agent_sequence = [entry["agent"] for entry in conversation_log]
        assert "ContentPlanner" in agent_sequence
        assert "Writer" in agent_sequence
        assert "Critic" in agent_sequence
        assert "SEOAgent" in agent_sequence
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, orchestrator):
        """Test error recovery when individual agents fail."""
        failure_count = {"planner": 0, "writer": 0}
        
        async def flaky_planner(*args, **kwargs):
            failure_count["planner"] += 1
            if failure_count["planner"] <= 2:  # Fail first 2 times
                raise Exception("Temporary planner failure")
            return {
                "title": "Recovered Blog",
                "sections": [{"title": "Section 1", "key_points": ["Point 1"], "word_count": 500}],
                "target_keywords": ["recovery"],
                "estimated_word_count": 500
            }
        
        async def stable_writer(*args, **kwargs):
            return {
                "title": "Recovered Blog",
                "content": "# Recovered Blog\n\nContent after recovery...",
                "word_count": 490,
                "code_blocks": [],
                "metadata": {}
            }
        
        with patch.multiple(
            orchestrator,
            _create_content_outline=flaky_planner,
            _generate_initial_content=stable_writer,
            _review_and_refine_content=stable_writer,
            _optimize_for_seo=AsyncMock(return_value={}),
            _enhance_with_code_examples=AsyncMock(return_value=[])
        ):
            result = await orchestrator.generate_blog(topic="Error Recovery Test")
        
        assert result.success is True
        assert failure_count["planner"] == 3  # Failed 2 times, succeeded on 3rd
        assert "Recovered Blog" in result.content
    
    @pytest.mark.asyncio
    async def test_different_content_types(self, orchestrator):
        """Test blog generation for different types of content."""
        content_types = [
            {
                "topic": "Machine Learning Basics",
                "description": "Introduction to ML concepts",
                "expected_keywords": ["machine learning", "algorithm", "model"]
            },
            {
                "topic": "Web Development with FastAPI",
                "description": "Building REST APIs",
                "expected_keywords": ["fastapi", "api", "web", "python"]
            },
            {
                "topic": "Database Design Principles",
                "description": "Relational database fundamentals",
                "expected_keywords": ["database", "sql", "design", "relationships"]
            }
        ]
        
        for content_type in content_types:
            # Mock appropriate responses for each content type
            mock_outline = {
                "title": f"Complete Guide to {content_type['topic']}",
                "sections": [
                    {"title": "Introduction", "key_points": ["Overview"], "word_count": 300},
                    {"title": "Core Concepts", "key_points": ["Fundamentals"], "word_count": 700},
                    {"title": "Practical Examples", "key_points": ["Implementation"], "word_count": 500}
                ],
                "target_keywords": content_type["expected_keywords"],
                "estimated_word_count": 1500
            }
            
            mock_content = {
                "title": mock_outline["title"],
                "content": f"# {mock_outline['title']}\n\nContent about {content_type['topic']}...",
                "word_count": 1480,
                "code_blocks": [],
                "metadata": {}
            }
            
            with patch.multiple(
                orchestrator,
                _create_content_outline=AsyncMock(return_value=mock_outline),
                _generate_initial_content=AsyncMock(return_value=mock_content),
                _review_and_refine_content=AsyncMock(return_value=mock_content),
                _optimize_for_seo=AsyncMock(return_value=mock_content),
                _enhance_with_code_examples=AsyncMock(return_value=[])
            ):
                result = await orchestrator.generate_blog(
                    topic=content_type["topic"],
                    description=content_type["description"]
                )
            
            assert result.success is True
            assert content_type["topic"] in result.content
            
            # Check that appropriate keywords appear in metadata
            for keyword in content_type["expected_keywords"]:
                assert keyword.lower() in str(result.metadata.get("target_keywords", [])).lower()
    
    @pytest.mark.asyncio
    async def test_performance_and_timing(self, orchestrator):
        """Test performance characteristics and timing."""
        start_time = datetime.now()
        
        # Mock fast responses
        with patch.multiple(
            orchestrator,
            _create_content_outline=AsyncMock(return_value={
                "title": "Performance Test Blog",
                "sections": [{"title": "Section", "key_points": ["Point"], "word_count": 500}],
                "target_keywords": ["performance"],
                "estimated_word_count": 500
            }),
            _generate_initial_content=AsyncMock(return_value={
                "title": "Performance Test Blog",
                "content": "# Performance Test\n\nContent...",
                "word_count": 480,
                "code_blocks": [],
                "metadata": {}
            }),
            _review_and_refine_content=AsyncMock(return_value={
                "title": "Performance Test Blog",
                "content": "# Performance Test\n\nContent...",
                "word_count": 480,
                "code_blocks": [],
                "metadata": {}
            }),
            _optimize_for_seo=AsyncMock(return_value={}),
            _enhance_with_code_examples=AsyncMock(return_value=[])
        ):
            result = await orchestrator.generate_blog(topic="Performance Test")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        assert result.success is True
        assert result.generation_time_seconds > 0
        assert execution_time < 10  # Should complete quickly with mocked responses
    
    @pytest.mark.asyncio
    async def test_concurrent_blog_generation(self, orchestrator):
        """Test generating multiple blogs concurrently."""
        topics = [
            "Python Testing Best Practices",
            "Docker Container Optimization",
            "RESTful API Design Patterns"
        ]
        
        # Mock responses for concurrent execution
        def create_mock_response(topic):
            return {
                "outline": {
                    "title": f"Guide to {topic}",
                    "sections": [{"title": "Section", "key_points": ["Point"], "word_count": 500}],
                    "target_keywords": [topic.split()[0].lower()],
                    "estimated_word_count": 500
                },
                "content": {
                    "title": f"Guide to {topic}",
                    "content": f"# Guide to {topic}\n\nContent about {topic}...",
                    "word_count": 480,
                    "code_blocks": [],
                    "metadata": {}
                }
            }
        
        async def mock_outline_for_topic(blog_input):
            topic = blog_input.topic
            return create_mock_response(topic)["outline"]
        
        async def mock_content_for_topic(*args, **kwargs):
            # Extract topic from first argument (outline or blog_input)
            first_arg = args[0]
            if hasattr(first_arg, 'topic'):
                topic = first_arg.topic
            elif hasattr(first_arg, 'title'):
                # Extract from title
                topic = first_arg.title.replace("Guide to ", "")
            else:
                topic = "Generic Topic"
            return create_mock_response(topic)["content"]
        
        with patch.multiple(
            orchestrator,
            _create_content_outline=mock_outline_for_topic,
            _generate_initial_content=mock_content_for_topic,
            _review_and_refine_content=mock_content_for_topic,
            _optimize_for_seo=AsyncMock(return_value={}),
            _enhance_with_code_examples=AsyncMock(return_value=[])
        ):
            # Generate blogs concurrently
            tasks = [
                orchestrator.generate_blog(topic=topic, description=f"Guide about {topic}")
                for topic in topics
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert result.success is True
            assert topics[i] in result.content
    
    @pytest.mark.asyncio
    async def test_memory_and_resource_cleanup(self, orchestrator):
        """Test that resources are properly cleaned up after generation."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate multiple blogs to test memory cleanup
        for i in range(3):
            with patch.multiple(
                orchestrator,
                _create_content_outline=AsyncMock(return_value={
                    "title": f"Memory Test Blog {i}",
                    "sections": [{"title": "Section", "key_points": ["Point"], "word_count": 500}],
                    "target_keywords": ["memory"],
                    "estimated_word_count": 500
                }),
                _generate_initial_content=AsyncMock(return_value={
                    "title": f"Memory Test Blog {i}",
                    "content": f"# Memory Test Blog {i}\n\n" + "Content... " * 1000,
                    "word_count": 5000,
                    "code_blocks": [],
                    "metadata": {}
                }),
                _review_and_refine_content=AsyncMock(return_value={
                    "title": f"Memory Test Blog {i}",
                    "content": f"# Memory Test Blog {i}\n\n" + "Content... " * 1000,
                    "word_count": 5000,
                    "code_blocks": [],
                    "metadata": {}
                }),
                _optimize_for_seo=AsyncMock(return_value={}),
                _enhance_with_code_examples=AsyncMock(return_value=[])
            ):
                result = await orchestrator.generate_blog(topic=f"Memory Test {i}")
                assert result.success is True
                
                # Force garbage collection
                del result
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"