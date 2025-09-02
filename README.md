# Automated Blog Generation System

A comprehensive multi-agent blog writing system powered by Microsoft AutoGen and OpenAI. This system leverages specialized AI agents to create high-quality, SEO-optimized technical blog content through collaborative workflows.

## 🌟 Features

- **Multi-Agent Collaboration**: Five specialized agents work together to create comprehensive blog content
- **Content Planning**: Strategic outline creation with audience analysis and structure planning
- **Expert Writing**: Technical writing with consistent tone and markdown formatting
- **Editorial Review**: Quality assurance with constructive feedback and iterative improvement
- **SEO Optimization**: Keyword research, content optimization, and meta description generation
- **Code Integration**: Automatic code example generation with proper formatting and explanations
- **Flexible Configuration**: Customizable workflow parameters and agent behavior
- **Error Recovery**: Graceful failure handling with partial content preservation

## 🤖 Agent Architecture

### ContentPlannerAgent
- Creates structured blog outlines from topics and context
- Analyzes target audience requirements
- Plans section flow and key points
- Identifies opportunities for code examples

### WriterAgent  
- Generates comprehensive blog content from outlines
- Maintains consistent technical writing style
- Formats content in proper markdown
- Incorporates feedback and revisions

### CriticAgent
- Reviews content quality, structure, and clarity
- Provides specific, actionable feedback
- Makes approval decisions based on quality thresholds
- Suggests prioritized improvements

### SEOAgent
- Conducts keyword research and analysis
- Optimizes titles and meta descriptions
- Integrates keywords naturally throughout content
- Provides SEO scoring and recommendations

### CodeAgent
- Identifies opportunities for code examples
- Generates clean, well-commented code snippets
- Creates practical, working examples
- Explains code functionality and usage

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Automated_Blog
```

2. Install dependencies:
```bash
uv install
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

Generate a blog post from command line:

```bash
# Basic blog generation
uv run python -m src.autogen_blog.multi_agent_blog_writer "Introduction to FastAPI"

# With additional context and customization
uv run python -m src.autogen_blog.multi_agent_blog_writer "Docker Best Practices" \
  --description "Focus on security and performance optimization" \
  --audience advanced \
  --length 2000 \
  --output docker_best_practices.md

# With book reference
uv run python -m src.autogen_blog.multi_agent_blog_writer "Python Async Programming" \
  --book-reference "Fluent Python by Luciano Ramalho" \
  --output async_python.md
```

### Programmatic Usage

```python
import asyncio
from src.autogen_blog import BlogWriterOrchestrator, AgentConfig, WorkflowConfig

async def generate_blog():
    # Configure agents
    agent_config = AgentConfig(
        openai_api_key="your-api-key",
        model="gpt-4",
        temperature=0.7
    )
    
    # Configure workflow
    workflow_config = WorkflowConfig(
        max_iterations=3,
        enable_code_agent=True,
        enable_seo_agent=True,
        quality_threshold=7.0
    )
    
    # Create orchestrator
    orchestrator = BlogWriterOrchestrator(agent_config, workflow_config)
    
    # Generate blog post
    result = await orchestrator.generate_blog(
        topic="Building REST APIs with FastAPI",
        description="Comprehensive guide for beginners"
    )
    
    if result.success:
        print(f"Generated {result.metadata['word_count']} words")
        print(result.content)
    else:
        print(f"Generation failed: {result.error_message}")

# Run the generation
asyncio.run(generate_blog())
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL` | Model to use | `gpt-4` |
| `OPENAI_TEMPERATURE` | Temperature setting | `0.7` |
| `OPENAI_MAX_TOKENS` | Max tokens per response | `4000` |
| `MAX_ITERATIONS` | Max refinement iterations | `3` |
| `ENABLE_CODE_AGENT` | Enable code examples | `true` |
| `ENABLE_SEO_AGENT` | Enable SEO optimization | `true` |
| `QUALITY_THRESHOLD` | Minimum quality score | `7.0` |

### Configuration Check

Verify your configuration:

```bash
uv run python -m src.autogen_blog.multi_agent_blog_writer --config-check
```

## 📋 Command Line Options

```bash
usage: multi_agent_blog_writer.py [-h] [-d DESCRIPTION] [-b BOOK_REFERENCE] 
                                  [-a {beginner,intermediate,advanced,expert}]
                                  [-l LENGTH] [-o OUTPUT] [-v] [--config-check] 
                                  topic

positional arguments:
  topic                 Main topic for the blog post

optional arguments:
  -h, --help            show this help message and exit
  -d, --description     Additional context or description
  -b, --book-reference  Reference book or source material
  -a, --audience        Target audience level (default: intermediate)
  -l, --length          Preferred word count (default: 1500)
  -o, --output          Output file path (default: display to stdout)
  -v, --verbose         Enable verbose logging
  --config-check        Check configuration and exit
```

## 🏗️ Project Structure

```
src/autogen_blog/
├── multi_agent_models.py          # Data models and configuration
├── base_agent.py                  # Base agent infrastructure
├── content_planner_agent.py       # Content planning agent
├── writer_agent.py                # Writing agent
├── critic_agent.py                # Review and feedback agent
├── seo_agent.py                   # SEO optimization agent
├── code_agent.py                  # Code example generation agent
├── blog_writer_orchestrator.py    # Main orchestration logic
├── multi_agent_blog_writer.py     # CLI interface
└── __init__.py                    # Package initialization
```

## 📚 Documentation

- **[Technical Implementation Guide](TECHNICAL_IMPLEMENTATION.md)** - Detailed technical architecture and specification mapping
- **[Requirements Document](.kiro/specs/multi-agent-blog-writer/requirements.md)** - Original system requirements
- **[Design Document](.kiro/specs/multi-agent-blog-writer/design.md)** - System design and architecture
- **[Implementation Tasks](.kiro/specs/multi-agent-blog-writer/tasks.md)** - Development task tracking

## 🔌 API Reference

### Core Classes

#### BlogWriterOrchestrator
Main orchestration class that coordinates all agents.

```python
orchestrator = BlogWriterOrchestrator(agent_config, workflow_config)
result = await orchestrator.generate_blog(topic, description, book_reference)
```

#### Configuration Classes

```python
# Agent configuration
agent_config = AgentConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=4000,
    openai_api_key="your-key",
    timeout_seconds=120
)

# Workflow configuration  
workflow_config = WorkflowConfig(
    max_iterations=3,
    enable_code_agent=True,
    enable_seo_agent=True,
    quality_threshold=7.0,
    parallel_processing=False
)
```

#### Data Models

```python
# Input data
blog_input = BlogInput(
    topic="Your Topic",
    description="Optional context",
    target_audience=TargetAudience.INTERMEDIATE,
    preferred_length=1500
)

# Result structure
class BlogResult:
    content: str                    # Generated markdown
    metadata: Dict[str, Any]        # Blog metadata
    generation_log: List[AgentMessage]  # Conversation log
    success: bool                   # Success status
    error_message: Optional[str]    # Error details
    generation_time_seconds: float  # Generation time
```

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
uv add --group test pytest pytest-asyncio pytest-mock

# Run tests
uv run pytest tests/
```

### Development Dependencies

```bash
uv add --group dev debugpy black isort mypy
```

## 📝 Output Examples

The system generates comprehensive blog posts with:

- **Structured Content**: Logical flow with clear sections and headers
- **SEO Optimization**: Optimized titles, meta descriptions, and keyword integration
- **Code Examples**: Practical, well-commented code snippets (for technical topics)
- **Professional Formatting**: Clean markdown with proper structure
- **Quality Assurance**: Content reviewed and refined for clarity and value

### Sample Output Structure:

```markdown
# Optimized Blog Title with Keywords

## Introduction
Engaging introduction that hooks readers and establishes value...

## Section 1: Core Concept
Detailed explanation with practical examples...

```python
# Well-commented code example
def example_function():
    """Clear docstring explaining the function."""
    return "practical example"
```

## Section 2: Advanced Topics
Building on previous concepts...

## Conclusion
Reinforces key points and provides actionable next steps...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the inline code documentation and examples
- **Issues**: Report bugs and request features via GitHub Issues
- **Configuration**: Use `--config-check` to validate your setup

## 🚧 Roadmap

- [ ] Web interface for blog generation
- [ ] Integration with popular CMS platforms
- [ ] Custom agent fine-tuning
- [ ] Batch processing capabilities
- [ ] Advanced analytics and metrics
- [ ] Multi-language support
