# CLAUDE.md

Build a comprehensive, production-ready automated tech blog system that leverages Microsoft AutoGen for multi-agent content creation, with robust infrastructure, content management, and deployment capabilities.

## Development Commands

This project uses `uv` as the package manager. Key commands:

- `uv run python main.py` - Run the main application
- `uv install` - Install dependencies from pyproject.toml
- `uv sync` - Sync dependencies and update lockfile

## Project Architecture

This is an automated blog generation system built with Python using the AutoGen framework for AI agents.

### Core Structure

```
src/autogen_blog/
├── agents.py           # Agent classes and orchestration logic
├── content_generator.py # Blog content generation functionality
├── trend_spotter.py    # Trend detection and analysis
├── tools.py            # Utility functions and tools
└── visuals.py          # Visualization and graphics generation
```

The application follows a modular AI agent architecture where:
- **Agents** (`agents.py`) coordinate the overall workflow
- **Trend Spotter** (`trend_spotter.py`) identifies trending topics
- **Content Generator** (`content_generator.py`) creates blog posts
- **Visuals** (`visuals.py`) generates accompanying graphics
- **Tools** (`tools.py`) provides shared utilities

### Key Dependencies

- `autogen-agentchat>=0.7.1` - Multi-agent conversation framework
- `autogen-ext[openai]>=0.7.1` - OpenAI integration for AutoGen
- `python-dotenv>=1.1.1` - Environment variable management

The entry point is `main.py` which imports and orchestrates the core modules.

## Development Notes

- Project is in early stages with placeholder module files
- Uses Python 3.12+ requirement
- No testing framework currently configured
- Environment variables likely needed for OpenAI API access