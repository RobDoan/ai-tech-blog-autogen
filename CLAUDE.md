# CLAUDE.md

Build a comprehensive, production-ready automated tech blog system that leverages Microsoft AutoGen for multi-agent content creation, with robust infrastructure, content management, and deployment capabilities.

## Development Commands

This project uses `uv` as the package manager. Key commands:

- `uv run python main.py` - Run the main application
- `uv install` - Install dependencies from pyproject.toml
- `uv sync` - Sync dependencies and update lockfile
- `uv sync --group dev` - Install development dependencies including linting tools
- `uv run ruff check src/` - Run linter on source code
- `uv run ruff format src/` - Format source code
- `uv run ruff check --fix src/` - Run linter and auto-fix issues

## Git Hooks

The project includes a pre-commit hook that automatically checks code quality before allowing commits:

**What the hook checks:**
- Ruff linting (code quality and style)
- Code formatting with ruff
- Debugging statements (pdb, breakpoint)
- TODO/FIXME comments (warning only)

**Hook behavior:**
- Blocks commits if linting or formatting fails
- Provides helpful error messages with fix commands
- Only checks staged Python files for efficiency

**If commit is blocked:**
```bash
# Fix linting issues
uv run ruff check --fix src/

# Fix formatting issues  
uv run ruff format src/

# Stage the fixed files
git add src/

# Try commit again
git commit -m "your message"
```

**To bypass hook temporarily (not recommended):**
```bash
git commit --no-verify -m "your message"
```

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