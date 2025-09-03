"""
AutoGen Blog Generation System

This module provides automated blog generation capabilities using AI agents.
Includes both the original single-agent system and the new multi-agent blog writer.
"""

from .blog_writer_orchestrator import BlogWriterOrchestrator
from .multi_agent_models import (
    AgentConfig,
    BlogContent,
    BlogInput,
    BlogResult,
    CodeBlock,
    ContentOutline,
    Section,
    TargetAudience,
    WorkflowConfig,
)

__version__ = "0.2.0"
__all__ = [
    "BlogInput",
    "ContentOutline",
    "Section",
    "BlogContent",
    "CodeBlock",
    "BlogResult",
    "AgentConfig",
    "WorkflowConfig",
    "TargetAudience",
    "BlogWriterOrchestrator",
]
# Test comment
