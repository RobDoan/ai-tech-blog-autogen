"""
Data models for the Multi-Agent Blog Writer system.

This module defines Pydantic models for structured data exchange between agents,
configuration classes, and result objects used throughout the blog generation workflow.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TargetAudience(str, Enum):
    """Target audience levels for blog content."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class MessageType(str, Enum):
    """Types of messages exchanged between agents."""

    OUTLINE = "outline"
    CONTENT = "content"
    FEEDBACK = "feedback"
    CODE = "code"
    SEO_ANALYSIS = "seo_analysis"
    APPROVAL = "approval"
    ERROR = "error"


class BlogInput(BaseModel):
    """Input data for blog generation workflow."""

    topic: str = Field(..., description="Main topic for the blog post")
    description: str | None = Field(
        None, description="Additional context or description"
    )
    book_reference: str | None = Field(
        None, description="Reference book or source material"
    )
    target_audience: TargetAudience = Field(
        TargetAudience.INTERMEDIATE, description="Target audience level"
    )
    preferred_length: int = Field(
        1500, description="Preferred word count for the blog post", ge=500, le=5000
    )

    @field_validator("topic")
    def topic_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        return v.strip()


class Section(BaseModel):
    """Individual section of a blog post outline."""

    heading: str = Field(..., description="Section heading")
    key_points: list[str] = Field(
        default_factory=list, description="Key points to cover in this section"
    )
    code_examples_needed: bool = Field(
        False, description="Whether this section needs code examples"
    )
    estimated_words: int = Field(
        200, description="Estimated word count for this section", ge=50
    )


class ContentOutline(BaseModel):
    """Structured outline for blog content."""

    title: str = Field(..., description="Blog post title")
    introduction: str = Field(..., description="Introduction summary")
    sections: list[Section] = Field(..., description="Main sections of the blog post")
    conclusion: str = Field(..., description="Conclusion summary")
    estimated_word_count: int = Field(0, description="Total estimated word count")
    target_keywords: list[str] = Field(
        default_factory=list, description="Target SEO keywords"
    )

    @field_validator("sections")
    def must_have_sections(cls, v):
        if len(v) < 1:
            raise ValueError("Outline must have at least one section")
        return v

    def __post_init__(self):
        """Calculate total estimated word count after validation."""
        if self.estimated_word_count == 0:
            self.estimated_word_count = (
                sum(section.estimated_words for section in self.sections) + 300
            )  # intro + conclusion


class CodeBlock(BaseModel):
    """Code example with metadata."""

    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="Code content")
    explanation: str = Field(..., description="Explanation of the code")
    line_numbers: bool = Field(True, description="Whether to show line numbers")
    filename: str | None = Field(None, description="Optional filename for the code")


class ContentMetadata(BaseModel):
    """Metadata for blog content."""

    word_count: int = Field(0, description="Actual word count")
    reading_time_minutes: int = Field(0, description="Estimated reading time")
    seo_score: float | None = Field(None, description="SEO optimization score")
    keywords: list[str] = Field(
        default_factory=list, description="Keywords found in content"
    )
    meta_description: str | None = Field(None, description="SEO meta description")


class BlogContent(BaseModel):
    """Complete blog content with metadata."""

    title: str = Field(..., description="Final blog title")
    content: str = Field(..., description="Markdown formatted content")
    sections: list[str] = Field(..., description="Section headings")
    code_blocks: list[CodeBlock] = Field(
        default_factory=list, description="Code examples included"
    )
    metadata: ContentMetadata = Field(
        default_factory=ContentMetadata, description="Content metadata"
    )

    @field_validator("content")
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class AgentMessage(BaseModel):
    """Message exchanged between agents."""

    agent_name: str = Field(..., description="Name of the sending agent")
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Message timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )


class ReviewFeedback(BaseModel):
    """Feedback from the critic agent."""

    overall_score: float = Field(
        ..., description="Overall quality score (0-10)", ge=0, le=10
    )
    strengths: list[str] = Field(..., description="Identified strengths")
    improvements: list[str] = Field(..., description="Suggested improvements")
    approved: bool = Field(False, description="Whether content is approved")
    specific_feedback: dict[str, str] = Field(
        default_factory=dict, description="Section-specific feedback"
    )


class KeywordAnalysis(BaseModel):
    """SEO keyword analysis results."""

    primary_keywords: list[str] = Field(..., description="Primary target keywords")
    secondary_keywords: list[str] = Field(
        default_factory=list, description="Secondary keywords"
    )
    trending_keywords: list[str] = Field(
        default_factory=list, description="Currently trending keywords"
    )
    competition_level: dict[str, str] = Field(
        default_factory=dict, description="Competition analysis"
    )
    search_volume: dict[str, int] = Field(
        default_factory=dict, description="Estimated search volumes"
    )


class SEOOptimizedContent(BaseModel):
    """Content with SEO optimizations applied."""

    optimized_title: str = Field(..., description="SEO optimized title")
    meta_description: str = Field(..., description="SEO meta description")
    optimized_content: str = Field(..., description="Content with SEO improvements")
    keywords_used: list[str] = Field(..., description="Keywords incorporated")
    seo_score: float = Field(..., description="Estimated SEO score", ge=0, le=100)


class CodeOpportunity(BaseModel):
    """Opportunity to add code examples."""

    section_title: str = Field(..., description="Section where code is needed")
    description: str = Field(
        ..., description="Description of what code should demonstrate"
    )
    programming_language: str = Field(
        ..., description="Recommended programming language"
    )
    complexity_level: str = Field(
        "intermediate", description="Complexity level of the code"
    )


class CodeExample(BaseModel):
    """Generated code example."""

    opportunity: CodeOpportunity = Field(
        ..., description="Original opportunity this addresses"
    )
    code_block: CodeBlock = Field(..., description="Generated code")
    integration_note: str = Field(
        ..., description="Note on how to integrate into content"
    )


class BlogResult(BaseModel):
    """Final result of blog generation process."""

    content: str = Field(..., description="Final markdown content")
    metadata: dict[str, Any] = Field(..., description="Blog metadata")
    generation_log: list[AgentMessage] = Field(
        ..., description="Complete conversation log"
    )
    success: bool = Field(..., description="Whether generation was successful")
    error_message: str | None = Field(
        None, description="Error message if generation failed"
    )
    generation_time_seconds: float | None = Field(
        None, description="Total generation time"
    )

    @field_validator("content")
    def content_required_if_success(cls, v, info):
        if info.data.get("success") and not v.strip():
            raise ValueError("Content is required for successful generation")
        return v


class AgentConfig(BaseModel):
    """Configuration for individual agents."""

    model: str = Field("gpt-4", description="OpenAI model to use")
    temperature: float = Field(0.7, description="Model temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(
        4000, description="Maximum tokens per response", ge=100, le=8000
    )
    openai_api_key: str = Field(..., description="OpenAI API key")
    timeout_seconds: int = Field(
        60, description="Request timeout in seconds", ge=10, le=300
    )

    class Config:
        # Hide the API key in string representation
        json_encoders = {str: lambda v: "***" if "api_key" in str(v) else v}


class WorkflowConfig(BaseModel):
    """Configuration for the overall workflow."""

    max_iterations: int = Field(
        3, description="Maximum refinement iterations", ge=1, le=10
    )
    enable_code_agent: bool = Field(True, description="Whether to use code agent")
    enable_seo_agent: bool = Field(True, description="Whether to use SEO agent")
    output_format: str = Field("markdown", description="Output format")
    save_conversation_log: bool = Field(
        True, description="Whether to save agent conversations"
    )
    parallel_processing: bool = Field(
        False, description="Enable parallel agent processing where possible"
    )
    quality_threshold: float = Field(
        7.0, description="Minimum quality score to accept content", ge=0.0, le=10.0
    )


class BlogGenerationError(Exception):
    """Base exception for blog generation errors."""

    pass


class AgentCommunicationError(BlogGenerationError):
    """Raised when agents cannot communicate effectively."""

    pass


class ContentQualityError(BlogGenerationError):
    """Raised when content doesn't meet quality standards."""

    pass


class SEOServiceError(BlogGenerationError):
    """Raised when SEO services are unavailable."""

    pass


class ConfigurationError(BlogGenerationError):
    """Raised when configuration is invalid."""

    pass


# Utility functions for data validation and serialization
def validate_blog_input(data: dict) -> BlogInput:
    """Validate and create BlogInput from dictionary."""
    return BlogInput(**data)


def serialize_agent_conversation(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Serialize agent conversation for storage."""
    return [msg.dict() for msg in messages]


def deserialize_agent_conversation(data: list[dict[str, Any]]) -> list[AgentMessage]:
    """Deserialize agent conversation from storage."""
    return [AgentMessage(**msg_data) for msg_data in data]
