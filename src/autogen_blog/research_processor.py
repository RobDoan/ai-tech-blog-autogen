"""
Research Processing Foundation for Conversational Blog Writer.

This module provides file parsing utilities and research processing capabilities
for ingesting and synthesizing knowledge from various file formats.
"""

import asyncio
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .multi_agent_models import BlogGenerationError


class ResearchProcessingError(BlogGenerationError):
    """Raised when research processing fails."""

    pass


class UnsupportedFileFormatError(ResearchProcessingError):
    """Raised when a file format is not supported."""

    pass


@dataclass
class FileMetadata:
    """Metadata for processed files."""

    path: Path
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    file_type: str
    mime_type: str
    encoding: str = "utf-8"


class ResearchFile(BaseModel):
    """Represents a processed research file with extracted content."""

    path: Path = Field(..., description="File path")
    content: str = Field(..., description="Extracted text content")
    file_type: str = Field(..., description="File type (md, txt, json, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="File metadata")
    extracted_insights: list[str] = Field(
        default_factory=list, description="Key insights extracted"
    )
    processing_errors: list[str] = Field(
        default_factory=list, description="Errors during processing"
    )
    confidence_score: float = Field(
        1.0, description="Confidence in extraction quality", ge=0.0, le=1.0
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("path")
    def path_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        return v


class Insight(BaseModel):
    """Represents a key insight extracted from research materials."""

    content: str = Field(..., description="The insight content")
    source_file: str = Field(..., description="Source file path")
    confidence_score: float = Field(
        0.8, description="Confidence in this insight", ge=0.0, le=1.0
    )
    category: str = Field(
        "general", description="Category: problem, solution, technology, best_practice"
    )
    technical_concepts: list[str] = Field(
        default_factory=list, description="Technical concepts mentioned"
    )
    code_references: list[str] = Field(
        default_factory=list, description="Code snippets or references"
    )
    importance_score: float = Field(
        0.5, description="Relative importance", ge=0.0, le=1.0
    )


class KnowledgeBase(BaseModel):
    """Consolidated knowledge base from processed research."""

    insights: list[Insight] = Field(
        default_factory=list, description="Extracted insights"
    )
    technical_concepts: set[str] = Field(
        default_factory=set, description="All technical concepts found"
    )
    code_examples: list[dict[str, str]] = Field(
        default_factory=list, description="Code examples found"
    )
    references: list[str] = Field(
        default_factory=list, description="Source files referenced"
    )
    summary: str = Field("", description="Summary of the knowledge base")
    processing_timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class FileParser:
    """Base class for file format parsers."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.supported_extensions: set[str] = set()
        self.supported_mime_types: set[str] = set()

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file."""
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return extension in self.supported_extensions or (
            mime_type and mime_type in self.supported_mime_types
        )

    async def parse_file(self, file_path: Path) -> ResearchFile:
        """Parse a file and extract content. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement parse_file")

    def _get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Extract metadata from file."""
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return FileMetadata(
            path=file_path,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            file_type=file_path.suffix.lower()[1:] if file_path.suffix else "unknown",
            mime_type=mime_type or "application/octet-stream",
        )


class MarkdownParser(FileParser):
    """Parser for Markdown files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {".md", ".markdown", ".mdown", ".mkd"}
        self.supported_mime_types = {"text/markdown", "text/x-markdown"}

    async def parse_file(self, file_path: Path) -> ResearchFile:
        """Parse markdown file and extract content."""
        try:
            metadata = self._get_file_metadata(file_path)

            # Read file content
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            # Extract text content (removing markdown syntax for analysis)
            text_content = self._extract_text_from_markdown(raw_content)

            # Extract insights from headers and structure
            insights = self._extract_markdown_insights(raw_content, str(file_path))

            return ResearchFile(
                path=file_path,
                content=text_content,
                file_type="markdown",
                metadata={
                    "original_format": "markdown",
                    "headers_count": len(
                        re.findall(r"^#+\s+", raw_content, re.MULTILINE)
                    ),
                    "code_blocks_count": len(re.findall(r"```", raw_content)) // 2,
                    "links_count": len(re.findall(r"\[.*?\]\(.*?\)", raw_content)),
                    **metadata.__dict__,
                },
                extracted_insights=[insight.content for insight in insights],
                confidence_score=0.9,
            )

        except Exception as e:
            self.logger.error(f"Failed to parse markdown file {file_path}: {e}")
            return ResearchFile(
                path=file_path,
                content="",
                file_type="markdown",
                metadata={},
                processing_errors=[str(e)],
                confidence_score=0.0,
            )

    def _extract_text_from_markdown(self, content: str) -> str:
        """Extract plain text from markdown content."""
        # Remove code blocks first
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

        # Remove inline code
        content = re.sub(r"`[^`]+`", "", content)

        # Remove links but keep text
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)

        # Remove images
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

        # Remove emphasis markers
        content = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", content)

        # Convert headers to plain text
        content = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)

        # Remove list markers
        content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s+", "", content, flags=re.MULTILINE)

        # Clean up whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = content.strip()

        return content

    def _extract_markdown_insights(
        self, content: str, source_file: str
    ) -> list[Insight]:
        """Extract insights from markdown structure."""
        insights = []

        # Extract from headers
        headers = re.findall(r"^(#+)\s+(.+)$", content, re.MULTILINE)
        for level_markers, header_text in headers:
            level = len(level_markers)
            if level <= 3:  # Focus on main headers
                insights.append(
                    Insight(
                        content=f"Topic: {header_text}",
                        source_file=source_file,
                        category="topic",
                        confidence_score=0.8,
                        importance_score=max(0.3, 1.0 - (level - 1) * 0.2),
                    )
                )

        # Extract from code blocks with language specification
        code_blocks = re.findall(r"```(\w+)\n(.*?)\n```", content, re.DOTALL)
        for language, code in code_blocks:
            if code.strip():
                insights.append(
                    Insight(
                        content=f"Code example in {language}",
                        source_file=source_file,
                        category="technology",
                        technical_concepts=[language],
                        code_references=[code.strip()[:200]],  # First 200 chars
                        confidence_score=0.9,
                        importance_score=0.7,
                    )
                )

        return insights


class TextParser(FileParser):
    """Parser for plain text files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {".txt", ".text", ".log"}
        self.supported_mime_types = {"text/plain"}

    async def parse_file(self, file_path: Path) -> ResearchFile:
        """Parse text file and extract content."""
        try:
            metadata = self._get_file_metadata(file_path)

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract basic insights from text
            insights = self._extract_text_insights(content, str(file_path))

            return ResearchFile(
                path=file_path,
                content=content,
                file_type="text",
                metadata={
                    "line_count": len(content.split("\n")),
                    "char_count": len(content),
                    **metadata.__dict__,
                },
                extracted_insights=[insight.content for insight in insights],
                confidence_score=0.7,
            )

        except Exception as e:
            self.logger.error(f"Failed to parse text file {file_path}: {e}")
            return ResearchFile(
                path=file_path,
                content="",
                file_type="text",
                metadata={},
                processing_errors=[str(e)],
                confidence_score=0.0,
            )

    def _extract_text_insights(self, content: str, source_file: str) -> list[Insight]:
        """Extract insights from plain text content."""
        insights = []

        # Look for technical terms
        technical_terms = re.findall(
            r"\b(?:API|SDK|JSON|XML|HTTP|HTTPS|REST|GraphQL|SQL|NoSQL|Docker|Kubernetes|React|Vue|Angular|Node\.js|Python|Java|JavaScript|TypeScript)\b",
            content,
            re.IGNORECASE,
        )

        if technical_terms:
            unique_terms = list(set(term.lower() for term in technical_terms))
            insights.append(
                Insight(
                    content=f"Technical concepts: {', '.join(unique_terms)}",
                    source_file=source_file,
                    category="technology",
                    technical_concepts=unique_terms,
                    confidence_score=0.6,
                    importance_score=0.5,
                )
            )

        return insights


class JSONParser(FileParser):
    """Parser for JSON files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {".json", ".jsonl"}
        self.supported_mime_types = {"application/json", "application/jsonl"}

    async def parse_file(self, file_path: Path) -> ResearchFile:
        """Parse JSON file and extract content."""
        try:
            metadata = self._get_file_metadata(file_path)

            with open(file_path, encoding="utf-8") as f:
                if file_path.suffix.lower() == ".jsonl":
                    # Handle JSON Lines format
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)

            # Convert JSON to readable text
            text_content = self._json_to_text(data)
            insights = self._extract_json_insights(data, str(file_path))

            return ResearchFile(
                path=file_path,
                content=text_content,
                file_type="json",
                metadata={
                    "structure": "list" if isinstance(data, list) else "object",
                    "items_count": len(data) if isinstance(data, (list, dict)) else 1,
                    **metadata.__dict__,
                },
                extracted_insights=[insight.content for insight in insights],
                confidence_score=0.8,
            )

        except Exception as e:
            self.logger.error(f"Failed to parse JSON file {file_path}: {e}")
            return ResearchFile(
                path=file_path,
                content="",
                file_type="json",
                metadata={},
                processing_errors=[str(e)],
                confidence_score=0.0,
            )

    def _json_to_text(self, data: dict | list | Any) -> str:
        """Convert JSON data to readable text."""
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (int, float, bool)):
                    text_parts.append(f"{key}: {value}")
                else:
                    text_parts.append(f"{key}: {type(value).__name__}")
            return "\n".join(text_parts)
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Extract common keys from objects
                if data:
                    keys = set()
                    for item in data[:5]:  # Sample first 5 items
                        if isinstance(item, dict):
                            keys.update(item.keys())
                    return f"List of objects with keys: {', '.join(sorted(keys))}"
            return f"List with {len(data)} items"
        else:
            return str(data)

    def _extract_json_insights(
        self, data: dict | list | Any, source_file: str
    ) -> list[Insight]:
        """Extract insights from JSON structure and content."""
        insights = []

        if isinstance(data, dict):
            # Extract from dictionary keys
            keys = list(data.keys())
            if keys:
                insights.append(
                    Insight(
                        content=f"Configuration/data keys: {', '.join(keys[:10])}",
                        source_file=source_file,
                        category="best_practice",
                        confidence_score=0.7,
                        importance_score=0.4,
                    )
                )

        elif isinstance(data, list) and data:
            insights.append(
                Insight(
                    content=f"Dataset with {len(data)} items",
                    source_file=source_file,
                    category="general",
                    confidence_score=0.8,
                    importance_score=0.5,
                )
            )

        return insights


class ResearchProcessor:
    """Main processor for research folder and files."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.parsers: list[FileParser] = [MarkdownParser(), TextParser(), JSONParser()]

        # Add advanced parsers if available
        try:
            from .advanced_file_parsers import (
                get_advanced_parsers,
                get_missing_dependencies,
            )

            advanced_parsers = get_advanced_parsers()
            self.parsers.extend(advanced_parsers)

            missing_deps = get_missing_dependencies()
            if missing_deps:
                self.logger.warning(
                    f"Some advanced file format support unavailable: {', '.join(missing_deps)}"
                )

            self.logger.info(f"Loaded {len(advanced_parsers)} advanced file parsers")

        except ImportError as e:
            self.logger.warning(f"Could not load advanced file parsers: {e}")

        self.max_file_size_mb = 10  # Max file size to process
        self.supported_extensions = set()

        # Collect all supported extensions
        for parser in self.parsers:
            self.supported_extensions.update(parser.supported_extensions)

    async def process_folder(
        self, folder_path: Path, recursive: bool = True
    ) -> KnowledgeBase:
        """
        Process all supported files in a folder and extract knowledge.

        Args:
            folder_path: Path to the research folder
            recursive: Whether to process subdirectories

        Returns:
            KnowledgeBase with consolidated knowledge

        Raises:
            ResearchProcessingError: If processing fails
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ResearchProcessingError(f"Invalid folder path: {folder_path}")

        self.logger.info(f"Processing research folder: {folder_path}")

        # Find all supported files
        files = self._find_supported_files(folder_path, recursive)

        if not files:
            self.logger.warning(f"No supported files found in {folder_path}")
            return KnowledgeBase()

        self.logger.info(f"Found {len(files)} supported files to process")

        # Process files concurrently
        processed_files = await self._process_files_concurrently(files)

        # Synthesize knowledge
        knowledge_base = await self._synthesize_knowledge(processed_files)

        self.logger.info(
            f"Generated knowledge base with {len(knowledge_base.insights)} insights"
        )
        return knowledge_base

    def _find_supported_files(self, folder_path: Path, recursive: bool) -> list[Path]:
        """Find all supported files in the folder."""
        files = []

        pattern = "**/*" if recursive else "*"

        for file_path in folder_path.glob(pattern):
            if file_path.is_file() and self._is_supported_file(file_path):
                # Check file size
                if file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                    self.logger.warning(f"Skipping large file: {file_path}")
                    continue

                files.append(file_path)

        return sorted(files)  # Sort for consistent processing order

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is supported by any parser."""
        return any(parser.can_parse(file_path) for parser in self.parsers)

    def _get_parser_for_file(self, file_path: Path) -> FileParser | None:
        """Get the appropriate parser for a file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    async def _process_files_concurrently(
        self, files: list[Path]
    ) -> list[ResearchFile]:
        """Process multiple files concurrently."""
        tasks = []

        for file_path in files:
            parser = self._get_parser_for_file(file_path)
            if parser:
                task = parser.parse_file(file_path)
                tasks.append(task)

        # Process with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent file operations

        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        guarded_tasks = [process_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*guarded_tasks, return_exceptions=True)

        # Filter out exceptions and failed results
        processed_files = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"File processing failed: {result}")
            elif isinstance(result, ResearchFile) and result.confidence_score > 0:
                processed_files.append(result)

        return processed_files

    async def _synthesize_knowledge(
        self, processed_files: list[ResearchFile]
    ) -> KnowledgeBase:
        """Synthesize knowledge from processed files."""
        all_insights = []
        all_technical_concepts = set()
        code_examples = []
        references = []

        for file_data in processed_files:
            references.append(str(file_data.path))

            # Convert extracted insights to Insight objects
            for insight_text in file_data.extracted_insights:
                insight = Insight(
                    content=insight_text,
                    source_file=str(file_data.path),
                    confidence_score=file_data.confidence_score,
                    category="general",
                )
                all_insights.append(insight)

            # Collect technical concepts from metadata
            if "technical_concepts" in file_data.metadata:
                concepts = file_data.metadata["technical_concepts"]
                if isinstance(concepts, list):
                    all_technical_concepts.update(concepts)

            # Collect code examples from metadata
            if (
                "code_blocks_count" in file_data.metadata
                and file_data.metadata["code_blocks_count"] > 0
            ):
                # This is a simplified approach - in practice, you'd extract actual code blocks
                code_examples.append(
                    {
                        "source": str(file_data.path),
                        "type": file_data.file_type,
                        "count": file_data.metadata["code_blocks_count"],
                    }
                )

        # Create summary
        summary = self._generate_knowledge_summary(processed_files, all_insights)

        return KnowledgeBase(
            insights=all_insights,
            technical_concepts=all_technical_concepts,
            code_examples=code_examples,
            references=references,
            summary=summary,
        )

    def _generate_knowledge_summary(
        self, processed_files: list[ResearchFile], insights: list[Insight]
    ) -> str:
        """Generate a summary of the processed knowledge."""
        file_types = {}
        total_insights = len(insights)

        for file_data in processed_files:
            file_type = file_data.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1

        file_summary = ", ".join(
            [f"{count} {ftype} files" for ftype, count in file_types.items()]
        )

        return f"Processed {len(processed_files)} files ({file_summary}) and extracted {total_insights} insights."

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return sorted(list(self.supported_extensions))
