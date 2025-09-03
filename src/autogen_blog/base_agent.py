"""
Base agent infrastructure for the Multi-Agent Blog Writer system.

This module provides the foundation classes and utilities for all specialized agents
in the blog generation workflow, including error handling, retry logic, and
structured message parsing.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .multi_agent_models import (
    AgentCommunicationError,
    AgentConfig,
    AgentMessage,
    BlogGenerationError,
    MessageType,
)


class BaseAgent(ABC):
    """
    Base class for all specialized agents in the blog generation workflow.

    Provides common functionality including:
    - OpenAI model client configuration
    - Error handling and retry logic
    - Message parsing utilities
    - Structured response handling
    """

    def __init__(self, name: str, config: AgentConfig):
        """Initialize base agent with configuration."""
        self.name = name
        self.config = config
        self.logger = self._setup_logger()

        # Initialize OpenAI model client
        self.model = self._create_model_client()

        # Create AutoGen AssistantAgent
        self.agent = AssistantAgent(
            name=self.name,
            model_client=self.model,
            system_message=self._get_system_message(),
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for this agent."""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"%(asctime)s - {self.name} - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_model_client(self) -> OpenAIChatCompletionClient:
        """Create and configure OpenAI model client."""
        return OpenAIChatCompletionClient(
            model=self.config.model,
            api_key=self.config.openai_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @abstractmethod
    def _get_system_message(self) -> str:
        """Get the system message that defines this agent's role and behavior."""
        pass

    async def query_agent(
        self,
        prompt: str,
        message_type: MessageType = MessageType.CONTENT,
        max_retries: int | None = None,
    ) -> AgentMessage:
        """
        Query the agent with retry logic and structured response handling.

        Args:
            prompt: The prompt to send to the agent
            message_type: Type of message being sent
            max_retries: Maximum number of retry attempts (uses config default if None)

        Returns:
            AgentMessage with the agent's response

        Raises:
            AgentCommunicationError: If agent fails to respond after all retries
        """
        max_retries = max_retries or self.config.timeout_seconds // 10

        for attempt in range(max_retries + 1):
            try:
                self.logger.info(
                    f"Sending {message_type.value} query (attempt {attempt + 1})"
                )

                # Create text message for AutoGen
                message = TextMessage(content=prompt, source="user")

                # Query the agent with timeout
                response = await asyncio.wait_for(
                    self.agent.on_messages([message], cancellation_token=None),
                    timeout=self.config.timeout_seconds,
                )

                # Extract response content
                if response.chat_message and response.chat_message.content:
                    content = response.chat_message.content.strip()

                    self.logger.info(f"Received response of length {len(content)}")

                    return AgentMessage(
                        agent_name=self.name,
                        message_type=message_type,
                        content=content,
                        timestamp=datetime.now(),
                        metadata={"attempt": attempt + 1, "success": True},
                    )
                else:
                    raise AgentCommunicationError("Agent returned empty response")

            except TimeoutError as e:
                self.logger.warning(f"Attempt {attempt + 1} timed out: {e}")
                if attempt == max_retries:
                    raise AgentCommunicationError(
                        f"Agent query timed out after {max_retries + 1} attempts"
                    )

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise AgentCommunicationError(
                        f"Agent query failed after {max_retries + 1} attempts: {e}"
                    )

            # Exponential backoff between retries
            if attempt < max_retries:
                wait_time = 2**attempt
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        # This should never be reached due to the exception handling above
        raise AgentCommunicationError("Unexpected error in query_agent")

    def parse_json_response(self, response: str) -> dict[str, Any] | None:
        """
        Parse JSON response from agent, handling common formatting issues.

        Args:
            response: Raw response string from agent

        Returns:
            Parsed JSON as dictionary, or None if parsing fails
        """
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    json_str = response[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

            # Try to extract JSON from code blocks without language specification
            if "```" in response:
                lines = response.split("\n")
                in_code_block = False
                json_lines = []

                for line in lines:
                    if line.strip() == "```":
                        if in_code_block:
                            # End of code block, try to parse accumulated JSON
                            json_str = "\n".join(json_lines).strip()
                            if json_str.startswith("{") or json_str.startswith("["):
                                try:
                                    return json.loads(json_str)
                                except json.JSONDecodeError:
                                    pass
                            json_lines = []
                        in_code_block = not in_code_block
                    elif in_code_block:
                        json_lines.append(line)

            self.logger.warning(f"Failed to parse JSON response: {response[:200]}...")
            return None

    def extract_structured_data(
        self, response: str, expected_keys: list[str]
    ) -> dict[str, Any]:
        """
        Extract structured data from agent response, with fallback parsing.

        Args:
            response: Raw response from agent
            expected_keys: Keys expected in the structured response

        Returns:
            Dictionary with extracted data, using fallbacks for missing keys
        """
        # Try JSON parsing first
        parsed = self.parse_json_response(response)
        if parsed and all(key in parsed for key in expected_keys):
            return parsed

        # Fallback: text parsing for key-value pairs
        result = {}
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if ":" in line:
                for key in expected_keys:
                    if line.lower().startswith(key.lower() + ":"):
                        value = line.split(":", 1)[1].strip()
                        # Clean up common formatting
                        value = value.strip("\"'`")
                        result[key] = value
                        break

        # Ensure all expected keys have values (use empty string as fallback)
        for key in expected_keys:
            if key not in result:
                result[key] = ""
                self.logger.warning(
                    f"Missing key '{key}' in agent response, using empty fallback"
                )

        return result

    async def validate_response(
        self, response: AgentMessage, validation_criteria: dict[str, Any]
    ) -> bool:
        """
        Validate agent response against criteria.

        Args:
            response: Agent response to validate
            validation_criteria: Criteria for validation

        Returns:
            True if response is valid, False otherwise
        """
        content = response.content

        # Check minimum length
        if "min_length" in validation_criteria:
            if len(content) < validation_criteria["min_length"]:
                self.logger.warning(
                    f"Response too short: {len(content)} < {validation_criteria['min_length']}"
                )
                return False

        # Check for required keywords
        if "required_keywords" in validation_criteria:
            content_lower = content.lower()
            for keyword in validation_criteria["required_keywords"]:
                if keyword.lower() not in content_lower:
                    self.logger.warning(f"Missing required keyword: {keyword}")
                    return False

        # Check JSON structure if expected
        if validation_criteria.get("expect_json", False):
            parsed = self.parse_json_response(content)
            if not parsed:
                self.logger.warning("Expected JSON response but parsing failed")
                return False

            if "required_json_keys" in validation_criteria:
                for key in validation_criteria["required_json_keys"]:
                    if key not in parsed:
                        self.logger.warning(f"Missing required JSON key: {key}")
                        return False

        return True

    def create_error_message(self, error: Exception, context: str = "") -> AgentMessage:
        """
        Create a standardized error message.

        Args:
            error: The exception that occurred
            context: Additional context about the error

        Returns:
            AgentMessage representing the error
        """
        return AgentMessage(
            agent_name=self.name,
            message_type=MessageType.ERROR,
            content=f"Error in {context}: {str(error)}",
            timestamp=datetime.now(),
            metadata={
                "error_type": type(error).__name__,
                "context": context,
                "success": False,
            },
        )


class MessageParser:
    """Utility class for parsing and formatting messages between agents."""

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with provided arguments.

        Args:
            template: Prompt template with placeholder variables
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise BlogGenerationError(f"Missing template variable: {e}")

    @staticmethod
    def extract_sections(content: str, section_markers: list[str]) -> dict[str, str]:
        """
        Extract sections from content based on markers.

        Args:
            content: Content to parse
            section_markers: List of section headers to look for

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = None
        current_content = []

        for line in content.split("\n"):
            line_lower = line.strip().lower()

            # Check if this line is a section marker
            for marker in section_markers:
                if marker.lower() in line_lower:
                    # Save previous section if it exists
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()

                    # Start new section
                    current_section = marker
                    current_content = []
                    break
            else:
                # Add line to current section
                if current_section:
                    current_content.append(line)

        # Save the last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def clean_content(content: str) -> str:
        """
        Clean and normalize content from agent responses.

        Args:
            content: Raw content from agent

        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = "\n".join(line.strip() for line in content.split("\n"))

        # Remove multiple consecutive empty lines
        while "\n\n\n" in content:
            content = content.replace("\n\n\n", "\n\n")

        # Strip leading/trailing whitespace
        content = content.strip()

        return content


# Agent state management for workflow coordination
class AgentState:
    """Manages state and coordination between agents in the workflow."""

    def __init__(self):
        self.conversation_history: list[AgentMessage] = []
        self.current_step = 0
        self.workflow_data: dict[str, Any] = {}
        self.errors: list[AgentMessage] = []

    def add_message(self, message: AgentMessage):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

        # Track errors separately
        if message.message_type == MessageType.ERROR:
            self.errors.append(message)

    def get_messages_by_agent(self, agent_name: str) -> list[AgentMessage]:
        """Get all messages from a specific agent."""
        return [
            msg for msg in self.conversation_history if msg.agent_name == agent_name
        ]

    def get_messages_by_type(self, message_type: MessageType) -> list[AgentMessage]:
        """Get all messages of a specific type."""
        return [
            msg for msg in self.conversation_history if msg.message_type == message_type
        ]

    def get_latest_message_by_type(
        self, message_type: MessageType
    ) -> AgentMessage | None:
        """Get the most recent message of a specific type."""
        messages = self.get_messages_by_type(message_type)
        return messages[-1] if messages else None

    def set_workflow_data(self, key: str, value: Any):
        """Set data in the workflow context."""
        self.workflow_data[key] = value

    def get_workflow_data(self, key: str, default: Any = None) -> Any:
        """Get data from the workflow context."""
        return self.workflow_data.get(key, default)

    def has_errors(self) -> bool:
        """Check if there are any errors in the workflow."""
        return len(self.errors) > 0

    def get_error_summary(self) -> str:
        """Get a summary of all errors encountered."""
        if not self.errors:
            return "No errors"

        error_lines = []
        for error in self.errors:
            error_lines.append(f"{error.agent_name}: {error.content}")

        return "\n".join(error_lines)
