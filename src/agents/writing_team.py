# agents/writing_team.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import List, Dict, Optional
from datetime import datetime

# Import ContentPlan from content_planner
try:
    from .content_planner import ContentPlan
except ImportError:
    # For direct imports when testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agents.content_planner import ContentPlan

class WritingTeamOrchestrator:
    def __init__(self, config: Dict):
        self.config = config

        # Initialize OpenAI model for AutoGen 0.7
        self.model = OpenAIChatCompletionClient(
            model=config.get("model", "gpt-4"),
            api_key=config["openai_api_key"],
            temperature=config.get("temperature", 0.7)
        )

        # Initialize specialized agents
        self.researcher = self._create_researcher_agent()
        self.writer = self._create_writer_agent()
        self.code_specialist = self._create_code_specialist_agent()
        self.editor = self._create_editor_agent()

        # Create team for collaboration using RoundRobinGroupChat
        self.team = RoundRobinGroupChat(
            participants=[self.researcher, self.writer, self.code_specialist, self.editor],
            max_turns=20
        )

    def _create_researcher_agent(self) -> AssistantAgent:
        """Create research specialist agent"""
        return AssistantAgent(
            name="researcher",
            model_client=self.model,
            system_message="""
            You are a tech research specialist. Your role is to:
            1. Gather comprehensive information about the given topic
            2. Find relevant examples, case studies, and references
            3. Identify key concepts and technical details
            4. Provide accurate, up-to-date information
            5. Suggest additional angles or subtopics to explore

            Always verify information from multiple sources and provide citations.
            Focus on technical accuracy and current best practices.
            """
        )

    def _create_writer_agent(self) -> AssistantAgent:
        """Create content writer agent"""
        return AssistantAgent(
            name="writer",
            model_client=self.model,
            system_message="""
            You are an expert tech content writer. Your role is to:
            1. Transform research into engaging, well-structured articles
            2. Write clear, accessible content for the target audience
            3. Create compelling introductions and conclusions
            4. Use appropriate technical terminology with explanations
            5. Structure content with proper headings and flow
            6. Include practical examples and use cases

            Write in a conversational yet professional tone.
            Always consider the reader's experience level and learning goals.
            """
        )

    def _create_code_specialist_agent(self) -> AssistantAgent:
        """Create code example specialist agent"""
        return AssistantAgent(
            name="code_specialist",
            model_client=self.model,
            system_message="""
            You are a code example specialist. Your role is to:
            1. Create working, tested code examples
            2. Write clear code comments and explanations
            3. Follow best practices and coding standards
            4. Provide multiple implementation approaches when relevant
            5. Include error handling and edge cases
            6. Suggest improvements and optimizations

            Always test code examples and ensure they work as intended.
            Explain complex code sections step by step.
            """
        )

    def _create_editor_agent(self) -> AssistantAgent:
        """Create editor agent"""
        return AssistantAgent(
            name="editor",
            model_client=self.model,
            system_message="""
            You are a technical editor. Your role is to:
            1. Review content for clarity, accuracy, and flow
            2. Check grammar, spelling, and style consistency
            3. Ensure technical accuracy and proper terminology
            4. Optimize content structure and readability
            5. Verify that code examples work and are well-explained
            6. Suggest improvements for SEO and engagement

            Provide constructive feedback and specific suggestions.
            Maintain high editorial standards while preserving the author's voice.
            """
        )

    async def create_content(self, content_plan: ContentPlan) -> Dict:
        """Orchestrate multi-agent content creation"""
        # Initialize the collaborative writing process
        initial_message = f"""
        We need to create a {content_plan.content_type} about "{content_plan.topic}"
        with the title "{content_plan.title}".

        Target details:
        - Audience: {content_plan.target_audience}
        - Estimated length: {content_plan.estimated_length} words
        - Keywords: {', '.join(content_plan.keywords)}
        - Publish date: {content_plan.publish_date}

        Let's start with research phase. Researcher, please gather comprehensive
        information about this topic.
        """

        # Start the collaborative process using AutoGen 0.7 team
        result = await self.team.run(
            task=initial_message,
            cancellation_token=None
        )

        # Extract messages from the result
        messages = []
        if hasattr(result, 'messages') and result.messages:
            messages = [
                {
                    "name": msg.source,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                for msg in result.messages
            ]

        # Extract the final content from the conversation
        final_content = self._extract_final_content_from_messages(messages)

        return {
            "content": final_content,
            "word_count": len(final_content.split()),
            "created_at": datetime.now().isoformat(),
            "agents_involved": ["researcher", "writer", "code_specialist", "editor"],
            "conversation_log": messages
        }

    def _extract_final_content_from_messages(self, messages: List[Dict]) -> str:
        """Extract the final polished content from agent conversations"""
        # Find the last message from the editor with the final content
        for message in reversed(messages):
            if message.get("name") == "editor" and "FINAL CONTENT" in message.get("content", ""):
                return message["content"]

        # Fallback: return the last substantial message
        if messages:
            return messages[-1].get("content", "")
        return ""

class ContentGenerationWorkflow:
    """Workflow manager for the entire content generation process"""

    def __init__(self, config: Dict):
        self.config = config
        self.writing_team = WritingTeamOrchestrator(config)

    async def generate_content_batch(self, content_plans: List[ContentPlan]) -> List[Dict]:
        """Generate content for multiple plans"""
        results = []

        for plan in content_plans:
            try:
                # Generate content using multi-agent system
                content_result = await self.writing_team.create_content(plan)

                # Add metadata
                content_result.update({
                    "plan_id": plan.topic,  # Using topic as ID for now
                    "title": plan.title,
                    "content_type": plan.content_type,
                    "target_audience": plan.target_audience,
                    "keywords": plan.keywords,
                    "status": "generated"
                })

                results.append(content_result)

            except Exception as e:
                error_result = {
                    "plan_id": plan.topic,
                    "title": plan.title,
                    "status": "error",
                    "error": str(e),
                    "created_at": datetime.now().isoformat()
                }
                results.append(error_result)

        return results