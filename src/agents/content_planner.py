# agents/content_planner.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio

# Import AggregatedTopic from the topic_aggregator module
try:
    from ..services.topic_discovery.topic_aggregator import AggregatedTopic
except ImportError:
    # For direct imports when testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from services.topic_discovery.topic_aggregator import AggregatedTopic


@dataclass
class ContentPlan:
    topic: str
    title: str
    content_type: str  # 'tutorial', 'guide', 'news', 'comparison'
    priority_score: float
    target_audience: str
    estimated_length: int
    required_resources: List[str]
    publish_date: datetime
    deadline: datetime
    keywords: List[str]
    competition_analysis: Dict

class ContentPlannerAgent:
    def __init__(self, config: Dict):
        self.config = config

        self.model = OpenAIChatCompletionClient(
            model=config.get("model", "gpt-4"),
            api_key=config["openai_api_key"],
            temperature=config.get("temperature", 0.3)
        )

        self.agent = AssistantAgent(
            name="content_planner",
            model_client=self.model,
            system_message=self._get_system_message()
        )

    async def create_content_plan(self, topics: List[AggregatedTopic]) -> List[ContentPlan]:
        """Create comprehensive content plan from trending topics"""
        content_plans = []

        for topic in topics[:20]:  # Process top 20 topics
            try:
                # Analyze topic potential
                analysis = await self._analyze_topic(topic)

                if analysis['is_suitable']:
                    # Generate content ideas
                    content_ideas = await self._generate_content_ideas(topic, analysis)

                    for idea in content_ideas:
                        plan = await self._create_detailed_plan(topic, idea, analysis)
                        content_plans.append(plan)

            except Exception as e:
                print(f"Error planning content for topic {topic.topic}: {e}")

        # Prioritize and schedule content
        prioritized_plans = self._prioritize_content(content_plans)
        scheduled_plans = self._schedule_content(prioritized_plans)

        return scheduled_plans

    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM using AutoGen 0.7 API"""
        try:
            # Create a text message
            message = TextMessage(content=prompt, source="user")

            # Use the agent to generate a response
            response = await self.agent.on_messages([message], cancellation_token=None)

            # Extract the content from the response
            if response.chat_message:
                return response.chat_message.content
            else:
                return "{}"  # Return empty JSON on failure

        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "{}"

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
            return {}

    async def _extract_keywords(self, topic: AggregatedTopic, idea: Dict) -> List[str]:
        """Extract keywords for SEO optimization"""
        prompt = f"""
        Extract 5-10 SEO keywords for the following content:

        Topic: {topic.topic}
        Title: {idea['title']}
        Content Type: {idea['content_type']}

        Return as a JSON array of keywords.
        """

        response = await self._query_llm(prompt)
        parsed = self._parse_json_response(response)
        return parsed if isinstance(parsed, list) else []

    async def _analyze_competition(self, title: str, keywords: List[str]) -> Dict:
        """Analyze competition for the content"""
        prompt = f"""
        Analyze the competition for this content:

        Title: {title}
        Keywords: {', '.join(keywords)}

        Provide analysis on:
        1. Competition level (1-10)
        2. Content gaps
        3. Differentiation opportunities

        Return as JSON.
        """

        response = await self._query_llm(prompt)
        return self._parse_json_response(response)

    def _determine_required_resources(self, idea: Dict) -> List[str]:
        """Determine required resources for content creation"""
        resources = ["Writer"]

        if idea.get('content_type') == 'tutorial':
            resources.extend(["Code Examples", "Screenshots"])
        elif idea.get('content_type') == 'comparison':
            resources.append("Research Analyst")
        elif idea.get('content_type') == 'guide':
            resources.extend(["Subject Matter Expert", "Diagrams"])

        return resources

    def _prioritize_content(self, content_plans: List[ContentPlan]) -> List[ContentPlan]:
        """Prioritize content plans by score"""
        return sorted(content_plans, key=lambda x: x.priority_score, reverse=True)

    def _schedule_content(self, prioritized_plans: List[ContentPlan]) -> List[ContentPlan]:
        """Schedule content plans avoiding conflicts"""
        scheduled_plans = []
        used_dates = set()

        for plan in prioritized_plans:
            # Ensure no date conflicts
            while plan.publish_date.date() in used_dates:
                plan.publish_date += timedelta(days=1)
                plan.deadline += timedelta(days=1)

            used_dates.add(plan.publish_date.date())
            scheduled_plans.append(plan)

        return scheduled_plans

    async def _analyze_topic(self, topic: AggregatedTopic) -> Dict:
        """Analyze topic suitability and potential"""
        prompt = f"""
        Analyze the following trending topic for content creation:

        Topic: {topic.topic}
        Sources: {', '.join(topic.sources)}
        Trend Score: {topic.total_score}
        Trend Direction: {topic.trend_direction}

        Please provide analysis on:
        1. Content suitability (is this good for a tech blog?)
        2. Target audience (beginner, intermediate, advanced)
        3. Content type recommendations (tutorial, guide, news, comparison)
        4. Competition level (how saturated is this topic?)
        5. SEO potential (search volume and difficulty)
        6. Estimated reader interest level (1-10)

        Respond in JSON format.
        """

        response = await self._query_llm(prompt)
        return self._parse_json_response(response)

    async def _generate_content_ideas(self, topic: AggregatedTopic, analysis: Dict) -> List[Dict]:
        """Generate specific content ideas for a topic"""
        prompt = f"""
        Based on the trending topic "{topic.topic}" and the following analysis:
        {analysis}

        Generate 3-5 specific content ideas. For each idea, provide:
        1. Compelling title
        2. Content type (tutorial, guide, comparison, news)
        3. Target audience level
        4. Key points to cover
        5. Estimated word count
        6. Unique angle or value proposition

        Focus on creating content that would be genuinely useful and engaging.
        Respond in JSON format with an array of content ideas.
        """

        response = await self._query_llm(prompt)
        return self._parse_json_response(response)

    async def _create_detailed_plan(self, topic: AggregatedTopic, idea: Dict, analysis: Dict) -> ContentPlan:
        """Create detailed content plan"""
        # Calculate priority score
        priority_score = self._calculate_priority_score(topic, analysis)

        # Determine publish date based on priority and topic freshness
        publish_date = self._calculate_optimal_publish_date(topic, priority_score)

        # Set deadline (3 days before publish date)
        deadline = publish_date - timedelta(days=3)

        # Extract keywords
        keywords = await self._extract_keywords(topic, idea)

        # Analyze competition
        competition = await self._analyze_competition(idea['title'], keywords)

        return ContentPlan(
            topic=topic.topic,
            title=idea['title'],
            content_type=idea['content_type'],
            priority_score=priority_score,
            target_audience=idea['target_audience'],
            estimated_length=idea['estimated_word_count'],
            required_resources=self._determine_required_resources(idea),
            publish_date=publish_date,
            deadline=deadline,
            keywords=keywords,
            competition_analysis=competition
        )

    def _calculate_priority_score(self, topic: AggregatedTopic, analysis: Dict) -> float:
        """Calculate content priority score"""
        base_score = topic.total_score

        # Trend direction multiplier
        trend_multiplier = {
            'rising': 1.5,
            'stable': 1.0,
            'declining': 0.7
        }.get(topic.trend_direction, 1.0)

        # Source diversity bonus
        source_bonus = len(topic.sources) * 0.1

        # Reader interest from analysis
        interest_multiplier = analysis.get('reader_interest', 5) / 5.0

        # Competition penalty
        competition_penalty = 1.0 - (analysis.get('competition_level', 5) / 10.0)

        return (base_score * trend_multiplier + source_bonus) * interest_multiplier * competition_penalty

    def _calculate_optimal_publish_date(self, topic: AggregatedTopic, priority_score: float) -> datetime:
        """Calculate optimal publish date"""
        from datetime import timezone
        now = datetime.now(timezone.utc)

        if priority_score > 100:  # High priority - publish within 1-2 days
            return now + timedelta(days=1, hours=8)  # Next business day morning
        elif priority_score > 50:  # Medium priority - publish within 3-5 days
            return now + timedelta(days=3, hours=8)
        else:  # Low priority - publish within 1-2 weeks
            return now + timedelta(days=7, hours=8)

    def _get_system_message(self) -> str:
        """Get system message for the content planner agent"""
        return """
        You are an expert content planner for a tech blog. Your role is to:
        1. Analyze trending topics for content potential
        2. Generate engaging content ideas
        3. Create detailed content plans
        4. Prioritize content based on audience value and business impact

        Always consider:
        - Target audience needs and skill levels
        - Content uniqueness and value proposition
        - SEO potential and competition
        - Resource requirements and feasibility
        - Optimal timing for publication

        Provide detailed, actionable plans that can be executed by writing teams.
        """