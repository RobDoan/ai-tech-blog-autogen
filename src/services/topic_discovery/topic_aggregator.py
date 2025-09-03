# services/topic_discovery/topic_aggregator.py
import asyncio
from dataclasses import dataclass
from datetime import datetime

from .github_scanner import GitHubTrendingScanner
from .news_scanner import TechNewsScanner
from .stackoverflow_scanner import StackOverflowScanner


@dataclass
class AggregatedTopic:
    topic: str
    total_score: float
    sources: list[str]
    first_seen: datetime
    last_updated: datetime
    trend_direction: str  # 'rising', 'stable', 'declining'
    metadata: dict

class TopicAggregator:
    def __init__(self, db_connection):
        self.db = db_connection
        self.scanners = {
            'github': GitHubTrendingScanner(),
            'stackoverflow': StackOverflowScanner(),
            'tech_news': TechNewsScanner()
        }

    async def discover_topics(self) -> list[AggregatedTopic]:
        """Run all scanners and aggregate topics"""
        all_topics = []

        # Run all scanners concurrently
        tasks = [
            self.scanners['github'].scan_trending_topics(),
            self.scanners['stackoverflow'].scan_trending_tags(),
            self.scanners['tech_news'].scan_tech_news()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all topics
        for result in results:
            if isinstance(result, list):
                all_topics.extend(result)

        # Aggregate and score topics
        aggregated = self._aggregate_topics(all_topics)

        # Store in database
        await self._store_topics(aggregated)

        return aggregated

    def _aggregate_topics(self, topics: list[dict]) -> list[AggregatedTopic]:
        """Aggregate topics from multiple sources"""
        topic_map = {}

        for topic_data in topics:
            topic_name = topic_data['topic'].lower()

            if topic_name in topic_map:
                # Update existing topic
                existing = topic_map[topic_name]
                existing['total_score'] += topic_data['score']
                existing['sources'].append(topic_data['source'])
                existing['last_updated'] = topic_data['discovered_at']
            else:
                # Create new topic entry
                topic_map[topic_name] = {
                    'topic': topic_data['topic'],
                    'total_score': topic_data['score'],
                    'sources': [topic_data['source']],
                    'first_seen': topic_data['discovered_at'],
                    'last_updated': topic_data['discovered_at'],
                    'metadata': topic_data.get('metadata', {})
                }

        # Convert to AggregatedTopic objects
        aggregated_topics = []
        for topic_name, data in topic_map.items():
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(topic_name)

            aggregated_topics.append(AggregatedTopic(
                topic=data['topic'],
                total_score=data['total_score'],
                sources=list(set(data['sources'])),  # Remove duplicates
                first_seen=data['first_seen'],
                last_updated=data['last_updated'],
                trend_direction=trend_direction,
                metadata=data['metadata']
            ))

        # Sort by total score
        aggregated_topics.sort(key=lambda x: x.total_score, reverse=True)

        return aggregated_topics[:100]  # Top 100 topics

    def _calculate_trend_direction(self, topic: str) -> str:
        """Calculate trend direction based on historical data"""
        # This would query historical data from the database
        # For now, return 'rising' as placeholder
        return 'rising'
