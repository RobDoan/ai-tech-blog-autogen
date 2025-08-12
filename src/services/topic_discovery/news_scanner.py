# services/topic_discovery/news_scanner.py
import aiohttp
import feedparser
from typing import List, Dict
from datetime import datetime, timezone


class TechNewsScanner:
    def __init__(self):
        self.news_sources = [
            {
                "name": "Hacker News",
                "rss_url": "https://hnrss.org/frontpage",
                "weight": 0.8
            },
            {
                "name": "TechCrunch",
                "rss_url": "https://techcrunch.com/feed/",
                "weight": 0.7
            },
            {
                "name": "Ars Technica",
                "rss_url": "https://feeds.arstechnica.com/arstechnica/index",
                "weight": 0.7
            },
            {
                "name": "The Verge",
                "rss_url": "https://www.theverge.com/rss/index.xml",
                "weight": 0.6
            },
            {
                "name": "Wired",
                "rss_url": "https://www.wired.com/feed/rss",
                "weight": 0.6
            }
        ]

    async def scan_tech_news(self) -> List[Dict]:
        """Scan tech news from multiple sources"""
        all_articles = []

        for source in self.news_sources:
            try:
                articles = await self._scan_rss_feed(source)
                all_articles.extend(articles)
            except Exception as e:
                print(f"Error scanning {source['name']}: {e}")

        # Extract topics from articles
        topics = self._extract_topics_from_articles(all_articles)
        return topics

    async def _scan_rss_feed(self, source: Dict) -> List[Dict]:
        """Scan individual RSS feed"""
        async with aiohttp.ClientSession() as session:
            async with session.get(source["rss_url"]) as response:
                content = await response.text()

        feed = feedparser.parse(content)
        articles = []

        for entry in feed.entries[:20]:  # Top 20 articles per source
            article = {
                "title": entry.title,
                "description": getattr(entry, 'description', ''),
                "link": entry.link,
                "published": getattr(entry, 'published', ''),
                "source": source["name"],
                "weight": source["weight"],
                "discovered_at": datetime.now(timezone.utc)
            }
            articles.append(article)

        return articles

    def _extract_topics_from_articles(self, articles: List[Dict]) -> List[Dict]:
        """Extract trending topics from article titles and descriptions"""
        # Tech keywords to look for
        tech_keywords = [
            'AI', 'Machine Learning', 'Python', 'JavaScript', 'React', 'Vue',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'DevOps',
            'Blockchain', 'Web3', 'NFT', 'Cryptocurrency', 'DeFi',
            'Mobile', 'iOS', 'Android', 'Flutter', 'React Native',
            'Data Science', 'Big Data', 'Analytics', 'API', 'GraphQL',
            'Cybersecurity', 'Privacy', 'GDPR', 'Authentication',
            'Cloud', 'Serverless', 'Microservices', 'Database',
            'Open Source', 'GitHub', 'Git', 'CI/CD'
        ]

        topic_scores = {}

        for article in articles:
            text = f"{article['title']} {article['description']}".lower()

            for keyword in tech_keywords:
                if keyword.lower() in text:
                    score = article['weight'] * self._calculate_relevance_score(text, keyword.lower())

                    if keyword in topic_scores:
                        topic_scores[keyword] += score
                    else:
                        topic_scores[keyword] = score

        # Convert to topic list
        topics = [
            {
                "topic": topic,
                "score": score,
                "source": "tech_news",
                "discovered_at": datetime.now(timezone.utc)
            }
            for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return topics[:30]  # Top 30 trending topics

    def _calculate_relevance_score(self, text: str, keyword: str) -> float:
        """Calculate relevance score for a keyword in text"""
        # Count occurrences
        count = text.count(keyword)

        # Title boost
        title_boost = 2.0 if keyword in text.split('.')[0] else 1.0

        return count * title_boost