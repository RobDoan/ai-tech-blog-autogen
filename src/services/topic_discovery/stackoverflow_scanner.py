# services/topic_discovery/stackoverflow_scanner.py
from datetime import UTC, datetime

import aiohttp


class StackOverflowScanner:
    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3"

    async def scan_trending_tags(self, timeframe: str = "week") -> list[dict]:
        """Scan trending tags on Stack Overflow"""
        fromdate = self._get_fromdate(timeframe)

        url = f"{self.base_url}/tags"
        params = {
            "order": "desc",
            "sort": "popular",
            "site": "stackoverflow",
            "fromdate": fromdate,
            "pagesize": 100
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

        trending_tags = []
        for tag in data.get("items", []):
            trending_tags.append({
                "topic": tag["name"],
                "score": tag["count"],
                "source": "stackoverflow",
                "discovered_at": datetime.now(UTC),
                "metadata": {
                    "question_count": tag["count"],
                    "has_synonyms": tag.get("has_synonyms", False)
                }
            })

        return trending_tags

    async def scan_trending_questions(self) -> list[dict]:
        """Scan trending questions for topic extraction"""
        url = f"{self.base_url}/questions"
        params = {
            "order": "desc",
            "sort": "hot",
            "site": "stackoverflow",
            "pagesize": 100,
            "filter": "withbody"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

        trending_questions = []
        for question in data.get("items", []):
            trending_questions.append({
                "title": question["title"],
                "tags": question["tags"],
                "score": question["score"],
                "view_count": question["view_count"],
                "source": "stackoverflow",
                "discovered_at": datetime.now(UTC)
            })

        return trending_questions
