from dataclasses import dataclass
from datetime import UTC, datetime

import aiohttp


@dataclass
class TrendingRepo:
    name: str
    description: str
    language: str
    stars: int
    trend_score: float
    topics: list[str]
    created_at: datetime


class GitHubTrendingScanner:
    def __init__(self, github_token: str):
        self.token = github_token
        self.base_url = "https://api.github.com"

    async def scan_trending_repositories(
        self, timeframe: str = "daily"
    ) -> list[TrendingRepo]:
        """Scan GitHub trending repositories"""
        since_date = self._get_since_date(timeframe)

        query = f"created:>{since_date} stars:>10"
        url = f"{self.base_url}/search/repositories"

        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100}

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json()

        return [self._parse_repo(repo) for repo in data.get("items", [])]

    async def scan_trending_topics(self) -> list[dict]:
        """Extract trending topics from repositories"""
        repos = await self.scan_trending_repositories()
        topic_counts = {}

        for repo in repos:
            for topic in repo.topics:
                if topic in topic_counts:
                    topic_counts[topic] += repo.trend_score
                else:
                    topic_counts[topic] = repo.trend_score

        # Sort topics by trend score
        trending_topics = [
            {
                "topic": topic,
                "score": score,
                "source": "github",
                "discovered_at": datetime.now(UTC),
            }
            for topic, score in sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return trending_topics[:50]  # Top 50 trending topics

    def _parse_repo(self, repo_data: dict) -> TrendingRepo:
        """Parse repository data from GitHub API"""
        return TrendingRepo(
            name=repo_data["full_name"],
            description=repo_data.get("description", ""),
            language=repo_data.get("language", ""),
            stars=repo_data["stargazers_count"],
            trend_score=self._calculate_trend_score(repo_data),
            topics=repo_data.get("topics", []),
            created_at=datetime.fromisoformat(
                repo_data["created_at"].replace("Z", "+00:00")
            ),
        )

    def _calculate_trend_score(self, repo_data: dict) -> float:
        """Calculate trend score based on stars, forks, and recency"""
        stars = repo_data["stargazers_count"]
        forks = repo_data["forks_count"]
        created_at = datetime.fromisoformat(
            repo_data["created_at"].replace("Z", "+00:00")
        )

        # Recency factor (newer repos get higher scores)
        days_old = (datetime.now(UTC) - created_at).days
        recency_factor = max(0.1, 1 - (days_old / 365))

        return (stars * 0.7 + forks * 0.3) * recency_factor
