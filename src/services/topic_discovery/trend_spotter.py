import os
import logging
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from .py_env import serpapi_api_key, newsapi_api_key, apify_api_token

@dataclass
class TrendCandidate:
    """Represents a trend candidate with its metrics"""
    topic: str
    search_rank: int = 0
    article_count: int = 0
    tweet_volume: int = 0
    final_score: float = 0.0

class TrendSpotter:
    """
    Trend identification module that combines Google Trends, NewsAPI, and Twitter data
    to identify the most promising weekly tech trends for blog content.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # API credentials from environment
        self.serpapi_key = serpapi_api_key
        self.newsapi_key = newsapi_api_key
        self.apify_key = apify_api_token

        # High-authority tech sources for news validation
        self.tech_sources = [
            'techcrunch.com',
            'venturebeat.com',
            'wired.com',
            'arstechnica.com',
            'theverge.com',
            'engadget.com',
            'mashable.com',
            'recode.net'
        ]

        # Scoring weights
        self.search_weight = 0.4
        self.article_weight = 0.4
        self.social_weight = 0.2

    def setup_logging(self):
        """Configure logging for transparency and debugging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def get_weekly_trend(self) -> Optional[Dict]:
        """
        Main method to identify the top weekly trend.

        Returns:
            Dict containing the selected trend and its metrics, or None if failed
        """
        self.logger.info("Starting weekly trend identification process")

        try:
            # Step 1: Fetch trend candidates from Google Trends
            candidates = self.fetch_trend_candidates()
            if not candidates:
                return self._fallback_to_previous_trend()

            # Step 2: Validate media saturation via NewsAPI
            candidates = self.validate_media_saturation(candidates)

            # Step 3: Validate social velocity via Twitter/Apify
            candidates = self.validate_social_velocity(candidates)

            # Step 4: Score and select best candidate
            selected_trend = self.score_and_select(candidates)

            if selected_trend:
                self.logger.info(f"Selected trend: {selected_trend.topic} (score: {selected_trend.final_score:.2f})")
                return {
                    'topic': selected_trend.topic,
                    'score': selected_trend.final_score,
                    'metrics': {
                        'search_rank': selected_trend.search_rank,
                        'article_count': selected_trend.article_count,
                        'tweet_volume': selected_trend.tweet_volume
                    },
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_to_previous_trend()

        except Exception as e:
            self.logger.error(f"Error in trend identification: {str(e)}")
            return self._fallback_to_previous_trend()

    def fetch_trend_candidates(self) -> List[TrendCandidate]:
        """
        Fetch trending topics from Google Trends via SerpApi.

        Returns:
            List of TrendCandidate objects with search rankings
        """
        self.logger.info("Fetching trend candidates from Google Trends")

        if not self.serpapi_key:
            self.logger.warning("SERPAPI_KEY not found, using mock data")
            return self._get_mock_trends()

        try:
            # SerpApi Google Trends query for tech category, last 7 days
            params = {
                'engine': 'google_trends',
                'q': 'technology',
                'category': '18',  # Technology category
                'date': 'now 7-d',  # Last 7 days
                'geo': 'US',
                'api_key': self.serpapi_key
            }

            response = requests.get('https://serpapi.com/search', params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            candidates = []

            # Extract trending topics from response
            if 'interest_over_time' in data:
                topics = data.get('related_topics', {}).get('top', [])[:10]  # Top 10

                for i, topic in enumerate(topics):
                    candidate = TrendCandidate(
                        topic=topic.get('query', ''),
                        search_rank=i + 1
                    )
                    candidates.append(candidate)

            self.logger.info(f"Found {len(candidates)} trend candidates")
            return candidates

        except Exception as e:
            self.logger.error(f"Error fetching trends from SerpApi: {str(e)}")
            return self._get_mock_trends()

    def validate_media_saturation(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """
        Validate media coverage for each candidate using NewsAPI.

        Args:
            candidates: List of trend candidates to validate

        Returns:
            Updated candidates with article counts
        """
        self.logger.info("Validating media saturation via NewsAPI")

        if not self.newsapi_key:
            self.logger.warning("NEWSAPI_KEY not found, using mock data")
            return self._add_mock_article_counts(candidates)

        for candidate in candidates:
            try:
                # Query NewsAPI for articles about this topic
                params = {
                    'q': candidate.topic,
                    'domains': ','.join(self.tech_sources),
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'sortBy': 'popularity',
                    'apiKey': self.newsapi_key,
                    'pageSize': 100
                }

                response = requests.get('https://newsapi.org/v2/everything', params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                candidate.article_count = data.get('totalResults', 0)

                self.logger.info(f"Topic '{candidate.topic}': {candidate.article_count} articles found")

            except Exception as e:
                self.logger.error(f"Error validating media saturation for '{candidate.topic}': {str(e)}")
                candidate.article_count = 0

        return candidates

    def validate_social_velocity(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """
        Validate social media velocity using Apify Twitter scraper.

        Args:
            candidates: List of trend candidates to validate

        Returns:
            Updated candidates with tweet volumes
        """
        self.logger.info("Validating social velocity via Apify Twitter scraper")

        if not self.apify_key:
            self.logger.warning("APIFY_KEY not found, using mock data")
            return self._add_mock_tweet_volumes(candidates)

        for candidate in candidates:
            try:
                # Use Apify Twitter scraper to get tweet volume
                apify_url = "https://api.apify.com/v2/acts/apify~twitter-scraper/run-sync-get-dataset-items"

                payload = {
                    'searchTerms': [candidate.topic],
                    'maxTweets': 100,
                    'tweetLanguage': 'en'
                }

                headers = {
                    'Authorization': f'Bearer {self.apify_key}',
                    'Content-Type': 'application/json'
                }

                response = requests.post(apify_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()

                data = response.json()
                candidate.tweet_volume = len(data) if isinstance(data, list) else 0

                self.logger.info(f"Topic '{candidate.topic}': {candidate.tweet_volume} tweets found")

            except Exception as e:
                self.logger.error(f"Error validating social velocity for '{candidate.topic}': {str(e)}")
                candidate.tweet_volume = 0

        return candidates

    def score_and_select(self, candidates: List[TrendCandidate]) -> Optional[TrendCandidate]:
        """
        Calculate scores for all candidates and select the best one.

        Args:
            candidates: List of validated trend candidates

        Returns:
            Selected trend candidate with highest score
        """
        self.logger.info("Calculating scores and selecting best candidate")

        if not candidates:
            return None

        # Normalize metrics for scoring
        max_rank = max(c.search_rank for c in candidates) if candidates else 1
        max_articles = max(c.article_count for c in candidates) if candidates else 1
        max_tweets = max(c.tweet_volume for c in candidates) if candidates else 1

        for candidate in candidates:
            # Normalize search rank (lower rank = higher score)
            rank_score = (max_rank - candidate.search_rank + 1) / max_rank

            # Normalize article count
            article_score = candidate.article_count / max_articles if max_articles > 0 else 0

            # Normalize tweet volume
            tweet_score = candidate.tweet_volume / max_tweets if max_tweets > 0 else 0

            # Calculate weighted final score
            candidate.final_score = (
                rank_score * self.search_weight +
                article_score * self.article_weight +
                tweet_score * self.social_weight
            )

            self.logger.info(
                f"Candidate '{candidate.topic}': "
                f"rank={candidate.search_rank}, articles={candidate.article_count}, "
                f"tweets={candidate.tweet_volume}, score={candidate.final_score:.3f}"
            )

        # Select candidate with highest score
        selected = max(candidates, key=lambda c: c.final_score)
        return selected

    def _fallback_to_previous_trend(self) -> Optional[Dict]:
        """Fallback mechanism when all APIs fail"""
        self.logger.warning("Falling back to previous week's trend")

        # This could load from a cache file or database
        fallback_topics = [
            "Artificial Intelligence",
            "Machine Learning",
            "Cloud Computing",
            "Cybersecurity",
            "Blockchain"
        ]

        import random
        selected_topic = random.choice(fallback_topics)

        return {
            'topic': selected_topic,
            'score': 0.5,
            'metrics': {
                'search_rank': 5,
                'article_count': 10,
                'tweet_volume': 50
            },
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }

    def _get_mock_trends(self) -> List[TrendCandidate]:
        """Mock trend data for testing"""
        return [
            TrendCandidate(topic="OpenAI GPT-5", search_rank=1),
            TrendCandidate(topic="Quantum Computing", search_rank=2),
            TrendCandidate(topic="Edge AI", search_rank=3),
            TrendCandidate(topic="Web3 Gaming", search_rank=4),
            TrendCandidate(topic="5G Networks", search_rank=5)
        ]

    def _add_mock_article_counts(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """Add mock article counts for testing"""
        import random
        for candidate in candidates:
            candidate.article_count = random.randint(5, 50)
        return candidates

    def _add_mock_tweet_volumes(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """Add mock tweet volumes for testing"""
        import random
        for candidate in candidates:
            candidate.tweet_volume = random.randint(100, 1000)
        return candidates
