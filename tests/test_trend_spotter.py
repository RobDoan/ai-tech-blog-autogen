import pytest
import unittest.mock as mock
from datetime import datetime
from unittest.mock import patch, MagicMock
import requests
import json

from services.topic_discovery.trend_spotter import TrendSpotter, TrendCandidate


class TestTrendCandidate:
    """Tests for the TrendCandidate dataclass"""

    def test_trend_candidate_creation(self):
        """Test basic TrendCandidate creation"""
        candidate = TrendCandidate(topic="AI Testing")
        assert candidate.topic == "AI Testing"
        assert candidate.search_rank == 0
        assert candidate.article_count == 0
        assert candidate.tweet_volume == 0
        assert candidate.final_score == 0.0

    def test_trend_candidate_with_values(self):
        """Test TrendCandidate with all values"""
        candidate = TrendCandidate(
            topic="Machine Learning",
            search_rank=1,
            article_count=25,
            tweet_volume=500,
            final_score=0.85
        )
        assert candidate.topic == "Machine Learning"
        assert candidate.search_rank == 1
        assert candidate.article_count == 25
        assert candidate.tweet_volume == 500
        assert candidate.final_score == 0.85


class TestTrendSpotter:
    """Tests for the TrendSpotter class"""

    @pytest.fixture
    def trend_spotter(self):
        """Create a TrendSpotter instance for testing"""
        with patch('src.autogen_blog.trend_spotter.serpapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.newsapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.apify_api_token', None):
            return TrendSpotter()

    @pytest.fixture
    def trend_spotter_with_keys(self):
        """Create a TrendSpotter instance with API keys for testing"""
        with patch('src.autogen_blog.trend_spotter.serpapi_api_key', 'test_serp_key'), \
             patch('src.autogen_blog.trend_spotter.newsapi_api_key', 'test_news_key'), \
             patch('src.autogen_blog.trend_spotter.apify_api_token', 'test_apify_key'):
            return TrendSpotter()

    @pytest.fixture
    def sample_candidates(self):
        """Create sample trend candidates for testing"""
        return [
            TrendCandidate(topic="OpenAI GPT-5", search_rank=1, article_count=30, tweet_volume=1000),
            TrendCandidate(topic="Quantum Computing", search_rank=2, article_count=20, tweet_volume=800),
            TrendCandidate(topic="Edge AI", search_rank=3, article_count=15, tweet_volume=600)
        ]

    def test_initialization_without_keys(self, trend_spotter):
        """Test TrendSpotter initialization without API keys"""
        assert trend_spotter.serpapi_key is None
        assert trend_spotter.newsapi_key is None
        assert trend_spotter.apify_key is None
        assert trend_spotter.search_weight == 0.4
        assert trend_spotter.article_weight == 0.4
        assert trend_spotter.social_weight == 0.2
        assert len(trend_spotter.tech_sources) == 8

    def test_initialization_with_keys(self, trend_spotter_with_keys):
        """Test TrendSpotter initialization with API keys"""
        assert trend_spotter_with_keys.serpapi_key == 'test_serp_key'
        assert trend_spotter_with_keys.newsapi_key == 'test_news_key'
        assert trend_spotter_with_keys.apify_key == 'test_apify_key'

    def test_get_mock_trends(self, trend_spotter):
        """Test mock trend data generation"""
        mock_trends = trend_spotter._get_mock_trends()
        assert len(mock_trends) == 5
        assert all(isinstance(t, TrendCandidate) for t in mock_trends)
        assert mock_trends[0].topic == "OpenAI GPT-5"
        assert mock_trends[0].search_rank == 1
        assert mock_trends[4].topic == "5G Networks"
        assert mock_trends[4].search_rank == 5

    def test_add_mock_article_counts(self, trend_spotter, sample_candidates):
        """Test adding mock article counts"""
        original_counts = [c.article_count for c in sample_candidates]
        updated = trend_spotter._add_mock_article_counts(sample_candidates)

        assert len(updated) == 3
        for i, candidate in enumerate(updated):
            # Mock should change the article count
            assert 5 <= candidate.article_count <= 50
            # Other fields should remain unchanged
            assert candidate.topic == sample_candidates[i].topic
            assert candidate.search_rank == sample_candidates[i].search_rank

    def test_add_mock_tweet_volumes(self, trend_spotter, sample_candidates):
        """Test adding mock tweet volumes"""
        updated = trend_spotter._add_mock_tweet_volumes(sample_candidates)

        assert len(updated) == 3
        for candidate in updated:
            assert 100 <= candidate.tweet_volume <= 1000

    @patch('requests.get')
    def test_fetch_trend_candidates_success(self, mock_get, trend_spotter_with_keys):
        """Test successful trend fetching from SerpApi"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'interest_over_time': [{'value': 100}],
            'related_topics': {
                'top': [
                    {'query': 'Artificial Intelligence'},
                    {'query': 'Machine Learning'},
                    {'query': 'Deep Learning'}
                ]
            }
        }
        mock_get.return_value = mock_response

        candidates = trend_spotter_with_keys.fetch_trend_candidates()

        assert len(candidates) == 3
        assert candidates[0].topic == 'Artificial Intelligence'
        assert candidates[0].search_rank == 1
        assert candidates[1].topic == 'Machine Learning'
        assert candidates[1].search_rank == 2
        assert candidates[2].topic == 'Deep Learning'
        assert candidates[2].search_rank == 3

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'https://serpapi.com/search' in call_args[0]
        assert call_args[1]['params']['engine'] == 'google_trends'
        assert call_args[1]['params']['api_key'] == 'test_serp_key'

    @patch('requests.get')
    def test_fetch_trend_candidates_api_error(self, mock_get, trend_spotter_with_keys):
        """Test trend fetching with API error"""
        mock_get.side_effect = requests.RequestException("API Error")

        candidates = trend_spotter_with_keys.fetch_trend_candidates()

        # Should fall back to mock data
        assert len(candidates) == 5
        assert candidates[0].topic == "OpenAI GPT-5"

    def test_fetch_trend_candidates_no_key(self, trend_spotter):
        """Test trend fetching without API key"""
        candidates = trend_spotter.fetch_trend_candidates()

        # Should use mock data
        assert len(candidates) == 5
        assert candidates[0].topic == "OpenAI GPT-5"

    @patch('requests.get')
    def test_validate_media_saturation_success(self, mock_get, trend_spotter_with_keys, sample_candidates):
        """Test successful media saturation validation"""
        # Mock NewsAPI response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'totalResults': 42}
        mock_get.return_value = mock_response

        updated_candidates = trend_spotter_with_keys.validate_media_saturation(sample_candidates)

        assert len(updated_candidates) == 3
        for candidate in updated_candidates:
            assert candidate.article_count == 42

        # Should make 3 API calls (one per candidate)
        assert mock_get.call_count == 3

        # Verify API parameters
        for call in mock_get.call_args_list:
            assert 'https://newsapi.org/v2/everything' in call[0]
            assert call[1]['params']['apiKey'] == 'test_news_key'
            assert 'techcrunch.com' in call[1]['params']['domains']

    @patch('requests.get')
    def test_validate_media_saturation_api_error(self, mock_get, trend_spotter_with_keys, sample_candidates):
        """Test media saturation validation with API error"""
        mock_get.side_effect = requests.RequestException("API Error")

        updated_candidates = trend_spotter_with_keys.validate_media_saturation(sample_candidates)

        # Should set article_count to 0 on error
        for candidate in updated_candidates:
            assert candidate.article_count == 0

    def test_validate_media_saturation_no_key(self, trend_spotter, sample_candidates):
        """Test media saturation validation without API key"""
        updated_candidates = trend_spotter.validate_media_saturation(sample_candidates)

        # Should use mock data
        for candidate in updated_candidates:
            assert 5 <= candidate.article_count <= 50

    @patch('requests.post')
    def test_validate_social_velocity_success(self, mock_post, trend_spotter_with_keys, sample_candidates):
        """Test successful social velocity validation"""
        # Mock Apify response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{'tweet': 'data'}] * 75  # 75 tweets
        mock_post.return_value = mock_response

        updated_candidates = trend_spotter_with_keys.validate_social_velocity(sample_candidates)

        assert len(updated_candidates) == 3
        for candidate in updated_candidates:
            assert candidate.tweet_volume == 75

        # Should make 3 API calls
        assert mock_post.call_count == 3

        # Verify API parameters
        for call in mock_post.call_args_list:
            assert 'apify.com' in call[0][0]
            assert call[1]['headers']['Authorization'] == 'Bearer test_apify_key'

    @patch('requests.post')
    def test_validate_social_velocity_api_error(self, mock_post, trend_spotter_with_keys, sample_candidates):
        """Test social velocity validation with API error"""
        mock_post.side_effect = requests.RequestException("API Error")

        updated_candidates = trend_spotter_with_keys.validate_social_velocity(sample_candidates)

        # Should set tweet_volume to 0 on error
        for candidate in updated_candidates:
            assert candidate.tweet_volume == 0

    def test_validate_social_velocity_no_key(self, trend_spotter, sample_candidates):
        """Test social velocity validation without API key"""
        updated_candidates = trend_spotter.validate_social_velocity(sample_candidates)

        # Should use mock data
        for candidate in updated_candidates:
            assert 100 <= candidate.tweet_volume <= 1000

    def test_score_and_select_basic(self, trend_spotter, sample_candidates):
        """Test basic scoring and selection"""
        selected = trend_spotter.score_and_select(sample_candidates)

        assert selected is not None
        assert isinstance(selected, TrendCandidate)
        assert selected.final_score > 0

        # Verify all candidates have scores
        for candidate in sample_candidates:
            assert candidate.final_score > 0

    def test_score_and_select_ranking(self, trend_spotter):
        """Test that scoring correctly ranks candidates"""
        candidates = [
            TrendCandidate(topic="Low Rank High Media", search_rank=1, article_count=50, tweet_volume=1000),
            TrendCandidate(topic="High Rank Low Media", search_rank=5, article_count=5, tweet_volume=100),
            TrendCandidate(topic="Balanced", search_rank=3, article_count=25, tweet_volume=500)
        ]

        selected = trend_spotter.score_and_select(candidates)

        assert selected.topic == "Low Rank High Media"  # Should have highest score

        # Verify scores are in expected order
        scores = {c.topic: c.final_score for c in candidates}
        assert scores["Low Rank High Media"] > scores["Balanced"]
        assert scores["Balanced"] > scores["High Rank Low Media"]

    def test_score_and_select_empty_list(self, trend_spotter):
        """Test scoring with empty candidate list"""
        selected = trend_spotter.score_and_select([])
        assert selected is None

    def test_score_and_select_single_candidate(self, trend_spotter):
        """Test scoring with single candidate"""
        candidates = [TrendCandidate(topic="Only One", search_rank=1, article_count=10, tweet_volume=100)]
        selected = trend_spotter.score_and_select(candidates)

        assert selected is not None
        assert selected.topic == "Only One"
        assert selected.final_score == 1.0  # Should get max score as only candidate

    def test_fallback_to_previous_trend(self, trend_spotter):
        """Test fallback mechanism"""
        result = trend_spotter._fallback_to_previous_trend()

        assert result is not None
        assert 'topic' in result
        assert 'score' in result
        assert 'metrics' in result
        assert 'timestamp' in result
        assert result['fallback'] is True
        assert result['score'] == 0.5

        # Topic should be one of the fallback topics
        fallback_topics = ["Artificial Intelligence", "Machine Learning", "Cloud Computing", "Cybersecurity", "Blockchain"]
        assert result['topic'] in fallback_topics

    @patch('src.autogen_blog.trend_spotter.TrendSpotter.fetch_trend_candidates')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.validate_media_saturation')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.validate_social_velocity')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.score_and_select')
    def test_get_weekly_trend_success(self, mock_score, mock_social, mock_media, mock_fetch, trend_spotter):
        """Test successful weekly trend identification"""
        # Setup mocks
        mock_candidates = [TrendCandidate(topic="Test Topic", search_rank=1)]
        mock_fetch.return_value = mock_candidates
        mock_media.return_value = mock_candidates
        mock_social.return_value = mock_candidates

        selected_candidate = TrendCandidate(
            topic="Selected Topic",
            search_rank=1,
            article_count=25,
            tweet_volume=500,
            final_score=0.85
        )
        mock_score.return_value = selected_candidate

        result = trend_spotter.get_weekly_trend()

        assert result is not None
        assert result['topic'] == "Selected Topic"
        assert result['score'] == 0.85
        assert result['metrics']['search_rank'] == 1
        assert result['metrics']['article_count'] == 25
        assert result['metrics']['tweet_volume'] == 500
        assert 'timestamp' in result
        assert 'fallback' not in result

        # Verify all methods were called
        mock_fetch.assert_called_once()
        mock_media.assert_called_once_with(mock_candidates)
        mock_social.assert_called_once_with(mock_candidates)
        mock_score.assert_called_once_with(mock_candidates)

    @patch('src.autogen_blog.trend_spotter.TrendSpotter.fetch_trend_candidates')
    def test_get_weekly_trend_no_candidates(self, mock_fetch, trend_spotter):
        """Test weekly trend identification when no candidates found"""
        mock_fetch.return_value = []

        result = trend_spotter.get_weekly_trend()

        assert result is not None
        assert result['fallback'] is True

    @patch('src.autogen_blog.trend_spotter.TrendSpotter.fetch_trend_candidates')
    def test_get_weekly_trend_exception(self, mock_fetch, trend_spotter):
        """Test weekly trend identification with exception"""
        mock_fetch.side_effect = Exception("Test exception")

        result = trend_spotter.get_weekly_trend()

        assert result is not None
        assert result['fallback'] is True

    @patch('src.autogen_blog.trend_spotter.TrendSpotter.score_and_select')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.validate_social_velocity')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.validate_media_saturation')
    @patch('src.autogen_blog.trend_spotter.TrendSpotter.fetch_trend_candidates')
    def test_get_weekly_trend_no_selection(self, mock_fetch, mock_media, mock_social, mock_score, trend_spotter):
        """Test weekly trend identification when no trend is selected"""
        mock_candidates = [TrendCandidate(topic="Test Topic")]
        mock_fetch.return_value = mock_candidates
        mock_media.return_value = mock_candidates
        mock_social.return_value = mock_candidates
        mock_score.return_value = None  # No selection made

        result = trend_spotter.get_weekly_trend()

        assert result is not None
        assert result['fallback'] is True


class TestTrendSpotterIntegration:
    """Integration tests for TrendSpotter"""

    def test_full_workflow_without_keys(self):
        """Test the complete workflow without API keys (using mocks)"""
        with patch('src.autogen_blog.trend_spotter.serpapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.newsapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.apify_api_token', None):

            spotter = TrendSpotter()
            result = spotter.get_weekly_trend()

            assert result is not None
            assert 'topic' in result
            assert 'score' in result
            assert 'metrics' in result
            assert 'timestamp' in result

            # Should use fallback or mock data
            assert result['score'] > 0

    def test_scoring_weights_sum_to_one(self):
        """Test that scoring weights sum to 1.0"""
        with patch('src.autogen_blog.trend_spotter.serpapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.newsapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.apify_api_token', None):

            spotter = TrendSpotter()
            total_weight = spotter.search_weight + spotter.article_weight + spotter.social_weight
            assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision

    def test_tech_sources_are_valid_domains(self):
        """Test that all tech sources are valid domain names"""
        with patch('src.autogen_blog.trend_spotter.serpapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.newsapi_api_key', None), \
             patch('src.autogen_blog.trend_spotter.apify_api_token', None):

            spotter = TrendSpotter()
            for source in spotter.tech_sources:
                assert '.' in source  # Basic domain validation
                assert not source.startswith('http')  # Should be domain only
                assert ' ' not in source  # No spaces in domain names


if __name__ == '__main__':
    pytest.main([__file__])