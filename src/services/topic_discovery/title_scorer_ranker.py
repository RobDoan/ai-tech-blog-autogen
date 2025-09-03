# src/services/topic_discovery/title_scorer_ranker.py
"""
Title Scorer and Ranker for Blog Title Discovery

This module implements sophisticated scoring and ranking algorithms for blog titles
based on specificity, engagement potential, technical depth, and other quality metrics.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .blog_title_generator import BlogTitleCandidate


@dataclass
class TitleScores:
    """Comprehensive scoring breakdown for a blog title"""

    specificity_score: float = 0.0  # 0-1: Concrete details, metrics, versions
    engagement_score: float = 0.0  # 0-1: Actionability, problem-solving appeal
    recency_score: float = 0.0  # 0-1: How recent/trending the content is
    authority_score: float = 0.0  # 0-1: Source credibility and technical depth
    uniqueness_score: float = 0.0  # 0-1: How unique/novel the title is
    seo_score: float = 0.0  # 0-1: Search engine optimization potential

    # Composite scores
    overall_score: float = 0.0  # Weighted combination of all scores
    ranking_score: float = 0.0  # Final score used for ranking

    # Score breakdown details
    score_breakdown: dict[str, float] = field(default_factory=dict)
    quality_indicators: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)


@dataclass
class RankedBlogTitle:
    """A blog title with comprehensive scoring and ranking information"""

    title: str
    original_candidate: BlogTitleCandidate
    scores: TitleScores

    # Ranking metadata
    rank: int = 0
    percentile: float = 0.0  # 0-100 percentile among all titles
    quality_tier: str = "good"  # "excellent", "good", "fair", "poor"

    # Analysis details
    target_metrics: list[str] = field(default_factory=list)
    competitive_keywords: list[str] = field(default_factory=list)
    estimated_audience_size: str = "medium"  # "large", "medium", "niche"


class TitleScorerRanker:
    """
    Advanced title scoring and ranking system that evaluates blog titles
    across multiple dimensions to identify the most engaging and specific content ideas.
    """

    def __init__(
        self,
        min_specificity_threshold: float = 0.4,
        weight_specificity: float = 0.25,
        weight_engagement: float = 0.25,
        weight_recency: float = 0.15,
        weight_authority: float = 0.15,
        weight_uniqueness: float = 0.10,
        weight_seo: float = 0.10,
    ):
        """
        Initialize Title Scorer and Ranker

        Args:
            min_specificity_threshold: Minimum specificity score to pass filtering
            weight_*: Weights for different scoring dimensions (should sum to 1.0)
        """
        self.logger = logging.getLogger(__name__)

        # Filtering thresholds
        self.min_specificity_threshold = min_specificity_threshold

        # Scoring weights
        self.weights = {
            "specificity": weight_specificity,
            "engagement": weight_engagement,
            "recency": weight_recency,
            "authority": weight_authority,
            "uniqueness": weight_uniqueness,
            "seo": weight_seo,
        }

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            self.logger.warning(
                f"Scoring weights sum to {weight_sum:.3f}, should be 1.0"
            )

        # Initialize scoring patterns and data
        self._compile_scoring_patterns()
        self._load_engagement_keywords()
        self._load_technical_authority_indicators()

        self.logger.info("Title Scorer and Ranker initialized")

    def _compile_scoring_patterns(self):
        """Compile regex patterns for various scoring metrics"""

        # Specificity indicators (concrete details)
        self.specificity_patterns = {
            "metrics": re.compile(
                r"\b(\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:ms|seconds?|x|times?|MB|GB))\b",
                re.IGNORECASE,
            ),
            "versions": re.compile(
                r"\b(?:v?\d+\.\d+(?:\.\d+)?|[A-Z][a-z]+\s+\d+(?:\.\d+)?)\b"
            ),
            "companies": re.compile(
                r"\b(Netflix|Google|Amazon|AWS|Facebook|Meta|Microsoft|Apple|Stripe|Uber|Airbnb|Spotify|Twitter|X|LinkedIn|GitHub|GitLab|Vercel|Netlify|Cloudflare|OpenAI|Anthropic)\b"
            ),
            "technologies": re.compile(
                r"\b(React|Vue|Angular|Next\.js|Node\.js|Python|JavaScript|TypeScript|Go|Rust|Java|C\+\+|C#|Swift|Kotlin|Docker|Kubernetes|PostgreSQL|MongoDB|Redis|GraphQL|REST|API)\b"
            ),
            "specific_numbers": re.compile(
                r"\b\d+\s+(?:ways?|steps?|methods?|techniques?|tips?|tricks?|strategies?)\b",
                re.IGNORECASE,
            ),
        }

        # Engagement indicators (action-oriented, problem-solving)
        self.engagement_patterns = {
            "action_words": re.compile(
                r"\b(how|why|what|when|where|guide|tutorial|learn|master|build|create|implement|optimize|improve|solve|fix)\b",
                re.IGNORECASE,
            ),
            "results": re.compile(
                r"\b(improved?|reduced?|increased?|optimized?|enhanced?|better|faster|easier|simpler|efficient)\b",
                re.IGNORECASE,
            ),
            "problems": re.compile(
                r"\b(problem|issue|challenge|bug|error|bottleneck|slow|broken|failing|difficult)\b",
                re.IGNORECASE,
            ),
            "benefits": re.compile(
                r"\b(save|gain|boost|accelerate|streamline|automate|scale|secure)\b",
                re.IGNORECASE,
            ),
        }

        # SEO-friendly patterns
        self.seo_patterns = {
            "questions": re.compile(
                r"\b(how|why|what|when|where|which|should|can|will|do|does|is|are)\b",
                re.IGNORECASE,
            ),
            "lists": re.compile(
                r"\b\d+\s+(?:best|top|ways?|tips?|methods?|tools?|frameworks?|libraries?)\b",
                re.IGNORECASE,
            ),
            "comparisons": re.compile(
                r"\bvs\.?|\bcompared?\s+to\b|\bbetter\s+than\b|\bversus\b",
                re.IGNORECASE,
            ),
        }

    def _load_engagement_keywords(self):
        """Load high-engagement keywords for different audiences"""

        self.engagement_keywords = {
            "developers": [
                "api",
                "framework",
                "library",
                "database",
                "performance",
                "optimization",
                "debugging",
                "testing",
                "deployment",
                "architecture",
                "microservices",
                "containerization",
                "ci/cd",
                "devops",
                "monitoring",
                "logging",
            ],
            "managers": [
                "productivity",
                "efficiency",
                "cost",
                "roi",
                "scaling",
                "team",
                "process",
                "workflow",
                "automation",
                "strategy",
                "planning",
                "resources",
                "budget",
                "timeline",
                "delivery",
            ],
            "beginners": [
                "introduction",
                "getting started",
                "basics",
                "fundamentals",
                "tutorial",
                "guide",
                "step by step",
                "learn",
                "understand",
                "explain",
                "simple",
                "easy",
                "beginner",
                "first time",
            ],
        }

    def _load_technical_authority_indicators(self):
        """Load indicators of technical authority and credibility"""

        self.authority_indicators = {
            "high_authority_companies": [
                "Netflix",
                "Google",
                "Amazon",
                "Facebook",
                "Microsoft",
                "Apple",
                "Stripe",
                "Uber",
                "Airbnb",
                "Spotify",
                "GitHub",
            ],
            "technical_depth_indicators": [
                "architecture",
                "implementation",
                "deep dive",
                "internals",
                "optimization",
                "performance",
                "scalability",
                "distributed",
                "system design",
            ],
            "innovation_indicators": [
                "new",
                "latest",
                "cutting-edge",
                "innovative",
                "breakthrough",
                "revolutionary",
                "next-generation",
                "state-of-the-art",
            ],
        }

    def rank_titles(self, titles: list[BlogTitleCandidate]) -> list[RankedBlogTitle]:
        """
        Score and rank blog titles based on comprehensive quality metrics

        Args:
            titles: List of blog title candidates to rank

        Returns:
            List of ranked blog titles sorted by overall quality
        """
        if not titles:
            return []

        self.logger.info(f"Scoring and ranking {len(titles)} blog titles")

        # Score each title
        ranked_titles = []
        for candidate in titles:
            scores = self._calculate_comprehensive_scores(candidate)

            ranked_title = RankedBlogTitle(
                title=candidate.title, original_candidate=candidate, scores=scores
            )

            # Add additional analysis
            self._analyze_target_metrics(ranked_title)
            self._analyze_competitive_keywords(ranked_title)
            self._estimate_audience_size(ranked_title)

            ranked_titles.append(ranked_title)

        # Filter by minimum specificity threshold
        filtered_titles = [
            title
            for title in ranked_titles
            if title.scores.specificity_score >= self.min_specificity_threshold
        ]

        if len(filtered_titles) < len(ranked_titles):
            filtered_count = len(ranked_titles) - len(filtered_titles)
            self.logger.info(
                f"Filtered out {filtered_count} titles below specificity threshold"
            )

        # Sort by ranking score
        sorted_titles = sorted(
            filtered_titles, key=lambda x: x.scores.ranking_score, reverse=True
        )

        # Assign ranks and percentiles
        for i, title in enumerate(sorted_titles):
            title.rank = i + 1
            title.percentile = (
                100 * (len(sorted_titles) - i) / len(sorted_titles)
                if sorted_titles
                else 0
            )
            title.quality_tier = self._determine_quality_tier(
                title.scores.ranking_score
            )

        self.logger.info(f"Ranked {len(sorted_titles)} qualifying titles")
        return sorted_titles

    def _calculate_comprehensive_scores(
        self, candidate: BlogTitleCandidate
    ) -> TitleScores:
        """Calculate comprehensive scores for a title candidate"""

        scores = TitleScores()
        title_text = candidate.title.lower()

        # 1. Specificity Score
        scores.specificity_score = self._calculate_specificity_score(
            candidate, title_text
        )

        # 2. Engagement Score
        scores.engagement_score = self._calculate_engagement_score(
            candidate, title_text
        )

        # 3. Recency Score
        scores.recency_score = self._calculate_recency_score(candidate)

        # 4. Authority Score
        scores.authority_score = self._calculate_authority_score(candidate, title_text)

        # 5. Uniqueness Score
        scores.uniqueness_score = self._calculate_uniqueness_score(
            candidate, title_text
        )

        # 6. SEO Score
        scores.seo_score = self._calculate_seo_score(candidate, title_text)

        # Calculate composite scores
        scores.overall_score = (
            scores.specificity_score * self.weights["specificity"]
            + scores.engagement_score * self.weights["engagement"]
            + scores.recency_score * self.weights["recency"]
            + scores.authority_score * self.weights["authority"]
            + scores.uniqueness_score * self.weights["uniqueness"]
            + scores.seo_score * self.weights["seo"]
        )

        # Apply quality bonuses and penalties
        scores.ranking_score = self._apply_ranking_adjustments(scores, candidate)

        # Generate score breakdown and suggestions
        self._populate_score_details(scores, candidate)

        return scores

    def _calculate_specificity_score(
        self, candidate: BlogTitleCandidate, title_text: str
    ) -> float:
        """Calculate specificity score based on concrete details"""

        score = 0.0

        # Check for specific patterns
        for pattern_type, pattern in self.specificity_patterns.items():
            matches = len(pattern.findall(candidate.title))

            if pattern_type == "metrics":
                score += min(
                    matches * 0.3, 0.4
                )  # Performance metrics are highly valuable
            elif pattern_type == "companies":
                score += min(matches * 0.25, 0.35)  # Company names add authority
            elif pattern_type == "technologies":
                score += min(matches * 0.15, 0.3)  # Technology names add specificity
            elif pattern_type == "versions":
                score += min(matches * 0.2, 0.25)  # Version numbers are specific
            elif pattern_type == "specific_numbers":
                score += min(matches * 0.2, 0.3)  # "5 ways to..." format

        # Bonus for candidate's own specificity indicators
        if candidate.metrics_mentioned:
            score += min(len(candidate.metrics_mentioned) * 0.15, 0.25)

        if candidate.companies_mentioned:
            score += min(len(candidate.companies_mentioned) * 0.1, 0.2)

        # Length penalty for overly long titles (less specific focus)
        word_count = len(candidate.title.split())
        if word_count > 12:
            score *= 0.9
        elif word_count < 6:
            score *= 0.85  # Too short might be too general

        return min(score, 1.0)

    def _calculate_engagement_score(
        self, candidate: BlogTitleCandidate, title_text: str
    ) -> float:
        """Calculate engagement score based on actionability and problem-solving appeal"""

        score = 0.2  # Base engagement score

        # Check engagement patterns
        for pattern_type, pattern in self.engagement_patterns.items():
            matches = len(pattern.findall(candidate.title))

            if pattern_type == "action_words":
                score += min(matches * 0.2, 0.3)
            elif pattern_type == "results":
                score += min(matches * 0.15, 0.25)
            elif pattern_type == "problems":
                score += min(matches * 0.1, 0.2)
            elif pattern_type == "benefits":
                score += min(matches * 0.1, 0.15)

        # Audience-specific keyword bonuses
        audience = candidate.technical_depth
        if audience in self.engagement_keywords:
            keyword_matches = sum(
                1
                for keyword in self.engagement_keywords[audience]
                if keyword in title_text
            )
            score += min(keyword_matches * 0.05, 0.15)

        # Pattern type bonuses
        if candidate.pattern_type == "how-to":
            score += 0.15  # How-to titles are inherently engaging
        elif candidate.pattern_type == "performance":
            score += 0.1  # Performance improvements are compelling
        elif candidate.pattern_type == "problem-solution":
            score += 0.12  # Problem-solving is highly engaging

        # Title structure bonuses
        if title_text.startswith(("how ", "why ", "what ")):
            score += 0.1

        if any(word in title_text for word in ["vs", "versus", "compared"]):
            score += 0.08  # Comparisons are engaging

        return min(score, 1.0)

    def _calculate_recency_score(self, candidate: BlogTitleCandidate) -> float:
        """Calculate recency score based on how recent/trending the content is"""

        # Base score for all recent content
        score = 0.5

        # Time-based scoring (assuming candidate.created_at represents content recency)
        now = datetime.now(UTC)
        age_hours = (now - candidate.created_at).total_seconds() / 3600

        if age_hours < 24:
            score += 0.3  # Very recent
        elif age_hours < 168:  # 1 week
            score += 0.2
        elif age_hours < 720:  # 1 month
            score += 0.1

        # Trending technology indicators
        trending_techs = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml",
            "llm",
            "gpt",
            "react 19",
            "node 20",
            "python 3.12",
            "typescript 5",
            "kubernetes",
            "docker",
            "serverless",
            "edge computing",
            "web3",
            "blockchain",
            "cryptocurrency",
        ]

        title_lower = candidate.title.lower()
        for tech in trending_techs:
            if tech in title_lower:
                score += 0.15
                break

        # Version number recency bonus
        if any(
            pattern.search(candidate.title)
            for pattern in [
                re.compile(r"\b20[2-9]\d\b"),  # Recent years
                re.compile(r"\b(?:latest|new|newest|updated?)\b", re.IGNORECASE),
            ]
        ):
            score += 0.1

        return min(score, 1.0)

    def _calculate_authority_score(
        self, candidate: BlogTitleCandidate, title_text: str
    ) -> float:
        """Calculate authority score based on source credibility and technical depth"""

        score = 0.3  # Base authority score

        # High-authority company mentions
        for company in self.authority_indicators["high_authority_companies"]:
            if company.lower() in title_text:
                score += 0.25
                break

        # Technical depth indicators
        depth_matches = sum(
            1
            for indicator in self.authority_indicators["technical_depth_indicators"]
            if indicator in title_text
        )
        score += min(depth_matches * 0.1, 0.3)

        # Innovation indicators
        innovation_matches = sum(
            1
            for indicator in self.authority_indicators["innovation_indicators"]
            if indicator in title_text
        )
        score += min(innovation_matches * 0.08, 0.2)

        # Technical depth from candidate
        if candidate.technical_depth == "advanced":
            score += 0.15
        elif candidate.technical_depth == "intermediate":
            score += 0.1

        # Confidence from generation process
        score += candidate.confidence * 0.2

        # Technologies mentioned (more = higher authority)
        tech_count = len(candidate.key_technologies)
        score += min(tech_count * 0.05, 0.15)

        return min(score, 1.0)

    def _calculate_uniqueness_score(
        self, candidate: BlogTitleCandidate, title_text: str
    ) -> float:
        """Calculate uniqueness score based on novelty and differentiation"""

        score = 0.5  # Base uniqueness score

        # Unique combination indicators
        unique_combinations = [
            (r"\b(netflix|stripe|uber)\b.*\b(reduced?|improved?)\b.*\b\d+%\b", 0.2),
            (r"\bwhy\b.*\boutperform", 0.15),
            (r"\b\d+\s+ways?\b.*\bwith\b.*\b(react|vue|angular)\b", 0.15),
            (r"\bhow\b.*\bsolve.*\busing\b", 0.12),
            (r"\b(migration|migrat)", 0.1),
        ]

        for pattern, bonus in unique_combinations:
            if re.search(pattern, title_text, re.IGNORECASE):
                score += bonus
                break

        # Penalty for overly common patterns
        common_patterns = [
            r"\b\d+\s+tips?\b",
            r"\bintroduction\s+to\b",
            r"\bgetting\s+started\b",
            r"\bbest\s+practices?\b",
        ]

        for pattern in common_patterns:
            if re.search(pattern, title_text, re.IGNORECASE):
                score -= 0.15
                break

        # Bonus for specific technical combinations
        if len(candidate.key_technologies) >= 2:
            score += 0.1  # Multiple technologies suggest unique approach

        # Penalty for very generic words
        generic_words = ["things", "stuff", "everything", "anything", "something"]
        if any(word in title_text for word in generic_words):
            score -= 0.2

        return max(min(score, 1.0), 0.1)  # Ensure minimum score

    def _calculate_seo_score(
        self, candidate: BlogTitleCandidate, title_text: str
    ) -> float:
        """Calculate SEO score based on search engine optimization potential"""

        score = 0.4  # Base SEO score

        # SEO-friendly patterns
        for pattern_type, pattern in self.seo_patterns.items():
            matches = len(pattern.findall(candidate.title))

            if pattern_type == "questions":
                score += min(matches * 0.15, 0.2)  # Question titles perform well
            elif pattern_type == "lists":
                score += min(matches * 0.12, 0.15)  # List titles are popular
            elif pattern_type == "comparisons":
                score += min(matches * 0.1, 0.12)  # Comparison titles get searches

        # Title length optimization (8-12 words ideal for SEO)
        word_count = len(candidate.title.split())
        if 8 <= word_count <= 12:
            score += 0.15
        elif 6 <= word_count <= 15:
            score += 0.1
        else:
            score -= 0.1

        # Keyword density (avoid keyword stuffing)
        if candidate.key_technologies:
            tech_mentions = sum(
                candidate.title.lower().count(tech.lower())
                for tech in candidate.key_technologies
            )
            if tech_mentions == 1:
                score += 0.1  # Perfect keyword density
            elif tech_mentions > 3:
                score -= 0.15  # Keyword stuffing penalty

        # Search intent alignment
        search_intents = ["how", "why", "what", "best", "vs", "guide", "tutorial"]
        if any(intent in title_text for intent in search_intents):
            score += 0.1

        return min(score, 1.0)

    def _apply_ranking_adjustments(
        self, scores: TitleScores, candidate: BlogTitleCandidate
    ) -> float:
        """Apply final adjustments to create ranking score"""

        ranking_score = scores.overall_score

        # Quality tier bonuses
        if scores.specificity_score > 0.8 and scores.engagement_score > 0.7:
            ranking_score += 0.05  # High-quality combination bonus

        if scores.authority_score > 0.8:
            ranking_score += 0.03  # Authority bonus

        # Pattern-specific adjustments
        if candidate.pattern_type == "performance" and candidate.metrics_mentioned:
            ranking_score += 0.02  # Performance with metrics bonus

        # Penalties
        if len(candidate.title.split()) > 15:
            ranking_score -= 0.05  # Overly long penalty

        if scores.specificity_score < 0.3:
            ranking_score -= 0.03  # Low specificity penalty

        return max(min(ranking_score, 1.0), 0.0)

    def _populate_score_details(
        self, scores: TitleScores, candidate: BlogTitleCandidate
    ):
        """Populate detailed score breakdown and suggestions"""

        scores.score_breakdown = {
            "specificity": scores.specificity_score,
            "engagement": scores.engagement_score,
            "recency": scores.recency_score,
            "authority": scores.authority_score,
            "uniqueness": scores.uniqueness_score,
            "seo": scores.seo_score,
        }

        # Quality indicators
        if scores.specificity_score > 0.7:
            scores.quality_indicators.append("High specificity with concrete details")
        if scores.engagement_score > 0.7:
            scores.quality_indicators.append("Strong engagement potential")
        if scores.authority_score > 0.7:
            scores.quality_indicators.append("High authority and credibility")

        # Improvement suggestions
        if scores.specificity_score < 0.5:
            scores.improvement_suggestions.append(
                "Add more specific metrics, company names, or version numbers"
            )
        if scores.engagement_score < 0.5:
            scores.improvement_suggestions.append(
                "Include more action words or problem-solving language"
            )
        if scores.seo_score < 0.5:
            scores.improvement_suggestions.append(
                "Consider question format or list structure for better SEO"
            )

    def _determine_quality_tier(self, ranking_score: float) -> str:
        """Determine quality tier based on ranking score"""

        if ranking_score >= 0.8:
            return "excellent"
        elif ranking_score >= 0.65:
            return "good"
        elif ranking_score >= 0.5:
            return "fair"
        else:
            return "poor"

    def _analyze_target_metrics(self, ranked_title: RankedBlogTitle):
        """Analyze and extract target metrics from the title"""

        metrics = []

        # Look for performance metrics
        metric_patterns = [
            r"(\d+(?:\.\d+)?%)",  # Percentages
            r"(\d+(?:\.\d+)?\s*x)",  # Multiplication factors
            r"(\d+(?:\.\d+)?\s*ms)",  # Milliseconds
            r"(\d+(?:\.\d+)?\s*seconds?)",  # Seconds
        ]

        for pattern in metric_patterns:
            matches = re.findall(pattern, ranked_title.title)
            metrics.extend(matches)

        ranked_title.target_metrics = metrics[:3]  # Keep top 3 metrics

    def _analyze_competitive_keywords(self, ranked_title: RankedBlogTitle):
        """Extract competitive keywords for content strategy"""

        title_words = ranked_title.title.lower().split()

        # High-value keywords for technical content
        competitive_keywords = [
            word
            for word in title_words
            if len(word) > 3
            and word
            not in [
                "with",
                "using",
                "from",
                "than",
                "that",
                "this",
                "they",
                "them",
                "their",
            ]
        ]

        # Add technology keywords
        tech_keywords = [
            tech.lower() for tech in ranked_title.original_candidate.key_technologies
        ]

        all_keywords = list(set(competitive_keywords + tech_keywords))
        ranked_title.competitive_keywords = all_keywords[:5]  # Keep top 5

    def _estimate_audience_size(self, ranked_title: RankedBlogTitle):
        """Estimate potential audience size for the title"""

        title_lower = ranked_title.title.lower()

        # Large audience indicators
        large_audience_keywords = [
            "javascript",
            "python",
            "react",
            "api",
            "how",
            "guide",
            "tutorial",
        ]

        # Niche audience indicators
        niche_audience_keywords = [
            "kubernetes",
            "graphql",
            "rust",
            "scala",
            "architecture",
            "optimization",
        ]

        large_matches = sum(
            1 for keyword in large_audience_keywords if keyword in title_lower
        )
        niche_matches = sum(
            1 for keyword in niche_audience_keywords if keyword in title_lower
        )

        if large_matches >= 2:
            ranked_title.estimated_audience_size = "large"
        elif niche_matches >= 2:
            ranked_title.estimated_audience_size = "niche"
        else:
            ranked_title.estimated_audience_size = "medium"

    def get_scoring_summary(
        self, ranked_titles: list[RankedBlogTitle]
    ) -> dict[str, Any]:
        """Generate a summary of scoring results for analysis"""

        if not ranked_titles:
            return {"message": "No titles to analyze"}

        # Calculate statistics
        scores = [title.scores.ranking_score for title in ranked_titles]

        summary = {
            "total_titles": len(ranked_titles),
            "score_statistics": {
                "mean": sum(scores) / len(scores),
                "median": sorted(scores)[len(scores) // 2],
                "min": min(scores),
                "max": max(scores),
            },
            "quality_distribution": {
                tier: len([t for t in ranked_titles if t.quality_tier == tier])
                for tier in ["excellent", "good", "fair", "poor"]
            },
            "top_patterns": Counter(
                t.original_candidate.pattern_type for t in ranked_titles
            ).most_common(3),
            "common_improvements": [],
        }

        # Common improvement suggestions
        all_suggestions = []
        for title in ranked_titles:
            all_suggestions.extend(title.scores.improvement_suggestions)

        suggestion_counts = Counter(all_suggestions)
        summary["common_improvements"] = suggestion_counts.most_common(5)

        return summary
