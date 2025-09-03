# src/services/topic_discovery/enhanced_weekly_trend_worker.py
"""
Enhanced Weekly Trend Worker for AI-Powered Blog Title Discovery

This module orchestrates the complete enhanced pipeline for discovering specific,
actionable blog post titles using AI semantic analysis and sophisticated scoring.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ai_semantic_analyzer import AISemanticAnalyzer, SemanticInsight
from .blog_title_generator import BlogTitleCandidate, BlogTitleGenerator
from .config import RSS_SOURCES, get_worker_config
from .context_enricher import ContextEnricher, EnrichedBlogTitle
from .enhanced_content_extractor import ArticleContent, EnhancedContentExtractor
from .pattern_detector import PatternAnalysis, PatternDetector
from .title_scorer_ranker import RankedBlogTitle, TitleScorerRanker
from .weekly_trend_worker import WeeklyTrendWorker  # For S3 upload functionality


class EnhancedWeeklyTrendWorker:
    """
    Enhanced orchestration class that combines all AI-powered components
    to discover specific, actionable blog post titles from RSS sources.
    """

    def __init__(self, config: dict | None = None, openai_api_key: str | None = None):
        """
        Initialize Enhanced Weekly Trend Worker
        
        Args:
            config: Worker configuration (uses defaults if None)
            openai_api_key: OpenAI API key for AI analysis
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config or get_worker_config()
        self.config.update({
            'max_blog_titles_per_run': self.config.get('max_blog_titles_per_run', 30),
            'min_specificity_threshold': self.config.get('min_specificity_threshold', 0.4),
            'enable_ai_analysis': self.config.get('enable_ai_analysis', True),
            'enable_pattern_detection': self.config.get('enable_pattern_detection', True),
            'content_extraction_timeout': self.config.get('content_extraction_timeout', 300),  # 5 minutes
            'ai_analysis_timeout': self.config.get('ai_analysis_timeout', 600),  # 10 minutes
        })

        # Initialize AI components
        self.content_extractor = EnhancedContentExtractor(
            timeout=30,
            max_concurrent=3
        )

        if self.config['enable_ai_analysis']:
            try:
                self.semantic_analyzer = AISemanticAnalyzer(api_key=openai_api_key)
                self.title_generator = BlogTitleGenerator(api_key=openai_api_key)
            except ValueError as e:
                self.logger.error(f"Failed to initialize AI components: {str(e)}")
                self.config['enable_ai_analysis'] = False

        # Initialize analysis components
        self.title_scorer = TitleScorerRanker(
            min_specificity_threshold=self.config['min_specificity_threshold']
        )
        self.context_enricher = ContextEnricher()

        if self.config['enable_pattern_detection']:
            self.pattern_detector = PatternDetector()

        # Initialize legacy worker for S3 functionality
        # Ensure required directories exist before initializing legacy worker
        from pathlib import Path
        status_file_path = Path(self.config['status_file_path'])
        status_file_path.parent.mkdir(parents=True, exist_ok=True)

        backup_dir = Path(self.config['local_backup_dir'])
        backup_dir.mkdir(parents=True, exist_ok=True)

        lock_file_path = Path(self.config['lock_file_path'])
        lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.legacy_worker = WeeklyTrendWorker(self.config)

        self.logger.info("Enhanced Weekly Trend Worker initialized")

    async def run_enhanced_discovery(self) -> dict[str, Any]:
        """
        Main orchestration method for enhanced blog title discovery
        
        Returns:
            Dict containing execution results and analytics
        """
        execution_start = datetime.now(UTC)
        results = {
            'status': 'failed',
            'started_at': execution_start.isoformat(),
            'completed_at': None,
            'blog_titles_discovered': 0,
            'csv_file_uploaded': None,
            'local_backup_file': None,
            'analytics': {},
            'error_message': None
        }

        try:
            self.logger.info("Starting enhanced blog title discovery process")

            # Phase 1: Enhanced Content Extraction
            self.logger.info("Phase 1: Enhanced content extraction from RSS sources")
            article_contents = await self._run_content_extraction()
            self.logger.info(f"Extracted {len(article_contents)} articles with enhanced content")

            # Phase 2: AI Semantic Analysis (if enabled)
            semantic_insights = []
            if self.config['enable_ai_analysis'] and article_contents:
                self.logger.info("Phase 2: AI semantic analysis of article content")
                semantic_insights = await self._run_semantic_analysis(article_contents)
                self.logger.info(f"Generated {len(semantic_insights)} semantic insights")

            # Phase 3: AI Blog Title Generation
            blog_title_candidates = []
            if semantic_insights:
                self.logger.info("Phase 3: AI-powered blog title generation")
                blog_title_candidates = await self._run_title_generation(semantic_insights)
                self.logger.info(f"Generated {len(blog_title_candidates)} blog title candidates")

            # Phase 4: Title Scoring and Ranking
            ranked_titles = []
            if blog_title_candidates:
                self.logger.info("Phase 4: Title scoring and ranking")
                ranked_titles = self._run_title_scoring(blog_title_candidates)
                self.logger.info(f"Ranked {len(ranked_titles)} qualified titles")

            # Phase 5: Context Enrichment
            enriched_titles = []
            if ranked_titles:
                self.logger.info("Phase 5: Context enrichment and supporting information")
                enriched_titles = await self._run_context_enrichment(ranked_titles, semantic_insights)
                self.logger.info(f"Enriched {len(enriched_titles)} titles with comprehensive context")

            # Phase 6: Pattern Detection (if enabled)
            pattern_analysis = None
            if self.config['enable_pattern_detection'] and enriched_titles:
                self.logger.info("Phase 6: Pattern detection and theme identification")
                pattern_analysis = await self._run_pattern_detection(enriched_titles)
                self.logger.info("Completed pattern analysis for theme identification")

            # Phase 7: Enhanced CSV Export
            if enriched_titles:
                self.logger.info("Phase 7: Enhanced CSV export with comprehensive context")
                csv_file_path = await self._export_enhanced_csv(enriched_titles, pattern_analysis)
                results['local_backup_file'] = str(csv_file_path)

                # Phase 8: S3 Upload
                self.logger.info("Phase 8: Uploading enhanced data to S3")
                s3_url = await self.legacy_worker._upload_to_s3(csv_file_path)
                if s3_url:
                    results['csv_file_uploaded'] = s3_url

            # Compile results and analytics
            execution_end = datetime.now(UTC)
            results.update({
                'status': 'completed',
                'completed_at': execution_end.isoformat(),
                'blog_titles_discovered': len(enriched_titles),
                'execution_duration': (execution_end - execution_start).total_seconds(),
                'analytics': self._compile_analytics(
                    article_contents, semantic_insights, blog_title_candidates,
                    ranked_titles, enriched_titles, pattern_analysis
                )
            })

            self.logger.info(f"Enhanced blog title discovery completed successfully in {results['execution_duration']:.1f}s")

        except Exception as e:
            error_msg = f"Error in enhanced blog title discovery: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results['error_message'] = error_msg
            results['completed_at'] = datetime.now(UTC).isoformat()

        # Update status file
        self.legacy_worker._update_status_file(results)

        # Cleanup AI components
        await self._cleanup_ai_components()

        return results

    async def _run_content_extraction(self) -> list[ArticleContent]:
        """Run enhanced content extraction phase"""

        try:
            # Set timeout for content extraction
            return await asyncio.wait_for(
                self.content_extractor.extract_article_content(RSS_SOURCES),
                timeout=self.config['content_extraction_timeout']
            )
        except TimeoutError:
            self.logger.error("Content extraction timed out")
            return []
        except Exception as e:
            self.logger.error(f"Content extraction failed: {str(e)}")
            return []

    async def _run_semantic_analysis(self, article_contents: list[ArticleContent]) -> list[SemanticInsight]:
        """Run AI semantic analysis phase"""

        if not hasattr(self, 'semantic_analyzer'):
            self.logger.warning("Semantic analyzer not available, skipping AI analysis")
            return []

        try:
            # Limit articles for analysis to manage cost and time
            max_articles_for_ai = 20
            selected_articles = self._select_articles_for_ai_analysis(article_contents, max_articles_for_ai)

            # Set timeout for AI analysis
            return await asyncio.wait_for(
                self.semantic_analyzer.analyze_content_semantics(selected_articles),
                timeout=self.config['ai_analysis_timeout']
            )
        except TimeoutError:
            self.logger.error("AI semantic analysis timed out")
            return []
        except Exception as e:
            self.logger.error(f"AI semantic analysis failed: {str(e)}")
            return []

    def _select_articles_for_ai_analysis(self, articles: list[ArticleContent], max_count: int) -> list[ArticleContent]:
        """Select the best articles for AI analysis based on quality metrics"""

        # Sort articles by quality score (from content extraction)
        sorted_articles = sorted(articles, key=lambda x: x.content_quality_score, reverse=True)

        # Also prioritize articles with technical details
        prioritized_articles = []
        for article in sorted_articles:
            priority_score = article.content_quality_score

            # Boost articles with metrics, companies, or technologies
            if article.technical_details.metrics:
                priority_score += 0.2
            if article.technical_details.company_names:
                priority_score += 0.15
            if article.technical_details.technologies:
                priority_score += 0.1

            prioritized_articles.append((article, priority_score))

        # Sort by priority score and return top articles
        prioritized_articles.sort(key=lambda x: x[1], reverse=True)
        return [article for article, _ in prioritized_articles[:max_count]]

    async def _run_title_generation(self, semantic_insights: list[SemanticInsight]) -> list[BlogTitleCandidate]:
        """Run AI blog title generation phase"""

        if not hasattr(self, 'title_generator'):
            self.logger.warning("Title generator not available, skipping title generation")
            return []

        try:
            max_titles = self.config['max_blog_titles_per_run']
            return await self.title_generator.generate_specific_titles(semantic_insights, max_titles)
        except Exception as e:
            self.logger.error(f"Title generation failed: {str(e)}")
            return []

    def _run_title_scoring(self, blog_title_candidates: list[BlogTitleCandidate]) -> list[RankedBlogTitle]:
        """Run title scoring and ranking phase"""

        try:
            return self.title_scorer.rank_titles(blog_title_candidates)
        except Exception as e:
            self.logger.error(f"Title scoring failed: {str(e)}")
            return []

    async def _run_context_enrichment(self, ranked_titles: list[RankedBlogTitle],
                                    semantic_insights: list[SemanticInsight]) -> list[EnrichedBlogTitle]:
        """Run context enrichment phase"""

        try:
            return await self.context_enricher.enrich_titles_context(ranked_titles, semantic_insights)
        except Exception as e:
            self.logger.error(f"Context enrichment failed: {str(e)}")
            return []

    async def _run_pattern_detection(self, enriched_titles: list[EnrichedBlogTitle]) -> PatternAnalysis | None:
        """Run pattern detection and theme identification phase"""

        if not hasattr(self, 'pattern_detector'):
            return None

        try:
            # Detect emerging themes
            themes = await self.pattern_detector.detect_emerging_themes(enriched_titles)

            # Suggest content series
            series = await self.pattern_detector.suggest_content_series(enriched_titles)

            # Identify trend connections
            connections = await self.pattern_detector.identify_trend_connections(enriched_titles)

            # Compile pattern analysis
            analysis = PatternAnalysis(
                emerging_themes=themes,
                content_series=series,
                trend_connections=connections,
                titles_analyzed=len(enriched_titles),
                confidence_score=sum(t.scores.ranking_score for t in enriched_titles) / len(enriched_titles) if enriched_titles else 0
            )

            # Extract additional analytics
            analysis.dominant_categories = self._extract_dominant_categories(enriched_titles)
            analysis.trending_technologies = self._extract_trending_technologies(enriched_titles)
            analysis.content_gaps = self._identify_content_gaps(themes, enriched_titles)

            return analysis

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return None

    def _extract_dominant_categories(self, enriched_titles: list[EnrichedBlogTitle]) -> list[str]:
        """Extract dominant content categories"""

        from collections import Counter

        categories = [title.context.content_category for title in enriched_titles]
        category_counts = Counter(categories)
        return [category for category, count in category_counts.most_common(5)]

    def _extract_trending_technologies(self, enriched_titles: list[EnrichedBlogTitle]) -> list[str]:
        """Extract trending technologies across all titles"""

        from collections import Counter

        all_technologies = []
        for title in enriched_titles:
            all_technologies.extend(title.context.key_technologies)

        tech_counts = Counter(all_technologies)
        return [tech for tech, count in tech_counts.most_common(10)]

    def _identify_content_gaps(self, themes: list, enriched_titles: list[EnrichedBlogTitle]) -> list[str]:
        """Identify potential content gaps based on theme analysis"""

        # This is a simplified implementation - could be enhanced with more sophisticated analysis
        covered_topics = set()
        for title in enriched_titles:
            covered_topics.update([tech.lower() for tech in title.context.key_technologies])

        # Common topics that might be underrepresented
        important_topics = [
            "kubernetes", "docker", "microservices", "graphql", "typescript",
            "machine learning", "ai", "security", "performance", "testing"
        ]

        gaps = [topic for topic in important_topics if topic not in covered_topics]
        return gaps[:5]  # Return top 5 gaps

    async def _export_enhanced_csv(self, enriched_titles: list[EnrichedBlogTitle],
                                 pattern_analysis: PatternAnalysis | None) -> Path:
        """Export enhanced blog titles to CSV with comprehensive context"""

        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced-blog-titles-{timestamp}.csv"
        csv_file_path = Path(self.config['local_backup_dir']) / filename

        # Ensure backup directory exists
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Enhanced CSV columns as specified in the design document
        csv_columns = [
            'blog_title', 'specificity_score', 'engagement_score', 'overall_score',
            'source_articles', 'key_technologies', 'supporting_metrics', 'content_angle',
            'target_audience', 'technical_depth', 'emerging_theme', 'series_potential',
            'discovered_at', 'confidence_level', 'pattern_type', 'seo_keywords',
            'writing_complexity', 'research_depth_required', 'estimated_word_count',
            'content_category', 'trending_score', 'evergreen_potential'
        ]

        import csv

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

            for enriched_title in enriched_titles:
                # Find related emerging theme
                emerging_theme = ""
                if pattern_analysis and pattern_analysis.emerging_themes:
                    for theme in pattern_analysis.emerging_themes:
                        if enriched_title.title in theme.related_titles:
                            emerging_theme = theme.theme_name
                            break

                # Prepare row data
                row = {
                    'blog_title': enriched_title.title,
                    'specificity_score': round(enriched_title.ranked_title.scores.specificity_score, 3),
                    'engagement_score': round(enriched_title.ranked_title.scores.engagement_score, 3),
                    'overall_score': round(enriched_title.ranked_title.scores.overall_score, 3),
                    'source_articles': '; '.join(enriched_title.context.source_articles[:3]),  # Limit to 3 sources
                    'key_technologies': ', '.join(enriched_title.context.key_technologies[:5]),
                    'supporting_metrics': '; '.join(enriched_title.supporting_details.performance_metrics[:3]),
                    'content_angle': enriched_title.content_angles[0].angle_type if enriched_title.content_angles else '',
                    'target_audience': enriched_title.context.target_audience,
                    'technical_depth': enriched_title.context.technical_depth,
                    'emerging_theme': emerging_theme,
                    'series_potential': '; '.join(enriched_title.series_potential[:3]),
                    'discovered_at': datetime.now(UTC).isoformat(),
                    'confidence_level': 'high' if enriched_title.ranked_title.scores.overall_score > 0.7 else 'medium' if enriched_title.ranked_title.scores.overall_score > 0.5 else 'low',
                    'pattern_type': enriched_title.ranked_title.original_candidate.pattern_type,
                    'seo_keywords': ', '.join(enriched_title.seo_keywords[:5]),
                    'writing_complexity': enriched_title.writing_complexity,
                    'research_depth_required': enriched_title.research_depth_required,
                    'estimated_word_count': enriched_title.content_angles[0].estimated_word_count if enriched_title.content_angles else 1500,
                    'content_category': enriched_title.context.content_category,
                    'trending_score': round(enriched_title.context.trending_score, 3),
                    'evergreen_potential': round(enriched_title.context.evergreen_potential, 3)
                }

                writer.writerow(row)

        self.logger.info(f"Exported {len(enriched_titles)} enhanced blog titles to {csv_file_path}")
        return csv_file_path

    def _compile_analytics(self, article_contents: list[ArticleContent],
                          semantic_insights: list[SemanticInsight],
                          blog_title_candidates: list[BlogTitleCandidate],
                          ranked_titles: list[RankedBlogTitle],
                          enriched_titles: list[EnrichedBlogTitle],
                          pattern_analysis: PatternAnalysis | None) -> dict[str, Any]:
        """Compile comprehensive analytics from the discovery process"""

        analytics = {
            'content_extraction': {
                'total_articles_extracted': len(article_contents),
                'avg_content_quality_score': sum(a.content_quality_score for a in article_contents) / len(article_contents) if article_contents else 0,
                'articles_with_full_content': sum(1 for a in article_contents if a.full_content),
                'avg_word_count': sum(a.word_count for a in article_contents) / len(article_contents) if article_contents else 0
            },
            'ai_analysis': {
                'semantic_insights_generated': len(semantic_insights),
                'avg_confidence_score': sum(s.confidence_score for s in semantic_insights) / len(semantic_insights) if semantic_insights else 0,
                'total_technical_concepts': sum(len(s.technical_concepts) for s in semantic_insights),
                'total_performance_metrics': sum(len(s.performance_metrics) for s in semantic_insights)
            },
            'title_generation': {
                'total_title_candidates': len(blog_title_candidates),
                'titles_by_pattern': self._analyze_title_patterns(blog_title_candidates),
                'avg_ai_confidence': sum(c.confidence for c in blog_title_candidates) / len(blog_title_candidates) if blog_title_candidates else 0
            },
            'title_scoring': {
                'titles_ranked': len(ranked_titles),
                'avg_specificity_score': sum(t.scores.specificity_score for t in ranked_titles) / len(ranked_titles) if ranked_titles else 0,
                'avg_engagement_score': sum(t.scores.engagement_score for t in ranked_titles) / len(ranked_titles) if ranked_titles else 0,
                'quality_distribution': self._analyze_quality_distribution(ranked_titles)
            },
            'context_enrichment': {
                'titles_enriched': len(enriched_titles),
                'content_angles_suggested': sum(len(t.content_angles) for t in enriched_titles),
                'series_opportunities': sum(len(t.series_potential) for t in enriched_titles)
            }
        }

        # Add pattern analysis if available
        if pattern_analysis:
            analytics['pattern_detection'] = {
                'emerging_themes_identified': len(pattern_analysis.emerging_themes),
                'content_series_suggested': len(pattern_analysis.content_series),
                'trend_connections_found': len(pattern_analysis.trend_connections),
                'dominant_categories': pattern_analysis.dominant_categories,
                'trending_technologies': pattern_analysis.trending_technologies[:5]
            }

        return analytics

    def _analyze_title_patterns(self, candidates: list[BlogTitleCandidate]) -> dict[str, int]:
        """Analyze distribution of title patterns"""

        from collections import Counter
        patterns = [c.pattern_type for c in candidates]
        return dict(Counter(patterns))

    def _analyze_quality_distribution(self, ranked_titles: list[RankedBlogTitle]) -> dict[str, int]:
        """Analyze distribution of title quality tiers"""

        from collections import Counter
        quality_tiers = [t.quality_tier for t in ranked_titles]
        return dict(Counter(quality_tiers))

    async def _cleanup_ai_components(self):
        """Cleanup AI components to free resources"""

        try:
            if hasattr(self, 'semantic_analyzer'):
                await self.semantic_analyzer.close()
            if hasattr(self, 'title_generator'):
                await self.title_generator.close()
        except Exception as e:
            self.logger.warning(f"Error during AI component cleanup: {str(e)}")

    async def run_dry_run(self) -> dict[str, Any]:
        """
        Run a dry-run of the enhanced discovery process for testing
        
        Returns:
            Dict containing dry-run results without S3 upload
        """
        self.logger.info("Running enhanced blog title discovery in dry-run mode")

        # Temporarily disable S3 upload
        original_s3_client = self.legacy_worker.s3_client
        self.legacy_worker.s3_client = None

        try:
            results = await self.run_enhanced_discovery()
            results['dry_run'] = True
            return results
        finally:
            # Restore S3 client
            self.legacy_worker.s3_client = original_s3_client

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of the current configuration"""

        return {
            'ai_analysis_enabled': self.config['enable_ai_analysis'],
            'pattern_detection_enabled': self.config['enable_pattern_detection'],
            'max_blog_titles_per_run': self.config['max_blog_titles_per_run'],
            'min_specificity_threshold': self.config['min_specificity_threshold'],
            'rss_sources_count': len(RSS_SOURCES),
            'content_extraction_timeout': self.config['content_extraction_timeout'],
            'ai_analysis_timeout': self.config['ai_analysis_timeout']
        }
