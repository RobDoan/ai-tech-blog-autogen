# tests/test_blog_title_generator.py
"""
Unit tests for Blog Title Generator
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.topic_discovery.blog_title_generator import (
    BlogTitleGenerator, BlogTitleCandidate
)
from src.services.topic_discovery.ai_semantic_analyzer import (
    SemanticInsight, TechnicalConcept, ImplicitTopic
)


class TestBlogTitleGenerator:
    """Test cases for BlogTitleGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance with mock API key"""
        with patch('src.services.topic_discovery.blog_title_generator.OPENAI_API_KEY', 'test-key'):
            return BlogTitleGenerator(api_key='test-api-key')
    
    @pytest.fixture
    def sample_insight(self):
        """Sample semantic insight for testing"""
        return SemanticInsight(
            article_id="test123",
            source_article="https://example.com/article",
            implicit_topics=[
                ImplicitTopic("GraphQL Performance", 0.9, "API optimization", "advanced")
            ],
            technical_concepts=[
                TechnicalConcept(
                    concept="API Federation",
                    implementation_approach="GraphQL federation with Apollo",
                    problem_solved="High API latency",
                    technologies_used=["GraphQL", "Apollo", "Node.js"],
                    business_impact="40% latency reduction"
                )
            ],
            problems_solved=["API latency", "Complex data fetching"],
            solutions_implemented=["GraphQL federation", "Query optimization"],
            performance_metrics=["40% latency reduction", "25% fewer API calls"],
            key_insights=[
                "Federation enables distributed architecture",
                "Query batching improves performance"
            ],
            target_audience="senior_developers",
            content_angle="case_study",
            confidence_score=0.85
        )
    
    @pytest.fixture
    def mock_ai_title_response(self):
        """Mock AI title generation response"""
        return {
            "titles": [
                {
                    "title": "How Netflix Reduced API Latency by 40% with GraphQL Federation",
                    "pattern_type": "performance",
                    "technologies": ["GraphQL", "Apollo", "Node.js"],
                    "metrics": ["40% latency reduction"],
                    "companies": ["Netflix"],
                    "confidence": 0.9,
                    "reasoning": "Specific metric, company, and technology mentioned"
                },
                {
                    "title": "GraphQL vs REST: Performance Comparison for High-Scale APIs",
                    "pattern_type": "comparison",
                    "technologies": ["GraphQL", "REST"],
                    "metrics": ["25% fewer calls"],
                    "companies": [],
                    "confidence": 0.8,
                    "reasoning": "Clear comparison with performance data"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_generate_specific_titles_success(self, generator, sample_insight, mock_ai_title_response):
        """Test successful title generation from insights"""
        
        with patch.object(generator, '_generate_titles_for_insight') as mock_generate:
            mock_candidates = [
                BlogTitleCandidate(
                    title="How Netflix Reduced API Latency by 40% with GraphQL Federation",
                    pattern_type="performance",
                    confidence=0.9
                ),
                BlogTitleCandidate(
                    title="GraphQL Federation: Building Scalable API Architecture",
                    pattern_type="implementation",
                    confidence=0.8
                )
            ]
            mock_generate.return_value = mock_candidates
            
            # Execute title generation
            titles = await generator.generate_specific_titles([sample_insight], max_titles=10)
            
            # Assertions
            assert len(titles) == 2
            assert all(isinstance(t, BlogTitleCandidate) for t in titles)
            assert titles[0].confidence > titles[1].confidence  # Should be sorted by quality
    
    @pytest.mark.asyncio
    async def test_generate_ai_titles_success(self, generator, sample_insight, mock_ai_title_response):
        """Test AI-powered title generation"""
        
        with patch.object(generator.client.chat.completions, 'create') as mock_create:
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"titles": []}'
            mock_create.return_value = mock_response
            
            with patch('json.loads', return_value=mock_ai_title_response):
                patterns = ["performance", "implementation"]
                
                titles = await generator._generate_ai_titles(sample_insight, patterns)
                
                # Check results
                assert len(titles) == 2
                assert isinstance(titles[0], BlogTitleCandidate)
                assert "Netflix" in titles[0].title
                assert titles[0].pattern_type == "performance"
                assert titles[0].confidence == 0.9
    
    def test_identify_suitable_patterns_performance(self, generator, sample_insight):
        """Test pattern identification for performance-focused content"""
        
        # Modify insight to emphasize performance
        sample_insight.performance_metrics = ["40% improvement", "25% reduction"]
        sample_insight.key_insights = ["improved latency", "reduced response time"]
        
        patterns = generator._identify_suitable_patterns(sample_insight)
        
        assert "performance" in patterns
    
    def test_identify_suitable_patterns_comparison(self, generator, sample_insight):
        """Test pattern identification for comparison content"""
        
        # Add multiple technologies to suggest comparison
        sample_insight.technical_concepts.append(
            TechnicalConcept(
                concept="REST API Design",
                technologies_used=["REST", "OpenAPI"],
                implementation_approach="Traditional REST implementation",
                problem_solved="API structure"
            )
        )
        
        patterns = generator._identify_suitable_patterns(sample_insight)
        
        assert "comparison" in patterns
    
    def test_generate_template_titles_performance(self, generator, sample_insight):
        """Test template-based title generation for performance pattern"""
        
        patterns = ["performance"]
        titles = generator._generate_template_titles(sample_insight, patterns)
        
        assert len(titles) > 0
        performance_titles = [t for t in titles if t.pattern_type == "performance"]
        assert len(performance_titles) > 0
        
        # Check that templates were populated with insight data
        title_text = performance_titles[0].title
        assert any(tech in title_text for tech in ["GraphQL", "API", "performance"])
    
    def test_substitute_template_variables(self, generator):
        """Test template variable substitution"""
        
        template = "How {company} improved {metric} using {technology}"
        technologies = ["GraphQL", "Apollo"]
        companies = ["Netflix"]
        metrics = ["API latency"]
        problems = []
        
        result = generator._substitute_template_variables(
            template, technologies, companies, metrics, problems
        )
        
        assert result is not None
        assert "Netflix" in result
        assert "GraphQL" in result
        # Should handle metric gracefully even if not in exact format
    
    def test_analyze_title_metrics(self, generator, sample_insight):
        """Test title metric analysis and scoring"""
        
        candidate = BlogTitleCandidate(
            title="How Netflix Reduced API Latency by 40% with GraphQL v2.1",
            pattern_type="performance",
            source_insights=[sample_insight.article_id]
        )
        
        generator._analyze_title_metrics(candidate, sample_insight)
        
        # Check specificity score (should be high due to company, metric, version)
        assert candidate.specificity_score > 0.6
        
        # Check engagement score (should be good for "how" + improvement)
        assert candidate.engagement_score > 0.4
    
    def test_deduplicate_titles_exact_duplicates(self, generator):
        """Test deduplication of exact duplicate titles"""
        
        titles = [
            BlogTitleCandidate(
                title="How to Use GraphQL",
                pattern_type="how-to",
                specificity_score=0.5,
                engagement_score=0.6
            ),
            BlogTitleCandidate(
                title="How to Use GraphQL",
                pattern_type="tutorial",
                specificity_score=0.7,
                engagement_score=0.5
            )
        ]
        
        unique_titles = generator._deduplicate_titles(titles)
        
        # Should keep only the better scoring version
        assert len(unique_titles) == 1
        assert unique_titles[0].specificity_score == 0.7  # Better score kept
    
    def test_deduplicate_titles_similar_titles(self, generator):
        """Test deduplication of very similar titles"""
        
        titles = [
            BlogTitleCandidate(
                title="GraphQL Performance Optimization Guide",
                pattern_type="tutorial",
                specificity_score=0.6,
                engagement_score=0.7
            ),
            BlogTitleCandidate(
                title="GraphQL Performance Optimization Tutorial",
                pattern_type="tutorial",
                specificity_score=0.5,
                engagement_score=0.6
            )
        ]
        
        unique_titles = generator._deduplicate_titles(titles)
        
        # Should remove very similar titles
        assert len(unique_titles) == 1
        assert unique_titles[0].engagement_score == 0.7  # Better score kept
    
    def test_prepare_insight_for_ai(self, generator, sample_insight):
        """Test preparation of insight data for AI processing"""
        
        patterns = ["performance", "implementation"]
        prepared = generator._prepare_insight_for_ai(sample_insight, patterns)
        
        # Check required fields
        assert "key_insights" in prepared
        assert "technical_concepts" in prepared
        assert "performance_metrics" in prepared
        assert "suitable_patterns" in prepared
        
        # Check data extraction
        assert len(prepared["key_insights"]) > 0
        assert len(prepared["performance_metrics"]) > 0
        assert "performance" in prepared["suitable_patterns"]
    
    def test_get_title_generation_system_prompt(self, generator):
        """Test system prompt for title generation"""
        
        prompt = generator._get_title_generation_system_prompt()
        
        # Check key elements
        assert "blog title generator" in prompt.lower()
        assert "json" in prompt.lower()
        assert "specific" in prompt.lower()
        assert "actionable" in prompt.lower()
    
    def test_build_title_generation_prompt(self, generator):
        """Test building title generation prompt"""
        
        insight_data = {
            "key_insights": ["GraphQL improves performance"],
            "technical_concepts": [],
            "performance_metrics": ["40% improvement"],
            "technologies": ["GraphQL"],
            "companies": ["Netflix"],
            "target_audience": "developers",
            "content_angle": "case_study",
            "suitable_patterns": ["performance"]
        }
        patterns = ["performance"]
        
        prompt = generator._build_title_generation_prompt(insight_data, patterns)
        
        # Check prompt contains key information
        assert "40% improvement" in prompt
        assert "GraphQL" in prompt
        assert "Netflix" in prompt
        assert "performance" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_titles_empty_insights(self, generator):
        """Test title generation with empty insights"""
        
        titles = await generator.generate_specific_titles([])
        
        assert titles == []
    
    @pytest.mark.asyncio
    async def test_generate_titles_low_quality_insights(self, generator):
        """Test title generation with low quality insights"""
        
        low_quality_insight = SemanticInsight(
            article_id="test",
            source_article="https://example.com",
            confidence_score=0.2,  # Very low confidence
            technical_concepts=[],  # No technical concepts
            performance_metrics=[],  # No metrics
            key_insights=[]  # No insights
        )
        
        titles = await generator.generate_specific_titles([low_quality_insight])
        
        # Should return empty for very low quality
        assert len(titles) == 0
    
    @pytest.mark.asyncio
    async def test_generate_titles_with_api_error(self, generator, sample_insight):
        """Test title generation with AI API errors"""
        
        with patch.object(generator, '_generate_ai_titles') as mock_ai:
            mock_ai.side_effect = Exception("API Error")
            
            with patch.object(generator, '_generate_template_titles') as mock_template:
                mock_template.return_value = [
                    BlogTitleCandidate(
                        title="GraphQL Development Guide",
                        pattern_type="tutorial",
                        generated_by="template_generation",
                        specificity_score=0.4
                    )
                ]
                
                titles = await generator.generate_specific_titles([sample_insight])
                
                # Should fall back to template generation
                assert len(titles) > 0
                assert titles[0].generated_by == "template_generation"
    
    @pytest.mark.asyncio
    async def test_close_client(self, generator):
        """Test proper client cleanup"""
        
        with patch.object(generator.client, 'close') as mock_close:
            await generator.close()
            mock_close.assert_called_once()
    
    def test_title_templates_initialization(self, generator):
        """Test that title templates are properly initialized"""
        
        assert "performance" in generator.title_templates
        assert "comparison" in generator.title_templates
        assert "implementation" in generator.title_templates
        assert "how-to" in generator.title_templates
        
        # Check that templates contain placeholder variables
        performance_templates = generator.title_templates["performance"]
        assert any("{company}" in template for template in performance_templates)
        assert any("{technology}" in template for template in performance_templates)