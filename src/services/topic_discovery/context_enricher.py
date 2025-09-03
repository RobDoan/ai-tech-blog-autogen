# src/services/topic_discovery/context_enricher.py
"""
Context Enricher for Blog Title Discovery

This module enriches blog titles with comprehensive supporting context, including
technical details, implementation guidance, content angles, and target audience analysis.
"""

import logging
from dataclasses import dataclass, field

from .ai_semantic_analyzer import SemanticInsight
from .title_scorer_ranker import RankedBlogTitle


@dataclass
class SupportingDetails:
    """Supporting technical details and context for a blog title"""
    code_examples: list[str] = field(default_factory=list)  # Relevant code snippets
    performance_metrics: list[str] = field(default_factory=list)  # Performance data
    implementation_challenges: list[str] = field(default_factory=list)  # Known difficulties
    business_impact: str = ""  # Business value and ROI
    prerequisites: list[str] = field(default_factory=list)  # Required knowledge/tools
    related_concepts: list[str] = field(default_factory=list)  # Connected topics
    external_resources: list[str] = field(default_factory=list)  # Useful links/docs


@dataclass
class ContentAngle:
    """Suggested angle/approach for covering the blog title topic"""
    angle_type: str  # "tutorial", "case_study", "comparison", "deep_dive", "overview"
    description: str
    target_sections: list[str] = field(default_factory=list)  # Suggested article sections
    estimated_word_count: int = 1500  # Suggested article length
    difficulty_level: str = "intermediate"  # "beginner", "intermediate", "advanced"
    time_to_read: str = "8 min"  # Estimated reading time


@dataclass
class TitleContext:
    """Comprehensive context information for a blog title"""
    source_articles: list[str] = field(default_factory=list)  # Original article URLs
    key_technologies: list[str] = field(default_factory=list)  # Main technologies
    target_audience: str = "developers"  # Primary audience
    technical_depth: str = "intermediate"  # Required technical background
    content_category: str = "development"  # "development", "architecture", "devops", etc.

    # Timing and relevance
    trending_score: float = 0.5  # How trending/timely this topic is
    evergreen_potential: float = 0.5  # How long-lasting the content value is
    seasonal_relevance: str | None = None  # "Q4", "conference_season", etc.

    # Competitive analysis
    content_saturation: str = "medium"  # "low", "medium", "high"
    unique_angle_score: float = 0.5  # How unique/differentiated the angle is


@dataclass
class EnrichedBlogTitle:
    """Blog title with comprehensive contextual enrichment"""
    title: str
    ranked_title: RankedBlogTitle

    # Enriched context
    context: TitleContext = field(default_factory=TitleContext)
    supporting_details: SupportingDetails = field(default_factory=SupportingDetails)
    content_angles: list[ContentAngle] = field(default_factory=list)

    # Content strategy
    series_potential: list[str] = field(default_factory=list)  # Related topics for series
    content_clusters: list[str] = field(default_factory=list)  # Topic clusters it belongs to
    seo_keywords: list[str] = field(default_factory=list)  # SEO keyword suggestions

    # Editorial guidance
    writing_complexity: str = "medium"  # "low", "medium", "high"
    research_depth_required: str = "moderate"  # "light", "moderate", "extensive"
    multimedia_suggestions: list[str] = field(default_factory=list)  # Diagrams, code blocks, etc.


class ContextEnricher:
    """
    Context enrichment system that adds comprehensive supporting information,
    content angles, and editorial guidance to blog titles.
    """

    def __init__(self):
        """Initialize Context Enricher"""
        self.logger = logging.getLogger(__name__)

        # Initialize content frameworks and templates
        self._initialize_content_frameworks()
        self._initialize_audience_profiles()
        self._initialize_technology_taxonomies()

        self.logger.info("Context Enricher initialized")

    def _initialize_content_frameworks(self):
        """Initialize content angle frameworks and templates"""

        self.content_frameworks = {
            "tutorial": {
                "sections": [
                    "Prerequisites and Setup",
                    "Step-by-Step Implementation",
                    "Common Pitfalls and Solutions",
                    "Testing and Validation",
                    "Next Steps and Advanced Topics"
                ],
                "word_count_range": (2000, 3500),
                "multimedia": ["code_blocks", "screenshots", "diagrams"]
            },
            "case_study": {
                "sections": [
                    "Problem Statement and Context",
                    "Solution Architecture",
                    "Implementation Details",
                    "Results and Metrics",
                    "Lessons Learned"
                ],
                "word_count_range": (1500, 2500),
                "multimedia": ["architecture_diagrams", "performance_charts", "code_snippets"]
            },
            "comparison": {
                "sections": [
                    "Comparison Criteria",
                    "Option A: Strengths and Weaknesses",
                    "Option B: Strengths and Weaknesses",
                    "Performance Benchmarks",
                    "Recommendations by Use Case"
                ],
                "word_count_range": (1800, 2800),
                "multimedia": ["comparison_tables", "benchmarks", "decision_trees"]
            },
            "deep_dive": {
                "sections": [
                    "Foundational Concepts",
                    "Technical Architecture",
                    "Advanced Implementation Patterns",
                    "Performance Optimization",
                    "Future Considerations"
                ],
                "word_count_range": (2500, 4000),
                "multimedia": ["technical_diagrams", "code_examples", "flowcharts"]
            },
            "overview": {
                "sections": [
                    "Introduction and Context",
                    "Key Features and Benefits",
                    "Getting Started Guide",
                    "Real-World Applications",
                    "Additional Resources"
                ],
                "word_count_range": (1200, 2000),
                "multimedia": ["infographics", "feature_screenshots", "quick_examples"]
            }
        }

    def _initialize_audience_profiles(self):
        """Initialize target audience profiles and characteristics"""

        self.audience_profiles = {
            "junior_developers": {
                "technical_depth": "beginner",
                "preferred_content": ["tutorials", "overviews", "getting_started"],
                "complexity_tolerance": "low",
                "common_pain_points": [
                    "Understanding fundamentals",
                    "Setting up development environment",
                    "Following best practices",
                    "Debugging basic issues"
                ]
            },
            "senior_developers": {
                "technical_depth": "advanced",
                "preferred_content": ["deep_dive", "case_study", "comparison"],
                "complexity_tolerance": "high",
                "common_pain_points": [
                    "Scalability challenges",
                    "Architecture decisions",
                    "Performance optimization",
                    "Technology evaluation"
                ]
            },
            "tech_leads": {
                "technical_depth": "intermediate",
                "preferred_content": ["case_study", "comparison", "overview"],
                "complexity_tolerance": "medium",
                "common_pain_points": [
                    "Team productivity",
                    "Technology adoption",
                    "Risk assessment",
                    "Resource planning"
                ]
            },
            "devops_engineers": {
                "technical_depth": "intermediate",
                "preferred_content": ["tutorial", "case_study", "deep_dive"],
                "complexity_tolerance": "high",
                "common_pain_points": [
                    "Deployment automation",
                    "Monitoring and observability",
                    "Infrastructure scaling",
                    "Security compliance"
                ]
            }
        }

    def _initialize_technology_taxonomies(self):
        """Initialize technology categories and relationships"""

        self.tech_taxonomy = {
            "frontend": {
                "frameworks": ["React", "Vue", "Angular", "Svelte"],
                "build_tools": ["Webpack", "Vite", "Parcel", "Rollup"],
                "state_management": ["Redux", "Vuex", "MobX", "Zustand"],
                "related_concepts": ["SPA", "PWA", "SSR", "SSG", "hydration"]
            },
            "backend": {
                "languages": ["Python", "Node.js", "Java", "Go", "Rust"],
                "frameworks": ["Express", "FastAPI", "Spring", "Django", "Flask"],
                "databases": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"],
                "related_concepts": ["API design", "microservices", "caching", "authentication"]
            },
            "cloud": {
                "providers": ["AWS", "Azure", "GCP", "Vercel", "Netlify"],
                "services": ["Lambda", "S3", "CloudFront", "RDS", "EKS"],
                "patterns": ["serverless", "containerization", "infrastructure_as_code"],
                "related_concepts": ["scalability", "availability", "cost_optimization"]
            },
            "devops": {
                "tools": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions"],
                "practices": ["CI/CD", "monitoring", "logging", "testing"],
                "platforms": ["Terraform", "Ansible", "Prometheus", "Grafana"],
                "related_concepts": ["deployment", "observability", "automation", "security"]
            }
        }

    async def enrich_titles_context(self, ranked_titles: list[RankedBlogTitle],
                                   insights: list[SemanticInsight] | None = None) -> list[EnrichedBlogTitle]:
        """
        Enrich blog titles with comprehensive context and supporting information
        
        Args:
            ranked_titles: List of ranked blog titles to enrich
            insights: Optional semantic insights for additional context
            
        Returns:
            List of enriched blog titles with comprehensive context
        """
        if not ranked_titles:
            return []

        self.logger.info(f"Enriching context for {len(ranked_titles)} blog titles")

        # Create insight lookup for efficient access
        insight_lookup = {}
        if insights:
            for insight in insights:
                insight_lookup[insight.article_id] = insight

        enriched_titles = []
        for ranked_title in ranked_titles:
            try:
                # Find related insights
                related_insights = self._find_related_insights(ranked_title, insight_lookup)

                # Create enriched title
                enriched = await self._enrich_single_title(ranked_title, related_insights)
                enriched_titles.append(enriched)

            except Exception as e:
                self.logger.error(f"Error enriching title '{ranked_title.title}': {str(e)}")
                # Create basic enriched title as fallback
                fallback = EnrichedBlogTitle(
                    title=ranked_title.title,
                    ranked_title=ranked_title
                )
                enriched_titles.append(fallback)

        # Post-process for cross-title analysis
        self._analyze_content_clusters(enriched_titles)
        self._identify_series_opportunities(enriched_titles)

        self.logger.info(f"Successfully enriched {len(enriched_titles)} blog titles with context")
        return enriched_titles

    def _find_related_insights(self, ranked_title: RankedBlogTitle,
                              insight_lookup: dict[str, SemanticInsight]) -> list[SemanticInsight]:
        """Find semantic insights related to the blog title"""

        related_insights = []

        # Direct insight relationships
        for insight_id in ranked_title.original_candidate.source_insights:
            if insight_id in insight_lookup:
                related_insights.append(insight_lookup[insight_id])

        return related_insights

    async def _enrich_single_title(self, ranked_title: RankedBlogTitle,
                                  related_insights: list[SemanticInsight]) -> EnrichedBlogTitle:
        """Enrich a single blog title with comprehensive context"""

        enriched = EnrichedBlogTitle(
            title=ranked_title.title,
            ranked_title=ranked_title
        )

        # Enrich context information
        enriched.context = self._build_title_context(ranked_title, related_insights)

        # Extract and enrich supporting details
        enriched.supporting_details = self._extract_supporting_details(ranked_title, related_insights)

        # Generate content angles
        enriched.content_angles = self._generate_content_angles(ranked_title, related_insights)

        # Analyze SEO opportunities
        enriched.seo_keywords = self._extract_seo_keywords(ranked_title)

        # Assess editorial requirements
        enriched.writing_complexity = self._assess_writing_complexity(ranked_title)
        enriched.research_depth_required = self._assess_research_depth(ranked_title, related_insights)

        # Generate multimedia suggestions
        enriched.multimedia_suggestions = self._suggest_multimedia_elements(ranked_title)

        return enriched

    def _build_title_context(self, ranked_title: RankedBlogTitle,
                           related_insights: list[SemanticInsight]) -> TitleContext:
        """Build comprehensive context information for the title"""

        context = TitleContext()

        # Extract source articles
        context.source_articles = [insight.source_article for insight in related_insights]

        # Determine key technologies
        all_technologies = set(ranked_title.original_candidate.key_technologies)
        for insight in related_insights:
            for concept in insight.technical_concepts:
                all_technologies.update(concept.technologies_used)
        context.key_technologies = list(all_technologies)[:5]  # Top 5

        # Determine target audience and technical depth
        context.target_audience = self._determine_target_audience(ranked_title, related_insights)
        context.technical_depth = ranked_title.original_candidate.technical_depth

        # Categorize content
        context.content_category = self._categorize_content(ranked_title, context.key_technologies)

        # Assess trending and evergreen potential
        context.trending_score = self._calculate_trending_score(ranked_title, related_insights)
        context.evergreen_potential = self._calculate_evergreen_potential(ranked_title)

        # Analyze competitive landscape
        context.content_saturation = self._assess_content_saturation(ranked_title)
        context.unique_angle_score = ranked_title.scores.uniqueness_score

        return context

    def _determine_target_audience(self, ranked_title: RankedBlogTitle,
                                 related_insights: list[SemanticInsight]) -> str:
        """Determine the primary target audience for the content"""

        title_lower = ranked_title.title.lower()

        # Audience indicators in title
        if any(word in title_lower for word in ['beginner', 'getting started', 'introduction', 'basics']):
            return "junior_developers"
        elif any(word in title_lower for word in ['advanced', 'optimization', 'architecture', 'scalability']):
            return "senior_developers"
        elif any(word in title_lower for word in ['devops', 'deployment', 'ci/cd', 'infrastructure']):
            return "devops_engineers"
        elif any(word in title_lower for word in ['team', 'management', 'strategy', 'decision']):
            return "tech_leads"

        # Fallback to technical depth
        technical_depth = ranked_title.original_candidate.technical_depth
        if technical_depth == "beginner":
            return "junior_developers"
        elif technical_depth == "advanced":
            return "senior_developers"
        else:
            return "developers"  # General developers

    def _categorize_content(self, ranked_title: RankedBlogTitle, technologies: list[str]) -> str:
        """Categorize the content based on technologies and context"""

        tech_lower = [tech.lower() for tech in technologies]

        # Check technology categories
        for category, tech_data in self.tech_taxonomy.items():
            category_techs = []
            for tech_list in tech_data.values():
                if isinstance(tech_list, list):
                    category_techs.extend([t.lower() for t in tech_list])

            if any(tech in category_techs for tech in tech_lower):
                return category

        # Fallback categorization by title analysis
        title_lower = ranked_title.title.lower()
        if any(word in title_lower for word in ['api', 'backend', 'server', 'database']):
            return "backend"
        elif any(word in title_lower for word in ['ui', 'frontend', 'react', 'vue', 'angular']):
            return "frontend"
        elif any(word in title_lower for word in ['cloud', 'aws', 'azure', 'deployment']):
            return "cloud"
        else:
            return "development"

    def _calculate_trending_score(self, ranked_title: RankedBlogTitle,
                                related_insights: list[SemanticInsight]) -> float:
        """Calculate how trending/timely the topic is"""

        # Use recency score as base
        base_score = ranked_title.scores.recency_score

        title_lower = ranked_title.title.lower()

        # Hot technology trends (2024-2025)
        trending_topics = [
            ('ai', 0.3), ('artificial intelligence', 0.3), ('machine learning', 0.25),
            ('llm', 0.3), ('gpt', 0.25), ('claude', 0.2),
            ('react 19', 0.2), ('next.js 15', 0.2), ('typescript 5', 0.15),
            ('rust', 0.15), ('go', 0.1), ('webassembly', 0.2),
            ('edge computing', 0.2), ('serverless', 0.15), ('microservices', 0.1)
        ]

        for topic, boost in trending_topics:
            if topic in title_lower:
                base_score += boost
                break

        return min(base_score, 1.0)

    def _calculate_evergreen_potential(self, ranked_title: RankedBlogTitle) -> float:
        """Calculate how long-lasting the content value will be"""

        title_lower = ranked_title.title.lower()

        # High evergreen potential indicators
        evergreen_indicators = [
            ('fundamentals', 0.3), ('principles', 0.25), ('concepts', 0.2),
            ('patterns', 0.2), ('best practices', 0.25), ('architecture', 0.2),
            ('design', 0.15), ('algorithm', 0.25)
        ]

        # Low evergreen potential indicators (version-specific, trending)
        temporal_indicators = [
            ('2024', -0.2), ('2025', -0.2), ('latest', -0.15), ('new', -0.1),
            ('v1.', -0.15), ('beta', -0.2), ('preview', -0.2)
        ]

        score = 0.5  # Base evergreen score

        for indicator, boost in evergreen_indicators:
            if indicator in title_lower:
                score += boost
                break

        for indicator, penalty in temporal_indicators:
            if indicator in title_lower:
                score += penalty  # penalty is negative
                break

        # Pattern-based adjustments
        if ranked_title.original_candidate.pattern_type == "how-to":
            score += 0.15  # How-to guides have lasting value
        elif ranked_title.original_candidate.pattern_type == "performance":
            score -= 0.1  # Performance articles may become dated

        return max(min(score, 1.0), 0.1)

    def _assess_content_saturation(self, ranked_title: RankedBlogTitle) -> str:
        """Assess how saturated the content space is for this topic"""

        title_lower = ranked_title.title.lower()

        # High saturation topics
        high_saturation = [
            'introduction to', 'getting started', 'tutorial', 'basics',
            'best practices', 'tips and tricks'
        ]

        # Low saturation topics
        low_saturation = [
            'advanced', 'optimization', 'performance', 'architecture',
            'case study', 'lessons learned', 'deep dive'
        ]

        if any(phrase in title_lower for phrase in high_saturation):
            return "high"
        elif any(phrase in title_lower for phrase in low_saturation):
            return "low"
        else:
            return "medium"

    def _extract_supporting_details(self, ranked_title: RankedBlogTitle,
                                  related_insights: list[SemanticInsight]) -> SupportingDetails:
        """Extract and compile supporting technical details"""

        details = SupportingDetails()

        # Aggregate performance metrics from insights
        for insight in related_insights:
            details.performance_metrics.extend(insight.performance_metrics)

            # Extract implementation challenges from technical concepts
            for concept in insight.technical_concepts:
                if concept.problem_solved:
                    details.implementation_challenges.append(concept.problem_solved)
                if concept.business_impact:
                    if not details.business_impact:
                        details.business_impact = concept.business_impact

        # Deduplicate and limit
        details.performance_metrics = list(set(details.performance_metrics))[:5]
        details.implementation_challenges = list(set(details.implementation_challenges))[:5]

        # Generate prerequisites based on technologies
        technologies = ranked_title.original_candidate.key_technologies
        details.prerequisites = self._generate_prerequisites(technologies)

        # Generate related concepts
        details.related_concepts = self._generate_related_concepts(ranked_title, technologies)

        # Generate code examples suggestions (placeholder)
        details.code_examples = self._suggest_code_examples(ranked_title, technologies)

        return details

    def _generate_prerequisites(self, technologies: list[str]) -> list[str]:
        """Generate prerequisite knowledge/tools based on technologies"""

        prerequisites = set()

        for tech in technologies:
            tech_lower = tech.lower()

            if tech_lower in ['react', 'vue', 'angular']:
                prerequisites.update(['JavaScript ES6+', 'HTML/CSS', 'Node.js'])
            elif tech_lower in ['python']:
                prerequisites.update(['Python 3.8+', 'pip/virtualenv'])
            elif tech_lower in ['docker', 'kubernetes']:
                prerequisites.update(['Docker basics', 'Container concepts'])
            elif tech_lower in ['aws', 'azure', 'gcp']:
                prerequisites.update(['Cloud computing basics', 'CLI tools'])

        return list(prerequisites)[:5]

    def _generate_related_concepts(self, ranked_title: RankedBlogTitle,
                                  technologies: list[str]) -> list[str]:
        """Generate related concepts for deeper exploration"""

        related = set()

        # Pattern-based relationships
        pattern = ranked_title.original_candidate.pattern_type
        if pattern == "performance":
            related.update(['benchmarking', 'profiling', 'caching', 'optimization'])
        elif pattern == "implementation":
            related.update(['architecture', 'design patterns', 'testing', 'deployment'])
        elif pattern == "comparison":
            related.update(['trade-offs', 'use cases', 'migration', 'evaluation criteria'])

        # Technology-based relationships
        for tech in technologies:
            tech_lower = tech.lower()
            for category, tech_data in self.tech_taxonomy.items():
                if tech_lower in str(tech_data).lower():
                    related.update(tech_data.get('related_concepts', []))

        return list(related)[:6]

    def _suggest_code_examples(self, ranked_title: RankedBlogTitle,
                              technologies: list[str]) -> list[str]:
        """Suggest types of code examples to include"""

        examples = []

        pattern = ranked_title.original_candidate.pattern_type
        if pattern == "tutorial" or pattern == "how-to":
            examples = ["Setup/configuration", "Basic implementation", "Advanced usage", "Error handling"]
        elif pattern == "performance":
            examples = ["Before/after code", "Performance benchmarks", "Optimization techniques"]
        elif pattern == "comparison":
            examples = ["Implementation A", "Implementation B", "Feature comparison"]
        else:
            examples = ["Core implementation", "Configuration", "Usage examples"]

        return examples[:4]

    def _generate_content_angles(self, ranked_title: RankedBlogTitle,
                               related_insights: list[SemanticInsight]) -> list[ContentAngle]:
        """Generate suggested content angles for the blog title"""

        angles = []

        # Determine primary angle based on title pattern
        primary_angle = self._determine_primary_angle(ranked_title)

        # Create primary content angle
        primary = self._create_content_angle(primary_angle, ranked_title, related_insights)
        angles.append(primary)

        # Generate alternative angles
        alternative_angles = self._generate_alternative_angles(ranked_title, primary_angle)
        for angle_type in alternative_angles[:2]:  # Max 2 alternatives
            alt_angle = self._create_content_angle(angle_type, ranked_title, related_insights)
            angles.append(alt_angle)

        return angles

    def _determine_primary_angle(self, ranked_title: RankedBlogTitle) -> str:
        """Determine the primary content angle based on title characteristics"""

        pattern = ranked_title.original_candidate.pattern_type
        title_lower = ranked_title.title.lower()

        # Map patterns to content angles
        pattern_mapping = {
            "how-to": "tutorial",
            "performance": "case_study",
            "comparison": "comparison",
            "implementation": "tutorial",
            "problem-solution": "case_study"
        }

        # Check for explicit angle indicators in title
        if any(word in title_lower for word in ['guide', 'tutorial', 'how to']):
            return "tutorial"
        elif any(word in title_lower for word in ['vs', 'versus', 'compared']):
            return "comparison"
        elif any(word in title_lower for word in ['case study', 'lessons', 'experience']):
            return "case_study"
        elif any(word in title_lower for word in ['deep dive', 'internals', 'architecture']):
            return "deep_dive"
        else:
            return pattern_mapping.get(pattern, "overview")

    def _create_content_angle(self, angle_type: str, ranked_title: RankedBlogTitle,
                            related_insights: list[SemanticInsight]) -> ContentAngle:
        """Create a detailed content angle specification"""

        framework = self.content_frameworks.get(angle_type, self.content_frameworks["overview"])

        # Calculate estimated word count based on complexity
        complexity_score = ranked_title.scores.authority_score + ranked_title.scores.specificity_score
        word_count_min, word_count_max = framework["word_count_range"]

        if complexity_score > 0.8:
            estimated_words = int(word_count_max * 0.9)
        elif complexity_score < 0.4:
            estimated_words = int(word_count_min * 1.1)
        else:
            estimated_words = int((word_count_min + word_count_max) / 2)

        # Generate angle-specific description
        description = self._generate_angle_description(angle_type, ranked_title)

        return ContentAngle(
            angle_type=angle_type,
            description=description,
            target_sections=framework["sections"].copy(),
            estimated_word_count=estimated_words,
            difficulty_level=ranked_title.original_candidate.technical_depth,
            time_to_read=f"{max(1, estimated_words // 200)} min"
        )

    def _generate_angle_description(self, angle_type: str, ranked_title: RankedBlogTitle) -> str:
        """Generate a description for the content angle"""

        title = ranked_title.title

        descriptions = {
            "tutorial": f"Step-by-step guide showing how to implement the techniques described in '{title}'. Includes practical examples, code snippets, and common pitfalls to avoid.",

            "case_study": f"In-depth analysis of the real-world implementation behind '{title}'. Examines the problem context, solution approach, results achieved, and lessons learned.",

            "comparison": f"Comprehensive comparison exploring the technologies and approaches mentioned in '{title}'. Evaluates pros, cons, and best-fit scenarios for each option.",

            "deep_dive": f"Technical deep-dive into the advanced concepts and implementation details referenced in '{title}'. Covers architecture, optimization techniques, and expert-level insights.",

            "overview": f"Comprehensive overview of the topic covered in '{title}'. Provides foundational knowledge, key concepts, and practical applications for getting started."
        }

        return descriptions.get(angle_type, f"Detailed exploration of the concepts presented in '{title}'.")

    def _generate_alternative_angles(self, ranked_title: RankedBlogTitle, primary_angle: str) -> list[str]:
        """Generate alternative content angles"""

        all_angles = ["tutorial", "case_study", "comparison", "deep_dive", "overview"]
        alternatives = [angle for angle in all_angles if angle != primary_angle]

        # Prioritize based on title characteristics
        title_lower = ranked_title.title.lower()

        if 'performance' in title_lower or 'optimization' in title_lower:
            # Performance topics work well as case studies or deep dives
            return [angle for angle in ["case_study", "deep_dive"] if angle in alternatives][:2]
        elif 'vs' in title_lower or 'comparison' in title_lower:
            # Comparison topics can also be tutorials or overviews
            return [angle for angle in ["tutorial", "overview"] if angle in alternatives][:2]
        else:
            # Return first two alternatives
            return alternatives[:2]

    def _extract_seo_keywords(self, ranked_title: RankedBlogTitle) -> list[str]:
        """Extract SEO keyword suggestions from the title and context"""

        title_words = ranked_title.title.lower().split()

        # Filter meaningful words (remove stop words)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'why', 'what'}
        meaningful_words = [word.strip('.,!?:;') for word in title_words if word not in stop_words and len(word) > 2]

        # Add technology keywords
        tech_keywords = [tech.lower() for tech in ranked_title.original_candidate.key_technologies]

        # Combine and deduplicate
        all_keywords = list(set(meaningful_words + tech_keywords))

        # Add long-tail keyword variations
        if len(all_keywords) >= 2:
            # Create 2-word combinations
            for i in range(len(all_keywords) - 1):
                for j in range(i + 1, min(i + 3, len(all_keywords))):
                    combo = f"{all_keywords[i]} {all_keywords[j]}"
                    all_keywords.append(combo)

        return all_keywords[:10]  # Return top 10 keywords

    def _assess_writing_complexity(self, ranked_title: RankedBlogTitle) -> str:
        """Assess the writing complexity required for this content"""

        complexity_indicators = {
            "high": ["architecture", "optimization", "performance", "scalability", "distributed", "microservices"],
            "medium": ["implementation", "framework", "library", "api", "development"],
            "low": ["introduction", "getting started", "basics", "tutorial", "guide"]
        }

        title_lower = ranked_title.title.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in title_lower for indicator in indicators):
                return level

        # Fallback based on technical depth
        if ranked_title.original_candidate.technical_depth == "advanced":
            return "high"
        elif ranked_title.original_candidate.technical_depth == "beginner":
            return "low"
        else:
            return "medium"

    def _assess_research_depth(self, ranked_title: RankedBlogTitle,
                              related_insights: list[SemanticInsight]) -> str:
        """Assess the research depth required for this content"""

        # Base assessment on available insights and complexity
        if not related_insights:
            return "extensive"  # No insights mean more research needed

        insight_quality = sum(insight.confidence_score for insight in related_insights) / len(related_insights)

        if insight_quality > 0.8 and len(related_insights) >= 2:
            return "light"
        elif insight_quality > 0.6:
            return "moderate"
        else:
            return "extensive"

    def _suggest_multimedia_elements(self, ranked_title: RankedBlogTitle) -> list[str]:
        """Suggest multimedia elements for the content"""

        suggestions = ["code_blocks"]  # Always include code blocks for technical content

        title_lower = ranked_title.title.lower()

        # Pattern-based suggestions
        if "architecture" in title_lower or "system" in title_lower:
            suggestions.append("architecture_diagrams")

        if "performance" in title_lower or "optimization" in title_lower:
            suggestions.extend(["performance_charts", "benchmarks"])

        if "comparison" in title_lower or "vs" in title_lower:
            suggestions.append("comparison_tables")

        if "tutorial" in title_lower or "how to" in title_lower:
            suggestions.extend(["screenshots", "step_by_step_diagrams"])

        if "workflow" in title_lower or "process" in title_lower:
            suggestions.append("flowcharts")

        return list(set(suggestions))  # Remove duplicates

    def _analyze_content_clusters(self, enriched_titles: list[EnrichedBlogTitle]):
        """Analyze content clusters across multiple titles"""

        # Group by content category
        category_groups = {}
        for title in enriched_titles:
            category = title.context.content_category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(title)

        # Assign cluster information
        for title in enriched_titles:
            category = title.context.content_category
            related_titles = [t.title for t in category_groups[category] if t.title != title.title]
            title.content_clusters = related_titles[:5]  # Top 5 related titles

    def _identify_series_opportunities(self, enriched_titles: list[EnrichedBlogTitle]):
        """Identify opportunities for content series across titles"""

        # Group by technology and find series opportunities
        tech_groups = {}
        for title in enriched_titles:
            for tech in title.context.key_technologies:
                if tech not in tech_groups:
                    tech_groups[tech] = []
                tech_groups[tech].append(title)

        # Identify series potential
        for title in enriched_titles:
            series_suggestions = []

            for tech in title.context.key_technologies:
                related_titles = tech_groups.get(tech, [])
                if len(related_titles) >= 3:  # Enough for a series
                    # Suggest series topics
                    series_suggestions.extend([
                        f"{tech} Fundamentals",
                        f"Advanced {tech} Techniques",
                        f"{tech} in Production",
                        f"{tech} Performance Optimization"
                    ])

            title.series_potential = list(set(series_suggestions))[:4]
