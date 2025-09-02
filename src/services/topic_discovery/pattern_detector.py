# src/services/topic_discovery/pattern_detector.py
"""
Pattern Detector for Blog Title Discovery

This module identifies emerging patterns, themes, and relationships across multiple
blog titles to detect trending topics, content series opportunities, and theme connections.
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import math

from .context_enricher import EnrichedBlogTitle
from .ai_semantic_analyzer import SemanticInsight


@dataclass
class EmergingTheme:
    """Represents an emerging theme identified across multiple blog titles"""
    theme_name: str
    description: str
    frequency: int  # Number of titles related to this theme
    trend_strength: float  # 0-1 score indicating how strong the trend is
    related_titles: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    key_technologies: List[str] = field(default_factory=list)
    growth_trajectory: str = "stable"  # "growing", "stable", "declining"
    theme_maturity: str = "emerging"  # "emerging", "mainstream", "declining"


@dataclass
class ContentSeries:
    """Represents a potential content series based on related titles"""
    series_name: str
    description: str
    related_titles: List[str] = field(default_factory=list)
    suggested_additional_titles: List[str] = field(default_factory=list)
    series_type: str = "technology_deep_dive"  # "tutorial_series", "case_study_series", etc.
    difficulty_progression: List[str] = field(default_factory=list)  # "beginner" -> "advanced"
    estimated_articles: int = 5


@dataclass 
class TrendConnection:
    """Represents a connection between different trends or themes"""
    connection_type: str  # "complementary", "competitive", "sequential", "causal"
    theme_a: str
    theme_b: str
    connection_strength: float  # 0-1 strength of the connection
    description: str
    shared_concepts: List[str] = field(default_factory=list)
    shared_technologies: List[str] = field(default_factory=list)


@dataclass
class PatternAnalysis:
    """Comprehensive pattern analysis results"""
    emerging_themes: List[EmergingTheme] = field(default_factory=list)
    content_series: List[ContentSeries] = field(default_factory=list)  
    trend_connections: List[TrendConnection] = field(default_factory=list)
    
    # Meta analysis
    dominant_categories: List[str] = field(default_factory=list)
    trending_technologies: List[str] = field(default_factory=list)
    content_gaps: List[str] = field(default_factory=list)
    seasonal_patterns: Dict[str, List[str]] = field(default_factory=dict)
    
    # Analysis metadata
    analysis_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    titles_analyzed: int = 0
    confidence_score: float = 0.0


class PatternDetector:
    """
    Advanced pattern detection system that identifies emerging themes, content series
    opportunities, and trend connections across multiple blog titles.
    """
    
    def __init__(self, 
                 min_theme_frequency: int = 2,
                 min_series_size: int = 3,
                 connection_threshold: float = 0.3):
        """
        Initialize Pattern Detector
        
        Args:
            min_theme_frequency: Minimum frequency to consider something a theme
            min_series_size: Minimum number of titles to form a series
            connection_threshold: Minimum strength to identify trend connections
        """
        self.logger = logging.getLogger(__name__)
        
        # Detection thresholds
        self.min_theme_frequency = min_theme_frequency
        self.min_series_size = min_series_size
        self.connection_threshold = connection_threshold
        
        # Initialize pattern recognition data
        self._initialize_technology_relationships()
        self._initialize_content_patterns()
        self._initialize_theme_taxonomies()
        
        self.logger.info("Pattern Detector initialized")
    
    def _initialize_technology_relationships(self):
        """Initialize known technology relationships and ecosystems"""
        
        self.tech_ecosystems = {
            "javascript_frontend": {
                "core": ["JavaScript", "TypeScript"],
                "frameworks": ["React", "Vue", "Angular", "Svelte"],
                "tools": ["Webpack", "Vite", "Next.js", "Nuxt.js"],
                "state": ["Redux", "Zustand", "Pinia", "MobX"]
            },
            "python_backend": {
                "core": ["Python"],
                "frameworks": ["Django", "FastAPI", "Flask", "Pyramid"],
                "tools": ["pip", "poetry", "pytest", "black"],
                "data": ["pandas", "numpy", "scikit-learn", "tensorflow"]
            },
            "cloud_native": {
                "platforms": ["AWS", "Azure", "GCP"],
                "containers": ["Docker", "Kubernetes", "Helm"],
                "serverless": ["Lambda", "Azure Functions", "Cloud Functions"],
                "monitoring": ["Prometheus", "Grafana", "Datadog"]
            },
            "ai_ml": {
                "frameworks": ["TensorFlow", "PyTorch", "scikit-learn"],
                "platforms": ["OpenAI", "Anthropic", "Hugging Face"],
                "tools": ["Jupyter", "MLflow", "Weights & Biases"],
                "applications": ["ChatGPT", "Claude", "Midjourney"]
            }
        }
        
        # Technology similarity mapping for clustering
        self.tech_similarities = {
            "React": ["Next.js", "Gatsby", "TypeScript", "JavaScript"],
            "Vue": ["Nuxt.js", "Vuex", "Pinia", "JavaScript"],
            "Python": ["Django", "FastAPI", "Flask", "pandas", "numpy"],
            "AWS": ["Lambda", "S3", "EC2", "CloudFormation"],
            "Docker": ["Kubernetes", "containerization", "microservices"],
            "AI": ["machine learning", "OpenAI", "LLM", "GPT", "Claude"]
        }
    
    def _initialize_content_patterns(self):
        """Initialize content pattern recognition rules"""
        
        self.content_pattern_indicators = {
            "performance_optimization": {
                "keywords": ["performance", "optimization", "faster", "improved", "reduced", "latency"],
                "patterns": [r"\d+%\s+(?:faster|improvement|reduction)", r"reduced.*by.*\d+"]
            },
            "migration_modernization": {
                "keywords": ["migration", "upgrade", "modernization", "from.*to", "replacing"],
                "patterns": [r"from\s+\w+\s+to\s+\w+", r"migrating.*to", r"upgrading.*to"]
            },
            "architecture_patterns": {
                "keywords": ["architecture", "design", "patterns", "microservices", "monolith"],
                "patterns": [r"architecture.*pattern", r"design.*system", r"microservices"]
            },
            "developer_productivity": {
                "keywords": ["productivity", "workflow", "automation", "tooling", "developer experience"],
                "patterns": [r"developer.*experience", r"workflow.*optimization", r"automation"]
            },
            "security_compliance": {
                "keywords": ["security", "authentication", "authorization", "compliance", "privacy"],
                "patterns": [r"security.*implementation", r"auth.*system", r"compliance"]
            }
        }
    
    def _initialize_theme_taxonomies(self):
        """Initialize theme classification taxonomies"""
        
        self.theme_hierarchies = {
            "Technical Implementation": {
                "subcategories": ["Backend Development", "Frontend Development", "Full-Stack", "Mobile"],
                "indicators": ["implementation", "development", "coding", "programming"]
            },
            "System Architecture": {
                "subcategories": ["Microservices", "Distributed Systems", "Cloud Architecture", "API Design"],
                "indicators": ["architecture", "system design", "scalability", "distributed"]
            },
            "DevOps & Infrastructure": {
                "subcategories": ["CI/CD", "Containerization", "Monitoring", "Deployment"],
                "indicators": ["devops", "infrastructure", "deployment", "ci/cd", "monitoring"]
            },
            "Data & AI": {
                "subcategories": ["Machine Learning", "Data Engineering", "Analytics", "AI Applications"],
                "indicators": ["ai", "machine learning", "data", "analytics", "ml"]
            },
            "Performance & Optimization": {
                "subcategories": ["Web Performance", "Database Optimization", "Caching", "Scaling"],
                "indicators": ["performance", "optimization", "caching", "scaling"]
            }
        }
    
    async def detect_emerging_themes(self, enriched_titles: List[EnrichedBlogTitle]) -> List[EmergingTheme]:
        """
        Detect emerging themes across multiple blog titles
        
        Args:
            enriched_titles: List of enriched blog titles to analyze
            
        Returns:
            List of detected emerging themes
        """
        if len(enriched_titles) < self.min_theme_frequency:
            return []
        
        self.logger.info(f"Detecting emerging themes from {len(enriched_titles)} titles")
        
        # Extract and cluster concepts
        concept_clusters = self._extract_concept_clusters(enriched_titles)
        
        # Identify technology themes
        tech_themes = self._identify_technology_themes(enriched_titles)
        
        # Identify content pattern themes
        pattern_themes = self._identify_pattern_themes(enriched_titles)
        
        # Merge and analyze all themes
        all_themes = tech_themes + pattern_themes
        merged_themes = self._merge_similar_themes(all_themes)
        
        # Calculate trend strength and characteristics
        for theme in merged_themes:
            theme.trend_strength = self._calculate_trend_strength(theme, enriched_titles)
            theme.growth_trajectory = self._assess_growth_trajectory(theme, enriched_titles)
            theme.theme_maturity = self._assess_theme_maturity(theme)
        
        # Filter and rank themes
        qualified_themes = [theme for theme in merged_themes if theme.frequency >= self.min_theme_frequency]
        ranked_themes = sorted(qualified_themes, key=lambda x: x.trend_strength, reverse=True)
        
        self.logger.info(f"Detected {len(ranked_themes)} emerging themes")
        return ranked_themes[:10]  # Return top 10 themes
    
    def _extract_concept_clusters(self, enriched_titles: List[EnrichedBlogTitle]) -> Dict[str, List[str]]:
        """Extract and cluster related concepts from titles"""
        
        # Collect all concepts from titles
        all_concepts = []
        for title in enriched_titles:
            # Extract from title text
            concepts = self._extract_concepts_from_text(title.title)
            all_concepts.extend(concepts)
            
            # Extract from technologies
            all_concepts.extend(title.context.key_technologies)
            
            # Extract from content clusters
            all_concepts.extend(title.content_clusters)
        
        # Count concept frequencies
        concept_counts = Counter(all_concepts)
        
        # Group similar concepts
        concept_clusters = defaultdict(list)
        processed = set()
        
        for concept, count in concept_counts.most_common():
            if concept in processed:
                continue
                
            # Find similar concepts
            similar_concepts = [concept]
            for other_concept, _ in concept_counts.items():
                if other_concept != concept and other_concept not in processed:
                    if self._are_concepts_similar(concept, other_concept):
                        similar_concepts.append(other_concept)
                        processed.add(other_concept)
            
            if len(similar_concepts) >= self.min_theme_frequency:
                cluster_name = self._generate_cluster_name(similar_concepts)
                concept_clusters[cluster_name] = similar_concepts
                processed.update(similar_concepts)
        
        return dict(concept_clusters)
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract meaningful concepts from text using NLP patterns"""
        
        text_lower = text.lower()
        concepts = []
        
        # Extract technology names
        tech_pattern = re.compile(r'\b(react|vue|angular|python|javascript|typescript|node\.js|docker|kubernetes|aws|azure|gcp|api|graphql|rest|microservices|serverless)\b', re.IGNORECASE)
        concepts.extend(tech_pattern.findall(text))
        
        # Extract action/concept patterns
        action_patterns = [
            r'\b(optimization|performance|scalability|architecture|implementation|deployment|monitoring|testing|debugging|security)\b',
            r'\b(migration|upgrade|refactoring|modernization|automation|integration)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Extract numeric concepts (versions, improvements)
        numeric_concepts = re.findall(r'\b(\d+(?:\.\d+)?%|\d+x\s+faster|v?\d+\.\d+)\b', text, re.IGNORECASE)
        concepts.extend(numeric_concepts)
        
        return [concept.lower() for concept in concepts if len(concept) > 2]
    
    def _are_concepts_similar(self, concept1: str, concept2: str) -> bool:
        """Determine if two concepts are similar enough to cluster"""
        
        # Direct similarity check
        if concept1.lower() in concept2.lower() or concept2.lower() in concept1.lower():
            return True
        
        # Technology ecosystem check
        for ecosystem in self.tech_ecosystems.values():
            for tech_list in ecosystem.values():
                tech_list_lower = [t.lower() for t in tech_list]
                if concept1.lower() in tech_list_lower and concept2.lower() in tech_list_lower:
                    return True
        
        # Semantic similarity patterns
        similarity_groups = [
            ["performance", "optimization", "speed", "faster", "efficiency"],
            ["architecture", "design", "structure", "pattern"],
            ["deployment", "release", "publishing", "delivery"],
            ["testing", "quality", "validation", "verification"],
            ["security", "authentication", "authorization", "privacy"]
        ]
        
        for group in similarity_groups:
            if concept1.lower() in group and concept2.lower() in group:
                return True
        
        return False
    
    def _generate_cluster_name(self, concepts: List[str]) -> str:
        """Generate a representative name for a concept cluster"""
        
        # Find the most common/representative concept
        concept_scores = {}
        
        for concept in concepts:
            score = 0
            
            # Prefer shorter, more general terms
            score += (20 - len(concept)) * 0.1
            
            # Prefer technology names
            if any(concept.lower() in eco_data for eco_data in self.tech_ecosystems.values() 
                  for tech_list in eco_data.values() 
                  for tech in tech_list):
                score += 2
            
            # Prefer common technical terms
            common_terms = ["performance", "architecture", "deployment", "security", "optimization"]
            if concept.lower() in common_terms:
                score += 1.5
            
            concept_scores[concept] = score
        
        # Return highest scoring concept
        best_concept = max(concept_scores.items(), key=lambda x: x[1])[0]
        return best_concept.title()
    
    def _identify_technology_themes(self, enriched_titles: List[EnrichedBlogTitle]) -> List[EmergingTheme]:
        """Identify themes based on technology clustering"""
        
        tech_themes = []
        
        # Group titles by technology ecosystems
        ecosystem_groups = defaultdict(list)
        
        for title in enriched_titles:
            title_techs = [tech.lower() for tech in title.context.key_technologies]
            
            # Check which ecosystems this title belongs to
            for ecosystem_name, ecosystem_data in self.tech_ecosystems.items():
                ecosystem_techs = []
                for tech_list in ecosystem_data.values():
                    ecosystem_techs.extend([t.lower() for t in tech_list])
                
                # If title has technologies from this ecosystem
                if any(tech in ecosystem_techs for tech in title_techs):
                    ecosystem_groups[ecosystem_name].append(title)
        
        # Create themes for significant ecosystem groups
        for ecosystem_name, titles in ecosystem_groups.items():
            if len(titles) >= self.min_theme_frequency:
                
                # Extract common technologies
                all_techs = []
                for title in titles:
                    all_techs.extend(title.context.key_technologies)
                common_techs = [tech for tech, count in Counter(all_techs).most_common(5)]
                
                # Create theme
                theme = EmergingTheme(
                    theme_name=self._humanize_ecosystem_name(ecosystem_name),
                    description=f"Emerging trends and developments in the {ecosystem_name.replace('_', ' ')} ecosystem",
                    frequency=len(titles),
                    trend_strength=0.0,  # Will be calculated later
                    related_titles=[title.title for title in titles],
                    related_concepts=[ecosystem_name.replace('_', ' ')],
                    key_technologies=common_techs
                )
                
                tech_themes.append(theme)
        
        return tech_themes
    
    def _identify_pattern_themes(self, enriched_titles: List[EnrichedBlogTitle]) -> List[EmergingTheme]:
        """Identify themes based on content patterns"""
        
        pattern_themes = []
        
        # Group titles by content patterns
        pattern_groups = defaultdict(list)
        
        for title in enriched_titles:
            title_text = title.title.lower()
            
            # Check against each content pattern
            for pattern_name, pattern_data in self.content_pattern_indicators.items():
                matches = 0
                
                # Check keywords
                for keyword in pattern_data["keywords"]:
                    if keyword in title_text:
                        matches += 1
                
                # Check regex patterns
                for pattern in pattern_data["patterns"]:
                    if re.search(pattern, title_text, re.IGNORECASE):
                        matches += 2  # Regex matches are stronger indicators
                
                if matches > 0:
                    pattern_groups[pattern_name].append((title, matches))
        
        # Create themes for significant pattern groups
        for pattern_name, title_matches in pattern_groups.items():
            if len(title_matches) >= self.min_theme_frequency:
                
                titles = [tm[0] for tm in title_matches]
                
                # Extract common concepts
                all_concepts = []
                for title in titles:
                    all_concepts.extend(self._extract_concepts_from_text(title.title))
                common_concepts = [concept for concept, count in Counter(all_concepts).most_common(5)]
                
                # Create theme
                theme = EmergingTheme(
                    theme_name=self._humanize_pattern_name(pattern_name),
                    description=f"Growing focus on {pattern_name.replace('_', ' ')} across various technologies",
                    frequency=len(titles),
                    trend_strength=0.0,  # Will be calculated later
                    related_titles=[title.title for title in titles],
                    related_concepts=common_concepts,
                    key_technologies=[]
                )
                
                pattern_themes.append(theme)
        
        return pattern_themes
    
    def _merge_similar_themes(self, themes: List[EmergingTheme]) -> List[EmergingTheme]:
        """Merge themes that are too similar"""
        
        merged_themes = []
        processed_indices = set()
        
        for i, theme in enumerate(themes):
            if i in processed_indices:
                continue
            
            # Find similar themes
            similar_themes = [theme]
            for j, other_theme in enumerate(themes[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                if self._are_themes_similar(theme, other_theme):
                    similar_themes.append(other_theme)
                    processed_indices.add(j)
            
            # Merge if we found similar themes
            if len(similar_themes) > 1:
                merged_theme = self._merge_themes(similar_themes)
                merged_themes.append(merged_theme)
            else:
                merged_themes.append(theme)
            
            processed_indices.add(i)
        
        return merged_themes
    
    def _are_themes_similar(self, theme1: EmergingTheme, theme2: EmergingTheme) -> bool:
        """Check if two themes are similar enough to merge"""
        
        # Check technology overlap
        tech_overlap = len(set(theme1.key_technologies).intersection(set(theme2.key_technologies)))
        tech_similarity = tech_overlap / max(len(theme1.key_technologies), len(theme2.key_technologies), 1)
        
        # Check concept overlap
        concept_overlap = len(set(theme1.related_concepts).intersection(set(theme2.related_concepts)))
        concept_similarity = concept_overlap / max(len(theme1.related_concepts), len(theme2.related_concepts), 1)
        
        # Check name similarity
        name_similarity = self._calculate_string_similarity(theme1.theme_name, theme2.theme_name)
        
        # Themes are similar if they have high overlap in any dimension
        return (tech_similarity > 0.6 or concept_similarity > 0.5 or name_similarity > 0.7)
    
    def _merge_themes(self, themes: List[EmergingTheme]) -> EmergingTheme:
        """Merge multiple similar themes into one"""
        
        # Use the theme with the highest frequency as the base
        base_theme = max(themes, key=lambda x: x.frequency)
        
        # Merge data from all themes
        merged_titles = []
        merged_concepts = []
        merged_technologies = []
        total_frequency = 0
        
        for theme in themes:
            merged_titles.extend(theme.related_titles)
            merged_concepts.extend(theme.related_concepts)
            merged_technologies.extend(theme.key_technologies)
            total_frequency += theme.frequency
        
        # Remove duplicates and limit
        merged_titles = list(set(merged_titles))
        merged_concepts = list(set(merged_concepts))[:10]
        merged_technologies = list(set(merged_technologies))[:8]
        
        return EmergingTheme(
            theme_name=base_theme.theme_name,
            description=base_theme.description,
            frequency=total_frequency,
            trend_strength=0.0,  # Will be recalculated
            related_titles=merged_titles,
            related_concepts=merged_concepts,
            key_technologies=merged_technologies
        )
    
    def _calculate_trend_strength(self, theme: EmergingTheme, enriched_titles: List[EnrichedBlogTitle]) -> float:
        """Calculate the strength of an emerging theme"""
        
        # Base score from frequency
        frequency_score = min(theme.frequency / len(enriched_titles), 0.4)
        
        # Technology relevance score
        tech_relevance = 0.0
        trending_technologies = ["ai", "react", "typescript", "kubernetes", "serverless", "python"]
        
        for tech in theme.key_technologies:
            if tech.lower() in trending_technologies:
                tech_relevance += 0.1
        
        tech_relevance = min(tech_relevance, 0.3)
        
        # Recency score (titles with higher recency scores boost theme strength)
        recency_score = 0.0
        title_lookup = {title.title: title for title in enriched_titles}
        
        for title_text in theme.related_titles:
            if title_text in title_lookup:
                title = title_lookup[title_text]
                recency_score += title.context.trending_score
        
        avg_recency = recency_score / len(theme.related_titles) if theme.related_titles else 0
        recency_contribution = min(avg_recency * 0.2, 0.2)
        
        # Diversity score (themes spanning multiple categories are stronger)
        categories = set()
        for title_text in theme.related_titles:
            if title_text in title_lookup:
                categories.add(title_lookup[title_text].context.content_category)
        
        diversity_score = min(len(categories) * 0.05, 0.1)
        
        total_score = frequency_score + tech_relevance + recency_contribution + diversity_score
        return min(total_score, 1.0)
    
    def _assess_growth_trajectory(self, theme: EmergingTheme, enriched_titles: List[EnrichedBlogTitle]) -> str:
        """Assess the growth trajectory of a theme"""
        
        # For now, use simple heuristics based on technology trends
        growing_indicators = ["ai", "machine learning", "kubernetes", "serverless", "typescript", "react 19"]
        stable_indicators = ["javascript", "python", "api", "database", "web"]
        declining_indicators = ["jquery", "angular.js", "php", "ruby"]
        
        theme_text = f"{theme.theme_name} {' '.join(theme.key_technologies)}".lower()
        
        if any(indicator in theme_text for indicator in growing_indicators):
            return "growing"
        elif any(indicator in theme_text for indicator in declining_indicators):
            return "declining"
        else:
            return "stable"
    
    def _assess_theme_maturity(self, theme: EmergingTheme) -> str:
        """Assess the maturity level of a theme"""
        
        # Use frequency and technology maturity as indicators
        if theme.frequency >= 8:
            return "mainstream"
        elif theme.frequency >= 4:
            return "developing"
        else:
            return "emerging"
    
    async def suggest_content_series(self, enriched_titles: List[EnrichedBlogTitle]) -> List[ContentSeries]:
        """
        Suggest potential content series based on related titles
        
        Args:
            enriched_titles: List of enriched blog titles to analyze
            
        Returns:
            List of suggested content series
        """
        if len(enriched_titles) < self.min_series_size:
            return []
        
        self.logger.info(f"Analyzing {len(enriched_titles)} titles for content series opportunities")
        
        # Group titles by various criteria
        tech_groups = self._group_by_technology(enriched_titles)
        pattern_groups = self._group_by_content_pattern(enriched_titles)
        category_groups = self._group_by_category(enriched_titles)
        
        series_candidates = []
        
        # Generate series from technology groups
        for tech, titles in tech_groups.items():
            if len(titles) >= self.min_series_size:
                series = self._create_technology_series(tech, titles)
                series_candidates.append(series)
        
        # Generate series from pattern groups
        for pattern, titles in pattern_groups.items():
            if len(titles) >= self.min_series_size:
                series = self._create_pattern_series(pattern, titles)
                series_candidates.append(series)
        
        # Generate series from category groups
        for category, titles in category_groups.items():
            if len(titles) >= self.min_series_size:
                series = self._create_category_series(category, titles)
                series_candidates.append(series)
        
        # Rank and filter series
        ranked_series = self._rank_content_series(series_candidates)
        
        self.logger.info(f"Identified {len(ranked_series)} potential content series")
        return ranked_series[:8]  # Return top 8 series
    
    def _group_by_technology(self, enriched_titles: List[EnrichedBlogTitle]) -> Dict[str, List[EnrichedBlogTitle]]:
        """Group titles by primary technology"""
        
        tech_groups = defaultdict(list)
        
        for title in enriched_titles:
            for tech in title.context.key_technologies:
                tech_groups[tech].append(title)
        
        # Return only groups with sufficient size
        return {tech: titles for tech, titles in tech_groups.items() if len(titles) >= self.min_series_size}
    
    def _group_by_content_pattern(self, enriched_titles: List[EnrichedBlogTitle]) -> Dict[str, List[EnrichedBlogTitle]]:
        """Group titles by content patterns"""
        
        pattern_groups = defaultdict(list)
        
        for title in enriched_titles:
            pattern = title.ranked_title.original_candidate.pattern_type
            pattern_groups[pattern].append(title)
        
        return {pattern: titles for pattern, titles in pattern_groups.items() if len(titles) >= self.min_series_size}
    
    def _group_by_category(self, enriched_titles: List[EnrichedBlogTitle]) -> Dict[str, List[EnrichedBlogTitle]]:
        """Group titles by content category"""
        
        category_groups = defaultdict(list)
        
        for title in enriched_titles:
            category = title.context.content_category
            category_groups[category].append(title)
        
        return {category: titles for category, titles in category_groups.items() if len(titles) >= self.min_series_size}
    
    def _create_technology_series(self, technology: str, titles: List[EnrichedBlogTitle]) -> ContentSeries:
        """Create a content series focused on a specific technology"""
        
        # Determine series progression
        difficulty_levels = []
        for title in titles:
            difficulty_levels.append(title.context.technical_depth)
        
        # Create logical progression
        progression = []
        if "beginner" in difficulty_levels:
            progression.append("beginner")
        if "intermediate" in difficulty_levels:
            progression.append("intermediate")
        if "advanced" in difficulty_levels:
            progression.append("advanced")
        
        # Generate additional title suggestions
        additional_titles = self._generate_technology_series_titles(technology, titles)
        
        return ContentSeries(
            series_name=f"Complete {technology} Guide",
            description=f"Comprehensive series covering {technology} from basics to advanced implementation",
            related_titles=[title.title for title in titles],
            suggested_additional_titles=additional_titles,
            series_type="technology_deep_dive",
            difficulty_progression=progression,
            estimated_articles=len(titles) + len(additional_titles)
        )
    
    def _create_pattern_series(self, pattern: str, titles: List[EnrichedBlogTitle]) -> ContentSeries:
        """Create a content series based on content patterns"""
        
        series_types = {
            "how-to": "tutorial_series",
            "performance": "optimization_series",
            "comparison": "evaluation_series",
            "implementation": "build_series"
        }
        
        additional_titles = self._generate_pattern_series_titles(pattern, titles)
        
        return ContentSeries(
            series_name=f"{pattern.replace('_', ' ').title()} Series",
            description=f"Collection of {pattern.replace('_', ' ')} focused articles",
            related_titles=[title.title for title in titles],
            suggested_additional_titles=additional_titles,
            series_type=series_types.get(pattern, "general_series"),
            difficulty_progression=["beginner", "intermediate", "advanced"],
            estimated_articles=len(titles) + len(additional_titles)
        )
    
    def _create_category_series(self, category: str, titles: List[EnrichedBlogTitle]) -> ContentSeries:
        """Create a content series based on content category"""
        
        additional_titles = self._generate_category_series_titles(category, titles)
        
        return ContentSeries(
            series_name=f"{category.replace('_', ' ').title()} Mastery",
            description=f"In-depth exploration of {category.replace('_', ' ')} concepts and practices",
            related_titles=[title.title for title in titles],
            suggested_additional_titles=additional_titles,
            series_type="category_deep_dive",
            difficulty_progression=["beginner", "intermediate", "advanced"],
            estimated_articles=len(titles) + len(additional_titles)
        )
    
    def _generate_technology_series_titles(self, technology: str, existing_titles: List[EnrichedBlogTitle]) -> List[str]:
        """Generate additional title suggestions for technology series"""
        
        existing_title_texts = [title.title.lower() for title in existing_titles]
        
        # Common technology series progression
        common_titles = [
            f"Getting Started with {technology}",
            f"{technology} Best Practices and Patterns",
            f"Advanced {technology} Techniques",
            f"{technology} Performance Optimization",
            f"Building Production Apps with {technology}",
            f"{technology} Testing Strategies",
            f"Debugging {technology} Applications",
            f"{technology} Security Considerations"
        ]
        
        # Filter out titles that are too similar to existing ones
        suggested = []
        for title in common_titles:
            if not any(self._calculate_string_similarity(title.lower(), existing.lower()) > 0.6 
                      for existing in existing_title_texts):
                suggested.append(title)
        
        return suggested[:4]  # Return top 4 suggestions
    
    def _generate_pattern_series_titles(self, pattern: str, existing_titles: List[EnrichedBlogTitle]) -> List[str]:
        """Generate additional titles for pattern-based series"""
        
        pattern_templates = {
            "how-to": [
                "How to Choose the Right Architecture",
                "How to Scale Your Application",
                "How to Implement Monitoring",
                "How to Handle Errors Gracefully"
            ],
            "performance": [
                "Database Query Optimization",
                "Frontend Bundle Optimization",
                "API Response Time Improvement",
                "Memory Usage Optimization"
            ],
            "comparison": [
                "SQL vs NoSQL for Your Use Case",
                "REST vs GraphQL API Design",
                "Monolith vs Microservices Decision",
                "Cloud Provider Comparison"
            ]
        }
        
        return pattern_templates.get(pattern, ["Additional Topics Coming Soon"])[:3]
    
    def _generate_category_series_titles(self, category: str, existing_titles: List[EnrichedBlogTitle]) -> List[str]:
        """Generate additional titles for category-based series"""
        
        category_templates = {
            "frontend": [
                "Modern CSS Techniques",
                "State Management Patterns",
                "Component Architecture Design",
                "Frontend Testing Strategies"
            ],
            "backend": [
                "API Design Principles", 
                "Database Schema Design",
                "Authentication and Authorization",
                "Background Job Processing"
            ],
            "cloud": [
                "Infrastructure as Code",
                "Container Orchestration",
                "Serverless Architecture Patterns",
                "Multi-Cloud Strategies"
            ]
        }
        
        return category_templates.get(category, ["Category-Specific Topics"])[:3]
    
    def _rank_content_series(self, series_candidates: List[ContentSeries]) -> List[ContentSeries]:
        """Rank content series by potential value and completeness"""
        
        for series in series_candidates:
            score = 0.0
            
            # Size score (larger series are more valuable)
            score += min(len(series.related_titles) / 10.0, 0.3)
            
            # Progression score (complete progression is better)
            if len(series.difficulty_progression) >= 3:
                score += 0.2
            elif len(series.difficulty_progression) >= 2:
                score += 0.1
            
            # Additional content potential
            score += min(len(series.suggested_additional_titles) / 5.0, 0.2)
            
            # Series type value
            valuable_types = ["technology_deep_dive", "tutorial_series", "optimization_series"]
            if series.series_type in valuable_types:
                score += 0.2
            
            series.estimated_articles = int(score * 100)  # Use score as proxy for ranking
        
        return sorted(series_candidates, key=lambda x: x.estimated_articles, reverse=True)
    
    async def identify_trend_connections(self, enriched_titles: List[EnrichedBlogTitle]) -> List[TrendConnection]:
        """
        Identify connections and relationships between different trends
        
        Args:
            enriched_titles: List of enriched blog titles to analyze
            
        Returns:
            List of trend connections
        """
        self.logger.info(f"Identifying trend connections across {len(enriched_titles)} titles")
        
        # Extract themes first
        themes = await self.detect_emerging_themes(enriched_titles)
        
        connections = []
        
        # Analyze pairwise theme connections
        for i, theme_a in enumerate(themes):
            for theme_b in themes[i+1:]:
                connection = self._analyze_theme_connection(theme_a, theme_b)
                if connection and connection.connection_strength >= self.connection_threshold:
                    connections.append(connection)
        
        # Sort by connection strength
        ranked_connections = sorted(connections, key=lambda x: x.connection_strength, reverse=True)
        
        self.logger.info(f"Identified {len(ranked_connections)} trend connections")
        return ranked_connections[:10]  # Return top 10 connections
    
    def _analyze_theme_connection(self, theme_a: EmergingTheme, theme_b: EmergingTheme) -> Optional[TrendConnection]:
        """Analyze the connection between two themes"""
        
        # Calculate technology overlap
        tech_a = set([tech.lower() for tech in theme_a.key_technologies])
        tech_b = set([tech.lower() for tech in theme_b.key_technologies])
        shared_tech = tech_a.intersection(tech_b)
        tech_connection = len(shared_tech) / max(len(tech_a), len(tech_b), 1)
        
        # Calculate concept overlap
        concept_a = set([concept.lower() for concept in theme_a.related_concepts])
        concept_b = set([concept.lower() for concept in theme_b.related_concepts])
        shared_concepts = concept_a.intersection(concept_b)
        concept_connection = len(shared_concepts) / max(len(concept_a), len(concept_b), 1)
        
        # Overall connection strength
        connection_strength = (tech_connection + concept_connection) / 2
        
        if connection_strength < self.connection_threshold:
            return None
        
        # Determine connection type
        connection_type = self._determine_connection_type(theme_a, theme_b, tech_connection, concept_connection)
        
        # Generate description
        description = self._generate_connection_description(theme_a, theme_b, connection_type, shared_tech, shared_concepts)
        
        return TrendConnection(
            connection_type=connection_type,
            theme_a=theme_a.theme_name,
            theme_b=theme_b.theme_name,
            connection_strength=connection_strength,
            description=description,
            shared_concepts=list(shared_concepts),
            shared_technologies=list(shared_tech)
        )
    
    def _determine_connection_type(self, theme_a: EmergingTheme, theme_b: EmergingTheme, 
                                  tech_connection: float, concept_connection: float) -> str:
        """Determine the type of connection between themes"""
        
        # High overlap suggests complementary relationship
        if tech_connection > 0.6 or concept_connection > 0.6:
            return "complementary"
        
        # Check for competitive relationships (similar space, different approaches)
        if self._are_themes_competitive(theme_a, theme_b):
            return "competitive"
        
        # Check for sequential relationships (one builds on the other)
        if self._are_themes_sequential(theme_a, theme_b):
            return "sequential"
        
        # Default to causal relationship
        return "causal"
    
    def _are_themes_competitive(self, theme_a: EmergingTheme, theme_b: EmergingTheme) -> bool:
        """Check if themes represent competitive approaches"""
        
        competitive_pairs = [
            ("react", "vue"), ("react", "angular"),
            ("aws", "azure"), ("aws", "gcp"), ("azure", "gcp"),
            ("sql", "nosql"), ("rest", "graphql"),
            ("monolith", "microservices")
        ]
        
        theme_a_lower = theme_a.theme_name.lower()
        theme_b_lower = theme_b.theme_name.lower()
        
        for pair in competitive_pairs:
            if (pair[0] in theme_a_lower and pair[1] in theme_b_lower) or \
               (pair[1] in theme_a_lower and pair[0] in theme_b_lower):
                return True
        
        return False
    
    def _are_themes_sequential(self, theme_a: EmergingTheme, theme_b: EmergingTheme) -> bool:
        """Check if themes represent sequential development stages"""
        
        sequential_indicators = [
            ("basic", "advanced"), ("introduction", "deep dive"),
            ("getting started", "optimization"), ("development", "deployment")
        ]
        
        theme_a_lower = theme_a.theme_name.lower()
        theme_b_lower = theme_b.theme_name.lower()
        
        for pair in sequential_indicators:
            if (pair[0] in theme_a_lower and pair[1] in theme_b_lower) or \
               (pair[1] in theme_a_lower and pair[0] in theme_b_lower):
                return True
        
        return False
    
    def _generate_connection_description(self, theme_a: EmergingTheme, theme_b: EmergingTheme,
                                        connection_type: str, shared_tech: Set[str], 
                                        shared_concepts: Set[str]) -> str:
        """Generate a description of the connection between themes"""
        
        descriptions = {
            "complementary": f"{theme_a.theme_name} and {theme_b.theme_name} work together synergistically, sharing technologies like {', '.join(list(shared_tech)[:2])}",
            
            "competitive": f"{theme_a.theme_name} and {theme_b.theme_name} represent alternative approaches to solving similar problems",
            
            "sequential": f"{theme_a.theme_name} typically leads to or builds upon {theme_b.theme_name} in the development process",
            
            "causal": f"{theme_a.theme_name} trends may be driving adoption and interest in {theme_b.theme_name}"
        }
        
        return descriptions.get(connection_type, f"{theme_a.theme_name} is related to {theme_b.theme_name}")
    
    def _humanize_ecosystem_name(self, ecosystem_name: str) -> str:
        """Convert ecosystem names to human-readable format"""
        
        humanized = {
            "javascript_frontend": "Modern Frontend Development",
            "python_backend": "Python Backend Engineering",
            "cloud_native": "Cloud-Native Architecture",
            "ai_ml": "AI and Machine Learning"
        }
        
        return humanized.get(ecosystem_name, ecosystem_name.replace("_", " ").title())
    
    def _humanize_pattern_name(self, pattern_name: str) -> str:
        """Convert pattern names to human-readable format"""
        
        humanized = {
            "performance_optimization": "Performance Optimization",
            "migration_modernization": "Migration and Modernization",
            "architecture_patterns": "Architecture and Design Patterns",
            "developer_productivity": "Developer Productivity",
            "security_compliance": "Security and Compliance"
        }
        
        return humanized.get(pattern_name, pattern_name.replace("_", " ").title())
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using simple heuristics"""
        
        if not str1 or not str2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0