# src/services/topic_discovery/enhanced_content_extractor.py
"""
Enhanced Content Extractor for Blog Title Discovery

This module extends existing RSS processing to extract detailed article content
for AI-powered semantic analysis and specific blog title generation.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from .config import RSS_SOURCES, RSSSourceConfig


@dataclass
class TechnicalDetails:
    """Extracted technical metrics and details from article content"""
    metrics: list[str] = field(default_factory=list)  # Performance numbers, percentages
    version_numbers: list[str] = field(default_factory=list)  # Software versions
    company_names: list[str] = field(default_factory=list)  # Organizations mentioned
    technologies: list[str] = field(default_factory=list)  # Programming languages, frameworks
    code_examples: list[str] = field(default_factory=list)  # Code snippets found
    implementation_details: list[str] = field(default_factory=list)  # Technical approaches


@dataclass
class ContentPatterns:
    """Identified content patterns that suggest actionable topics"""
    pattern_type: str  # "how-to", "performance", "comparison", "implementation"
    confidence_score: float  # 0-1 confidence in pattern detection
    key_phrases: list[str] = field(default_factory=list)
    actionable_indicators: list[str] = field(default_factory=list)


@dataclass
class ArticleContent:
    """Comprehensive article content for AI analysis"""
    title: str
    summary: str
    full_content: str | None
    source_url: str
    source_name: str
    published_date: datetime
    author: str | None = None

    # Enhanced extraction data
    technical_details: TechnicalDetails = field(default_factory=TechnicalDetails)
    content_patterns: ContentPatterns = field(default_factory=lambda: ContentPatterns("general", 0.0))
    raw_html: str | None = None

    # Metadata
    word_count: int = 0
    language: str = "en"
    content_quality_score: float = 0.0  # Based on length, structure, etc.


class EnhancedContentExtractor:
    """
    Enhanced content extraction that goes beyond basic RSS parsing
    to capture detailed article content for AI semantic analysis.
    """

    def __init__(self, timeout: int = 30, max_concurrent: int = 5):
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # Content extraction patterns
        self._compile_patterns()

        # HTTP session for fetching full content
        self.session: aiohttp.ClientSession | None = None

    def _compile_patterns(self):
        """Compile regex patterns for content analysis"""

        # Performance metrics patterns
        self.metrics_patterns = [
            re.compile(r'\b(\d+(?:\.\d+)?%)\b'),  # Percentages
            re.compile(r'\b(\d+(?:\.\d+)?\s*(?:ms|seconds?|minutes?|hours?))\b', re.IGNORECASE),  # Time
            re.compile(r'\b(\d+(?:\.\d+)?\s*(?:MB|GB|KB|bytes?))\b', re.IGNORECASE),  # Memory/size
            re.compile(r'\b(\d+(?:\.\d+)?\s*(?:x|times?)\s*(?:faster|slower|better|worse))\b', re.IGNORECASE),
            re.compile(r'\b(improved?|reduced?|increased?)\s+(?:by\s+)?(\d+(?:\.\d+)?%?)\b', re.IGNORECASE),
            re.compile(r'\b(\d+(?:\.\d+)?\s*(?:QPS|RPS|req/s|requests?\s*per\s*second))\b', re.IGNORECASE)
        ]

        # Version number patterns
        self.version_patterns = [
            re.compile(r'\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z]\w*)?)\b'),  # Standard versions
            re.compile(r'\b([A-Z][a-z]+\s+\d+(?:\.\d+)?)\b'),  # "React 19", "Python 3.12"
            re.compile(r'\b(Node\.js\s+\d+(?:\.\d+)?)\b', re.IGNORECASE),
        ]

        # Company/organization patterns
        self.company_patterns = [
            re.compile(r'\b(Netflix|Google|Facebook|Meta|Amazon|AWS|Microsoft|Apple|Stripe|Uber|Airbnb|Spotify|Twitter|X|LinkedIn|GitHub|GitLab|Vercel|Netlify|Cloudflare)\b'),
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:team|engineering|developers?|announced|released|built|created|launched)\b')
        ]

        # Technology patterns
        self.tech_patterns = [
            re.compile(r'\b(React|Vue|Angular|Next\.js|Node\.js|Python|JavaScript|TypeScript|Go|Rust|Java|C\+\+|C#|Swift|Kotlin)\b'),
            re.compile(r'\b(Docker|Kubernetes|AWS|Azure|GCP|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch)\b'),
            re.compile(r'\b(GraphQL|REST|API|microservices?|serverless|containerization|CI/CD)\b', re.IGNORECASE)
        ]

        # Actionable content patterns
        self.actionable_patterns = {
            "how-to": [
                re.compile(r'\bhow\s+to\b', re.IGNORECASE),
                re.compile(r'\bstep\s+by\s+step\b', re.IGNORECASE),
                re.compile(r'\bguide\s+to\b', re.IGNORECASE),
                re.compile(r'\btutorial\b', re.IGNORECASE)
            ],
            "performance": [
                re.compile(r'\b(?:improved?|optimized?|faster|better|enhanced?)\b', re.IGNORECASE),
                re.compile(r'\b(?:performance|speed|efficiency|latency|throughput)\b', re.IGNORECASE),
                re.compile(r'\b(?:reduced?|decreased?|minimized?)\s+(?:by\s+)?\d+', re.IGNORECASE)
            ],
            "comparison": [
                re.compile(r'\bvs\.?\s+\b', re.IGNORECASE),
                re.compile(r'\bcompared?\s+(?:to|with)\b', re.IGNORECASE),
                re.compile(r'\b(?:better|worse|faster|slower)\s+than\b', re.IGNORECASE),
                re.compile(r'\b(?:advantages?|disadvantages?|pros?|cons?)\s+of\b', re.IGNORECASE)
            ],
            "implementation": [
                re.compile(r'\bimplemented?|implementing|implementation\b', re.IGNORECASE),
                re.compile(r'\bbuilt|building|creating|developed?\b', re.IGNORECASE),
                re.compile(r'\busing|with|via|through\b', re.IGNORECASE)
            ]
        }

    async def extract_article_content(self, source_configs: list[RSSSourceConfig] | None = None) -> list[ArticleContent]:
        """
        Extract comprehensive article content from RSS sources
        
        Args:
            source_configs: RSS sources to process (uses default if None)
            
        Returns:
            List of extracted and analyzed article content
        """
        if source_configs is None:
            source_configs = RSS_SOURCES

        self.logger.info(f"Starting enhanced content extraction from {len(source_configs)} RSS sources")

        # Initialize HTTP session
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session

            # Process RSS feeds concurrently with limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._process_rss_source(semaphore, source_config)
                for source_config in source_configs
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results and filter out exceptions
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"RSS processing error: {str(result)}")

        self.session = None
        self.logger.info(f"Extracted {len(all_articles)} articles with enhanced content")
        return all_articles

    async def _process_rss_source(self, semaphore: asyncio.Semaphore, source_config: RSSSourceConfig) -> list[ArticleContent]:
        """Process a single RSS source with enhanced content extraction"""

        async with semaphore:
            try:
                self.logger.info(f"Processing RSS source: {source_config.name}")

                # Fetch RSS feed
                async with self.session.get(source_config.rss_url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch RSS from {source_config.name}: HTTP {response.status}")
                        return []

                    rss_content = await response.text()

                # Parse RSS feed
                feed = feedparser.parse(rss_content)
                if not feed.entries:
                    self.logger.warning(f"No entries found in RSS feed: {source_config.name}")
                    return []

                # Process entries (limit to recent ones for performance)
                recent_entries = feed.entries[:10]  # Limit to 10 most recent

                articles = []
                for entry in recent_entries:
                    try:
                        article = await self._extract_article_details(entry, source_config)
                        if article:
                            articles.append(article)
                    except Exception as e:
                        self.logger.error(f"Error processing article '{entry.get('title', 'Unknown')}': {str(e)}")

                self.logger.info(f"Processed {len(articles)} articles from {source_config.name}")
                return articles

            except Exception as e:
                self.logger.error(f"Error processing RSS source {source_config.name}: {str(e)}")
                return []

    async def _extract_article_details(self, entry: dict, source_config: RSSSourceConfig) -> ArticleContent | None:
        """Extract comprehensive details from a single RSS entry"""

        try:
            # Basic RSS data
            title = entry.get('title', '').strip()
            if not title:
                return None

            summary = entry.get('summary', '') or entry.get('description', '')
            source_url = entry.get('link', '')
            published_date = self._parse_date(entry.get('published'))
            author = entry.get('author')

            # Initialize article content
            article = ArticleContent(
                title=title,
                summary=self._clean_html(summary),
                full_content=None,
                source_url=source_url,
                source_name=source_config.name,
                published_date=published_date,
                author=author
            )

            # Try to fetch full article content
            if source_url:
                full_content, raw_html = await self._fetch_full_article_content(source_url)
                article.full_content = full_content
                article.raw_html = raw_html
                article.word_count = len(full_content.split()) if full_content else len(article.summary.split())
            else:
                article.word_count = len(article.summary.split())

            # Extract technical details
            content_to_analyze = article.full_content or article.summary
            article.technical_details = self._extract_technical_details(content_to_analyze)

            # Identify content patterns
            try:
                article.content_patterns = self._identify_content_patterns(title, content_to_analyze)
            except Exception as e:
                self.logger.error(f"Error in _identify_content_patterns: {str(e)}")
                # Create a default pattern as fallback
                article.content_patterns = ContentPatterns(
                    pattern_type="general",
                    confidence_score=0.1,
                    key_phrases=[],
                    actionable_indicators=[]
                )

            # Calculate content quality score
            article.content_quality_score = self._calculate_content_quality(article)

            return article

        except Exception as e:
            self.logger.error(f"Error extracting article details: {str(e)}")
            return None

    async def _fetch_full_article_content(self, url: str) -> tuple[str | None, str | None]:
        """Fetch and extract full article content from URL"""

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch full content from {url}: HTTP {response.status}")
                    return None, None

                raw_html = await response.text()

                # Parse HTML and extract main content
                soup = BeautifulSoup(raw_html, 'html.parser')

                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                    tag.decompose()

                # Try common content selectors
                content_selectors = [
                    'article',
                    '.post-content',
                    '.entry-content',
                    '.content',
                    'main',
                    '.article-body',
                    '#content'
                ]

                content_text = None
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content_text = content_elem.get_text(separator=' ', strip=True)
                        break

                # Fallback to body content
                if not content_text:
                    body = soup.find('body')
                    if body:
                        content_text = body.get_text(separator=' ', strip=True)

                # Clean and limit content
                if content_text:
                    content_text = self._clean_content_text(content_text)
                    # Limit to reasonable size (first 5000 words for analysis)
                    words = content_text.split()
                    if len(words) > 5000:
                        content_text = ' '.join(words[:5000])

                return content_text, raw_html[:10000]  # Limit raw HTML storage

        except Exception as e:
            self.logger.warning(f"Error fetching full content from {url}: {str(e)}")
            return None, None

    def _extract_technical_details(self, content: str) -> TechnicalDetails:
        """Extract technical metrics, versions, companies, and technologies"""

        details = TechnicalDetails()

        if not content:
            return details

        # Extract metrics
        for pattern in self.metrics_patterns:
            matches = pattern.findall(content)
            details.metrics.extend(matches)

        # Extract version numbers
        for pattern in self.version_patterns:
            matches = pattern.findall(content)
            details.version_numbers.extend(matches)

        # Extract company names
        for pattern in self.company_patterns:
            matches = pattern.findall(content)
            if isinstance(matches[0] if matches else None, tuple):
                # Handle patterns with groups
                details.company_names.extend([match[0] for match in matches])
            else:
                details.company_names.extend(matches)

        # Extract technologies
        for pattern in self.tech_patterns:
            matches = pattern.findall(content)
            details.technologies.extend(matches)

        # Extract code examples (simple heuristic)
        code_indicators = ['```', 'function ', 'def ', 'class ', 'import ', '  return', '  if ']
        for indicator in code_indicators:
            if indicator in content:
                # Find code blocks around the indicator
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if indicator in line:
                        # Capture surrounding context as code example
                        start = max(0, i-2)
                        end = min(len(lines), i+5)
                        code_snippet = '\n'.join(lines[start:end]).strip()
                        if len(code_snippet) > 20:  # Only meaningful snippets
                            details.code_examples.append(code_snippet[:200])  # Limit length
                        break

        # Deduplicate lists
        details.metrics = list(set(details.metrics))
        details.version_numbers = list(set(details.version_numbers))
        details.company_names = list(set(details.company_names))
        details.technologies = list(set(details.technologies))
        details.code_examples = list(set(details.code_examples))

        return details

    def _identify_content_patterns(self, title: str, content: str) -> ContentPatterns:
        """Identify actionable content patterns in the article"""

        combined_text = f"{title} {content}".lower()
        pattern_scores = {}
        all_key_phrases = []

        # Check each pattern type
        for pattern_type, patterns in self.actionable_patterns.items():
            matches = 0
            pattern_phrases = []

            for pattern in patterns:
                found_matches = pattern.findall(combined_text)
                matches += len(found_matches)
                pattern_phrases.extend(found_matches)

            if matches > 0:
                # Calculate confidence based on number of matches and content length
                confidence = min(matches / 5.0, 1.0)  # Normalize to 0-1
                pattern_scores[pattern_type] = confidence
                all_key_phrases.extend(pattern_phrases)

        # Select the highest scoring pattern
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return ContentPatterns(
                pattern_type=best_pattern[0],
                confidence_score=best_pattern[1],
                key_phrases=all_key_phrases[:10],  # Limit to top 10
                actionable_indicators=self._find_actionable_indicators(title, content)
            )
        else:
            return ContentPatterns(
                pattern_type="general",
                confidence_score=0.1,
                key_phrases=[],
                actionable_indicators=[]
            )

    def _find_actionable_indicators(self, title: str, content: str) -> list[str]:
        """Find specific phrases that indicate actionable content"""

        indicators = []
        text = f"{title} {content}".lower()

        # Look for specific actionable phrases
        actionable_phrases = [
            "step by step", "how to", "guide to", "tutorial",
            "improved by", "reduced by", "increased by", "faster than",
            "vs", "compared to", "better than", "worse than",
            "implementation", "building", "creating", "developing"
        ]

        for phrase in actionable_phrases:
            if phrase in text:
                # Extract context around the phrase
                start_idx = text.find(phrase)
                context_start = max(0, start_idx - 30)
                context_end = min(len(text), start_idx + len(phrase) + 30)
                context = text[context_start:context_end].strip()
                indicators.append(context)

        return indicators[:5]  # Limit to 5 most relevant

    def _calculate_content_quality(self, article: ArticleContent) -> float:
        """Calculate content quality score based on various factors"""

        score = 0.0

        # Word count factor (sweet spot around 500-2000 words)
        if article.word_count > 100:
            word_score = min(article.word_count / 1000.0, 1.0)
            score += word_score * 0.3

        # Technical detail richness
        tech_details = article.technical_details
        tech_score = (
            len(tech_details.metrics) * 0.1 +
            len(tech_details.version_numbers) * 0.1 +
            len(tech_details.company_names) * 0.1 +
            len(tech_details.technologies) * 0.1 +
            len(tech_details.code_examples) * 0.1
        )
        score += min(tech_score, 0.4)

        # Content pattern confidence
        score += article.content_patterns.confidence_score * 0.3

        # Normalize to 0-1 range
        return min(score, 1.0)

    def _parse_date(self, date_str: str | None) -> datetime:
        """Parse publication date from RSS entry"""

        if not date_str:
            return datetime.now(UTC)

        try:
            # Try feedparser's parsed time first
            if hasattr(date_str, 'struct_time'):
                import time
                return datetime.fromtimestamp(time.mktime(date_str), tz=UTC)

            # Fallback to string parsing
            from dateutil import parser
            return parser.parse(date_str).replace(tzinfo=UTC)

        except Exception:
            return datetime.now(UTC)

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text"""

        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def _clean_content_text(self, text: str) -> str:
        """Clean and normalize extracted content text"""

        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common noise patterns
        noise_patterns = [
            r'\b(?:Click here|Read more|Continue reading|Share this|Subscribe|Newsletter)\b',
            r'\b(?:Advertisement|Sponsored|Promoted)\b',
            r'\b(?:Cookie|Privacy|Terms of Service|Terms of Use)\b'
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()
