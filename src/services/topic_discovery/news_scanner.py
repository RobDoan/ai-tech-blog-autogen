# services/topic_discovery/news_scanner.py
import asyncio
import logging
from datetime import UTC, datetime

import aiohttp
import feedparser


class TechNewsScanner:
    """Refactored news scanner using high-quality RSS feeds from major tech companies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # High-quality RSS feeds from major tech companies as specified in requirements
        self.news_sources = [
            {
                "name": "Netflix Tech Blog",
                "rss_url": "https://netflixtechblog.com/feed",
                "weight": 0.9,
                "timeout": 30,
            },
            {
                "name": "GDB Blog",
                "rss_url": "https://feeds.feedburner.com/GDBcode",
                "weight": 0.8,
                "timeout": 30,
            },
            {
                "name": "Facebook Engineering",
                "rss_url": "https://engineering.fb.com/feed/",
                "weight": 0.9,
                "timeout": 30,
            },
            {
                "name": "AWS Blog",
                "rss_url": "https://aws.amazon.com/blogs/aws/feed/",
                "weight": 0.9,
                "timeout": 30,
            },
            {
                "name": "Stripe Engineering",
                "rss_url": "https://stripe.com/blog/engineering/feed.xml",
                "weight": 0.8,
                "timeout": 30,
            },
            {
                "name": "Hacker News",
                "rss_url": "https://news.ycombinator.com/rss",
                "weight": 0.7,
                "timeout": 30,
            },
        ]

    async def scan_tech_news(self) -> list[dict]:
        """Scan tech news from multiple high-quality RSS sources with enhanced error handling"""
        self.logger.info(f"Starting news scan from {len(self.news_sources)} sources")
        all_articles = []
        successful_sources = 0
        failed_sources = 0

        # Process feeds concurrently for better performance
        tasks = [
            self._scan_rss_feed_with_error_handling(source)
            for source in self.news_sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            source = self.news_sources[i]
            if isinstance(result, Exception):
                failed_sources += 1
                self.logger.error(
                    f"Failed to scan {source['name']}: {str(result)}",
                    extra={
                        "source": source["name"],
                        "url": source["rss_url"],
                        "error": str(result),
                    },
                )
            elif result:
                successful_sources += 1
                all_articles.extend(result)
                self.logger.info(
                    f"Successfully scanned {source['name']}: {len(result)} articles",
                    extra={"source": source["name"], "article_count": len(result)},
                )
            else:
                failed_sources += 1
                self.logger.warning(f"No articles found from {source['name']}")

        self.logger.info(
            f"Scan complete: {successful_sources} successful, {failed_sources} failed, {len(all_articles)} total articles"
        )

        # Extract topics from articles with enhanced processing
        topics = self._extract_topics_from_articles(all_articles)
        return topics

    async def _scan_rss_feed_with_error_handling(
        self, source: dict
    ) -> list[dict] | None:
        """Wrapper for RSS feed scanning with comprehensive error handling"""
        try:
            return await self._scan_rss_feed(source)
        except TimeoutError:
            self.logger.error(
                f"Timeout scanning {source['name']} after {source.get('timeout', 30)}s"
            )
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP client error scanning {source['name']}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error scanning {source['name']}: {str(e)}")
            return None

    async def _scan_rss_feed(self, source: dict) -> list[dict]:
        """Scan individual RSS feed with timeout and proper headers"""
        timeout = aiohttp.ClientTimeout(total=source.get("timeout", 30))
        headers = {
            "User-Agent": "TechTrendWorker/1.0 (https://github.com/RobDoan/ai-tech-blog-autogen)"
        }

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(source["rss_url"]) as response:
                response.raise_for_status()  # Raise for HTTP errors
                content = await response.text()

        feed = feedparser.parse(content)
        articles = []

        # Check if feed parsing was successful
        if hasattr(feed, "bozo") and feed.bozo:
            self.logger.warning(
                f"Feed parsing issues for {source['name']}: {getattr(feed, 'bozo_exception', 'Unknown error')}"
            )

        # Process up to 25 articles per source for better coverage
        entries = feed.entries[:25] if hasattr(feed, "entries") else []

        for entry in entries:
            try:
                article = {
                    "title": getattr(entry, "title", ""),
                    "description": getattr(
                        entry, "description", getattr(entry, "summary", "")
                    ),
                    "link": getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                    "source": source["name"],
                    "source_url": source["rss_url"],
                    "weight": source["weight"],
                    "discovered_at": datetime.now(UTC),
                }

                # Only add articles with valid title and content
                if article["title"] and (article["description"] or article["link"]):
                    articles.append(article)

            except Exception as e:
                self.logger.warning(
                    f"Error processing entry from {source['name']}: {str(e)}"
                )
                continue

        self.logger.debug(
            f"Extracted {len(articles)} valid articles from {source['name']}"
        )
        return articles

    def _extract_topics_from_articles(self, articles: list[dict]) -> list[dict]:
        """Extract trending topics from articles with enhanced keyword matching and source attribution"""
        self.logger.info(f"Extracting topics from {len(articles)} articles")

        # Enhanced tech keywords with broader coverage
        tech_keywords = [
            # AI/ML
            "AI",
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
            "GPT",
            "OpenAI",
            "ChatGPT",
            "LLM",
            "Large Language Models",
            "Computer Vision",
            "NLP",
            # Programming Languages
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "Go",
            "Rust",
            "C++",
            "Swift",
            "Kotlin",
            # Frameworks & Libraries
            "React",
            "Vue",
            "Angular",
            "Node.js",
            "Django",
            "Flask",
            "Spring",
            "TensorFlow",
            "PyTorch",
            # Cloud & Infrastructure
            "AWS",
            "Azure",
            "GCP",
            "Google Cloud",
            "Docker",
            "Kubernetes",
            "DevOps",
            "CI/CD",
            "Serverless",
            "Microservices",
            "API Gateway",
            "Load Balancer",
            "CDN",
            # Blockchain & Web3
            "Blockchain",
            "Web3",
            "NFT",
            "Cryptocurrency",
            "Bitcoin",
            "Ethereum",
            "DeFi",
            "Smart Contracts",
            "DAO",
            "Metaverse",
            # Mobile & Frontend
            "Mobile",
            "iOS",
            "Android",
            "Flutter",
            "React Native",
            "Progressive Web App",
            "PWA",
            # Data & Analytics
            "Data Science",
            "Big Data",
            "Analytics",
            "Business Intelligence",
            "ETL",
            "Data Pipeline",
            "Apache Spark",
            "Hadoop",
            "Kafka",
            "Elasticsearch",
            # APIs & Architecture
            "API",
            "REST",
            "GraphQL",
            "gRPC",
            "WebSocket",
            "Event Driven",
            "Message Queue",
            # Security
            "Cybersecurity",
            "Security",
            "Privacy",
            "GDPR",
            "Authentication",
            "OAuth",
            "Zero Trust",
            "Penetration Testing",
            "Vulnerability",
            "Encryption",
            # Databases
            "Database",
            "SQL",
            "NoSQL",
            "PostgreSQL",
            "MongoDB",
            "Redis",
            "Cassandra",
            "DynamoDB",
            # Development Tools
            "GitHub",
            "Git",
            "GitLab",
            "Jenkins",
            "GitHub Actions",
            "Terraform",
            "Ansible",
            # Emerging Tech
            "Quantum Computing",
            "Edge Computing",
            "5G",
            "IoT",
            "AR",
            "VR",
            "Augmented Reality",
            "Virtual Reality",
        ]

        topic_data = {}

        for article in articles:
            text = f"{article['title']} {article['description']}".lower()
            article_source = article["source"]

            for keyword in tech_keywords:
                if keyword.lower() in text:
                    # Calculate enhanced relevance score
                    relevance_score = self._calculate_enhanced_relevance_score(
                        text, keyword.lower(), article["title"].lower()
                    )
                    weighted_score = article["weight"] * relevance_score

                    if keyword not in topic_data:
                        topic_data[keyword] = {
                            "total_score": 0,
                            "article_count": 0,
                            "sources": set(),
                            "confidence_factors": [],
                        }

                    topic_data[keyword]["total_score"] += weighted_score
                    topic_data[keyword]["article_count"] += 1
                    topic_data[keyword]["sources"].add(article_source)
                    topic_data[keyword]["confidence_factors"].append(
                        {
                            "source": article_source,
                            "weight": article["weight"],
                            "relevance": relevance_score,
                            "in_title": keyword.lower() in article["title"].lower(),
                        }
                    )

        # Convert to topic list with enhanced metadata
        topics = []
        for topic, data in topic_data.items():
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                data["total_score"],
                data["article_count"],
                len(data["sources"]),
                data["confidence_factors"],
            )

            topic_entry = {
                "topic": topic,
                "score": round(data["total_score"], 3),
                "news_score": round(data["total_score"], 3),  # For CSV compatibility
                "external_score": 0.0,  # Will be filled by external trend analysis
                "source": "rss_analysis",
                "sources": ",".join(sorted(data["sources"])),
                "article_count": data["article_count"],
                "discovery_method": "rss_analysis",
                "confidence_level": confidence_level,
                "discovered_at": datetime.now(UTC),
            }
            topics.append(topic_entry)

        # Sort by score and return top topics
        sorted_topics = sorted(topics, key=lambda x: x["score"], reverse=True)
        self.logger.info(f"Extracted {len(sorted_topics)} unique topics")

        return sorted_topics[:50]  # Return top 50 trending topics

    def _calculate_confidence_level(
        self,
        total_score: float,
        article_count: int,
        source_count: int,
        confidence_factors: list[dict],
    ) -> str:
        """Calculate confidence level based on multiple factors"""
        # Base score threshold
        if total_score >= 5.0 and article_count >= 5 and source_count >= 3:
            return "high"
        elif total_score >= 2.0 and article_count >= 3 and source_count >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_enhanced_relevance_score(
        self, text: str, keyword: str, title: str
    ) -> float:
        """Calculate enhanced relevance score for a keyword with multiple factors"""
        base_score = 0.0

        # Count occurrences in full text
        text_count = text.count(keyword)
        base_score += text_count * 1.0

        # Title presence boost (most important signal)
        if keyword in title:
            base_score += 3.0

        # Word boundary matching (prefer exact word matches)
        import re

        word_pattern = rf"\b{re.escape(keyword)}\b"
        exact_matches = len(re.findall(word_pattern, text, re.IGNORECASE))
        base_score += exact_matches * 0.5

        # Position boost (earlier in text = more relevant)
        first_occurrence = text.find(keyword)
        if first_occurrence != -1:
            # Score boost inversely proportional to position
            position_boost = max(0.5, 2.0 - (first_occurrence / len(text)))
            base_score += position_boost

        # Frequency normalization (avoid keyword stuffing)
        text_length = len(text.split())
        if text_length > 0:
            frequency_ratio = text_count / text_length
            # Diminishing returns for high frequency
            normalized_frequency = min(frequency_ratio * 10, 2.0)
            base_score += normalized_frequency

        return max(base_score, 0.1)  # Minimum score to avoid zeros
