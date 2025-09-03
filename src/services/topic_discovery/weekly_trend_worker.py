# src/services/topic_discovery/weekly_trend_worker.py
"""
Weekly Trend Worker - Main orchestration class for automated trend discovery
and CSV export to S3. Replaces database dependencies with cloud storage.
"""

import asyncio
import csv
import fcntl
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

from src.py_env import (
    aws_access_key_id,
    aws_region,
    aws_secret_access_key,
    s3_bucket_name,
)

from .news_scanner import TechNewsScanner
from .trend_spotter import TrendSpotter


class FileLockManager:
    """Context manager for file-based locking to prevent resource leaks"""

    def __init__(self, lock_file_path: Path, logger: logging.Logger):
        self.lock_file_path = lock_file_path
        self.logger = logger
        self.lock_file = None

    def __enter__(self):
        """Acquire lock when entering context"""
        try:
            self.lock_file = open(self.lock_file_path, "w")
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(f"{os.getpid()}\n{datetime.now(UTC).isoformat()}\n")
            self.lock_file.flush()
            return self
        except OSError as e:
            if self.lock_file and not self.lock_file.closed:
                try:
                    self.lock_file.close()
                except Exception:
                    pass
            self.lock_file = None
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock when exiting context"""
        if self.lock_file and not self.lock_file.closed:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                self.logger.warning(f"Error releasing file lock: {str(e)}")
            finally:
                try:
                    self.lock_file.close()
                except Exception as e:
                    self.logger.warning(f"Error closing lock file: {str(e)}")

        # Clean up lock file
        try:
            if self.lock_file_path.exists():
                self.lock_file_path.unlink()
        except Exception as e:
            self.logger.warning(f"Error removing lock file: {str(e)}")


class WeeklyTrendWorker:
    """
    Orchestrates the weekly trend discovery process combining RSS feeds and external APIs,
    then exports results to CSV and uploads to S3.
    """

    def __init__(self, config: dict | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()

        # Initialize components
        self.news_scanner = TechNewsScanner()
        self.trend_spotter = TrendSpotter()

        # S3 client setup
        self.s3_client = None
        self._init_s3_client()

        # File paths
        self.status_file_path = Path(self.config["status_file_path"])
        self.backup_dir = Path(self.config["local_backup_dir"])
        self.lock_file_path = Path(self.config["lock_file_path"])

        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.status_file_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> dict:
        """Default configuration for the worker"""
        return {
            "max_trends_per_run": 50,
            "score_threshold": 0.1,
            "concurrent_execution_check": True,
            "retry_attempts": 3,
            "retry_delay": 5,
            "s3_bucket": s3_bucket_name or "trending-topics-data",
            "s3_key_prefix": "trends/",
            "local_backup_dir": "./data/backups",
            "status_file_path": "./data/worker_status.json",
            "lock_file_path": "./data/worker.lock",
        }

    def _init_s3_client(self):
        """Initialize S3 client with error handling"""
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region or "us-east-1",
                )
                self.logger.info("S3 client initialized successfully")
            else:
                # Try to use default AWS credentials (IAM role, etc.)
                self.s3_client = boto3.client(
                    "s3", region_name=aws_region or "us-east-1"
                )
                self.logger.info("S3 client initialized with default credentials")

            # Test S3 access
            self._test_s3_access()

        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            self.s3_client = None

    def _test_s3_access(self):
        """Test S3 bucket access"""
        if not self.s3_client:
            return False

        try:
            self.s3_client.head_bucket(Bucket=self.config["s3_bucket"])
            self.logger.info(f"S3 bucket '{self.config['s3_bucket']}' is accessible")
            return True
        except Exception as e:
            self.logger.warning(f"S3 bucket access test failed: {str(e)}")
            return False

    async def run_weekly_discovery(self) -> dict[str, Any]:
        """
        Main orchestration method for weekly trend discovery

        Returns:
            Dict containing execution results and status
        """
        execution_start = datetime.now(UTC)
        results = {
            "status": "failed",
            "started_at": execution_start.isoformat(),
            "completed_at": None,
            "trends_discovered": 0,
            "csv_file_uploaded": None,
            "local_backup_file": None,
            "error_message": None,
        }

        # Handle concurrent execution check with proper resource management
        if self.config["concurrent_execution_check"]:
            try:
                with FileLockManager(self.lock_file_path, self.logger):
                    return await self._run_discovery_with_lock(execution_start, results)
            except OSError:
                error_msg = "Another worker instance is already running"
                self.logger.error(error_msg)
                results["error_message"] = error_msg
                results["completed_at"] = datetime.now(UTC).isoformat()
                self._update_status_file(results)
                return results
        else:
            return await self._run_discovery_with_lock(execution_start, results)

    async def _run_discovery_with_lock(
        self, execution_start: datetime, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Run discovery process with lock already acquired"""
        try:
            self.logger.info("Starting weekly trend discovery process")

            # Step 1: Collect trends from RSS news sources
            self.logger.info("Phase 1: Collecting trends from RSS feeds")
            news_trends = await self._collect_news_trends()
            self.logger.info(f"Collected {len(news_trends)} trends from RSS feeds")

            # Step 2: Collect trends from external APIs
            self.logger.info("Phase 2: Collecting trends from external APIs")
            external_trends = await self._collect_external_trends()
            self.logger.info("Collected external trend data")

            # Step 3: Aggregate and score all trends
            self.logger.info("Phase 3: Aggregating and scoring trends")
            aggregated_trends = await self._aggregate_and_score_trends(
                news_trends, external_trends
            )
            self.logger.info(f"Aggregated {len(aggregated_trends)} total trends")

            # Step 4: Export to CSV
            self.logger.info("Phase 4: Exporting trends to CSV")
            csv_file_path = await self._export_trends_to_csv(aggregated_trends)
            results["local_backup_file"] = str(csv_file_path)

            # Step 5: Upload to S3
            self.logger.info("Phase 5: Uploading to S3")
            s3_url = await self._upload_to_s3(csv_file_path)
            if s3_url:
                results["csv_file_uploaded"] = s3_url

            # Update results
            execution_end = datetime.now(UTC)
            results.update(
                {
                    "status": "completed",
                    "completed_at": execution_end.isoformat(),
                    "trends_discovered": len(aggregated_trends),
                    "execution_duration": (
                        execution_end - execution_start
                    ).total_seconds(),
                }
            )

            self.logger.info(
                f"Weekly trend discovery completed successfully in {results['execution_duration']:.1f}s"
            )

        except Exception as e:
            error_msg = f"Error in weekly trend discovery: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results["error_message"] = error_msg
            results["completed_at"] = datetime.now(UTC).isoformat()

        # Update status file with results
        self._update_status_file(results)
        return results

    async def _collect_news_trends(self) -> list[dict]:
        """Collect trends from RSS news sources"""
        try:
            return await self.news_scanner.scan_tech_news()
        except Exception as e:
            self.logger.error(f"Error collecting news trends: {str(e)}")
            return []

    async def _collect_external_trends(self) -> dict:
        """Collect trends from external APIs"""
        try:
            # Run external trend collection in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            external_data = await loop.run_in_executor(
                None, self.trend_spotter.get_weekly_trend
            )
            return external_data or {}
        except Exception as e:
            self.logger.error(f"Error collecting external trends: {str(e)}")
            return {}

    async def _aggregate_and_score_trends(
        self, news_trends: list[dict], external_trend: dict
    ) -> list[dict]:
        """
        Combine and score trends from multiple sources

        Args:
            news_trends: List of trends from RSS feeds
            external_trend: Single trend from external APIs

        Returns:
            List of aggregated trends with combined scores
        """
        self.logger.info(
            f"Aggregating {len(news_trends)} news trends with external data"
        )

        # Create a mapping for easier lookup
        trend_map = {}

        # Add all news trends to the map
        for trend in news_trends:
            topic = trend["topic"].lower().strip()
            if topic not in trend_map:
                trend_map[topic] = {
                    "topic": trend["topic"],
                    "news_score": trend["score"],
                    "external_score": 0.0,
                    "sources": trend.get("sources", ""),
                    "article_count": trend.get("article_count", 0),
                    "discovery_method": trend.get("discovery_method", "rss_analysis"),
                    "confidence_level": trend.get("confidence_level", "low"),
                    "discovered_at": trend["discovered_at"],
                    "duplicate_flag": False,
                }
            else:
                # Merge duplicate topics
                trend_map[topic]["news_score"] += trend["score"]
                trend_map[topic]["article_count"] += trend.get("article_count", 0)
                trend_map[topic]["duplicate_flag"] = True

        # Add external trend data
        if external_trend and "topic" in external_trend:
            external_topic = external_trend["topic"].lower().strip()
            external_score = external_trend.get("score", 0.0)

            if external_topic in trend_map:
                # Boost existing trend with external data
                trend_map[external_topic]["external_score"] = external_score
                trend_map[external_topic]["discovery_method"] = "combined"
                # Upgrade confidence if external validation is strong
                if external_score > 0.7:
                    trend_map[external_topic]["confidence_level"] = "high"
            else:
                # Add new external trend
                trend_map[external_topic] = {
                    "topic": external_trend["topic"],
                    "news_score": 0.0,
                    "external_score": external_score,
                    "sources": "external_api",
                    "article_count": 0,
                    "discovery_method": "api_analysis",
                    "confidence_level": "high" if external_score > 0.7 else "medium",
                    "discovered_at": datetime.now(UTC),
                    "duplicate_flag": False,
                }

        # Calculate final scores and prepare output
        aggregated_trends = []
        for trend_data in trend_map.values():
            # Combined scoring algorithm
            news_weight = 0.6
            external_weight = 0.4

            final_score = (
                trend_data["news_score"] * news_weight
                + trend_data["external_score"] * external_weight
            )

            # Apply threshold filter
            if final_score >= self.config["score_threshold"]:
                trend_entry = {
                    "topic": trend_data["topic"],
                    "source": "weekly_worker",
                    "score": round(final_score, 3),
                    "news_score": round(trend_data["news_score"], 3),
                    "external_score": round(trend_data["external_score"], 3),
                    "sources": trend_data["sources"],
                    "article_count": trend_data["article_count"],
                    "discovery_method": trend_data["discovery_method"],
                    "confidence_level": trend_data["confidence_level"],
                    "discovered_at": trend_data["discovered_at"].isoformat()
                    if hasattr(trend_data["discovered_at"], "isoformat")
                    else str(trend_data["discovered_at"]),
                    "duplicate_flag": trend_data["duplicate_flag"],
                }
                aggregated_trends.append(trend_entry)

        # Sort by final score and limit results
        sorted_trends = sorted(
            aggregated_trends, key=lambda x: x["score"], reverse=True
        )
        limited_trends = sorted_trends[: self.config["max_trends_per_run"]]

        self.logger.info(f"Final aggregated trends: {len(limited_trends)}")
        return limited_trends

    async def _export_trends_to_csv(self, trends: list[dict]) -> Path:
        """
        Export trends to CSV file with standardized format

        Returns:
            Path to the created CSV file
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"weekly-trends-{timestamp}.csv"
        csv_file_path = self.backup_dir / filename

        # CSV columns as specified in the design document
        csv_columns = [
            "topic",
            "source",
            "score",
            "news_score",
            "external_score",
            "sources",
            "article_count",
            "discovery_method",
            "confidence_level",
            "discovered_at",
            "duplicate_flag",
        ]

        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

            for trend in trends:
                # Ensure all required fields are present
                row = {col: trend.get(col, "") for col in csv_columns}
                writer.writerow(row)

        self.logger.info(f"Exported {len(trends)} trends to {csv_file_path}")
        return csv_file_path

    async def _upload_to_s3(self, csv_file_path: Path) -> str | None:
        """
        Upload CSV file to S3 with retry logic

        Args:
            csv_file_path: Path to the CSV file to upload

        Returns:
            S3 URL if successful, None if failed
        """
        if not self.s3_client:
            self.logger.warning("S3 client not available, skipping upload")
            return None

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
        s3_key = f"{self.config['s3_key_prefix']}weekly-trends-{timestamp}.csv"

        for attempt in range(self.config["retry_attempts"]):
            try:
                # Upload with metadata
                extra_args = {
                    "Metadata": {
                        "source": "weekly-trend-worker",
                        "data_type": "trending_topics",
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    "ContentType": "text/csv",
                }

                self.s3_client.upload_file(
                    str(csv_file_path),
                    self.config["s3_bucket"],
                    s3_key,
                    ExtraArgs=extra_args,
                )

                s3_url = f"s3://{self.config['s3_bucket']}/{s3_key}"
                self.logger.info(f"Successfully uploaded to {s3_url}")
                return s3_url

            except Exception as e:
                self.logger.error(f"S3 upload attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config["retry_attempts"] - 1:
                    wait_time = self.config["retry_delay"] * (
                        2**attempt
                    )  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        self.logger.error("All S3 upload attempts failed")
        return None

    def _update_status_file(self, results: dict):
        """Update local status file with execution results"""
        try:
            status_data = {
                "worker_type": "weekly_trend_worker",
                "last_run_status": results["status"],
                "last_started_at": results["started_at"],
                "last_completed_at": results["completed_at"],
                "trends_discovered": results["trends_discovered"],
                "csv_file_uploaded": results.get("csv_file_uploaded"),
                "local_backup_file": results.get("local_backup_file"),
                "error_message": results.get("error_message"),
                "next_scheduled_run": None,  # Can be set by scheduler
                "updated_at": datetime.now(UTC).isoformat(),
            }

            with open(self.status_file_path, "w") as f:
                json.dump(status_data, f, indent=2)

            self.logger.info(f"Status file updated: {self.status_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to update status file: {str(e)}")

    def get_last_status(self) -> dict:
        """Get the last execution status"""
        try:
            if self.status_file_path.exists():
                with open(self.status_file_path) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading status file: {str(e)}")

        return {"worker_type": "weekly_trend_worker", "status": "never_run"}

    def cleanup(self):
        """Cleanup method - no longer needed with context manager approach"""
        # Context managers handle all resource cleanup automatically
        pass
