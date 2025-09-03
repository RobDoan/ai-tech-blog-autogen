"""
File-based storage system to replace database functionality.
Exports data to JSON and CSV files for easy access and portability.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileStorage:
    """File-based storage system for blog data"""

    def __init__(self, base_path: str = "./data"):
        """Initialize file storage with base directory"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.base_path / "blog_posts").mkdir(exist_ok=True)
        (self.base_path / "trending_topics").mkdir(exist_ok=True)
        (self.base_path / "content_generations").mkdir(exist_ok=True)

    def save_blog_post(self, blog_post_data: dict[str, Any]) -> str:
        """Save blog post data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blog_post_{timestamp}.json"
        filepath = self.base_path / "blog_posts" / filename

        # Add metadata
        blog_post_data.update(
            {"created_at": datetime.now().isoformat(), "file_id": filename}
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(blog_post_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def save_trending_topics(self, topics_data: list[dict[str, Any]]) -> str:
        """Save trending topics to both JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_filename = f"trending_topics_{timestamp}.json"
        json_filepath = self.base_path / "trending_topics" / json_filename

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "topics": topics_data},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save as CSV
        csv_filename = f"trending_topics_{timestamp}.csv"
        csv_filepath = self.base_path / "trending_topics" / csv_filename

        if topics_data:
            with open(csv_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=topics_data[0].keys())
                writer.writeheader()
                writer.writerows(topics_data)

        return str(json_filepath)

    def save_content_generation(self, generation_data: dict[str, Any]) -> str:
        """Save content generation data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"content_gen_{timestamp}.json"
        filepath = self.base_path / "content_generations" / filename

        # Add metadata
        generation_data.update(
            {"created_at": datetime.now().isoformat(), "file_id": filename}
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_blog_posts(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load blog posts from JSON files"""
        blog_posts = []
        blog_posts_dir = self.base_path / "blog_posts"

        if not blog_posts_dir.exists():
            return blog_posts

        json_files = sorted(blog_posts_dir.glob("*.json"), reverse=True)

        if limit:
            json_files = json_files[:limit]

        for filepath in json_files:
            try:
                with open(filepath, encoding="utf-8") as f:
                    blog_posts.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue

        return blog_posts

    def load_trending_topics(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load trending topics from JSON files"""
        all_topics = []
        topics_dir = self.base_path / "trending_topics"

        if not topics_dir.exists():
            return all_topics

        json_files = sorted(topics_dir.glob("*.json"), reverse=True)

        if limit:
            json_files = json_files[:limit]

        for filepath in json_files:
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    if "topics" in data:
                        all_topics.extend(data["topics"])
            except (OSError, json.JSONDecodeError):
                continue

        return all_topics

    def export_to_csv(self, data_type: str, output_path: str | None = None) -> str:
        """Export all data of specified type to a single CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            output_path = str(self.base_path / f"export_{data_type}_{timestamp}.csv")

        if data_type == "blog_posts":
            data = self.load_blog_posts()
        elif data_type == "trending_topics":
            data = self.load_trending_topics()
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if not data:
            return output_path

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Flatten nested dictionaries for CSV export
            flattened_data = []
            for item in data:
                flattened_item = self._flatten_dict(item)
                flattened_data.append(flattened_item)

            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)

        return output_path

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v) if v else ""))
            else:
                items.append((new_key, v))
        return dict(items)

    def get_storage_stats(self) -> dict[str, int]:
        """Get statistics about stored data"""
        stats = {}

        for subdir in ["blog_posts", "trending_topics", "content_generations"]:
            dir_path = self.base_path / subdir
            if dir_path.exists():
                stats[subdir] = len(list(dir_path.glob("*.json")))
            else:
                stats[subdir] = 0

        return stats


# Convenience functions to maintain API compatibility
def save_blog_post(data: dict[str, Any]) -> str:
    """Save blog post data"""
    storage = FileStorage()
    return storage.save_blog_post(data)


def save_trending_topics(topics: list[dict[str, Any]]) -> str:
    """Save trending topics data"""
    storage = FileStorage()
    return storage.save_trending_topics(topics)


def save_content_generation(data: dict[str, Any]) -> str:
    """Save content generation data"""
    storage = FileStorage()
    return storage.save_content_generation(data)


def export_all_data() -> dict[str, str]:
    """Export all data to CSV files"""
    storage = FileStorage()
    exports = {}

    try:
        exports["blog_posts"] = storage.export_to_csv("blog_posts")
        print(f"✅ Exported blog posts to: {exports['blog_posts']}")
    except Exception as e:
        print(f"❌ Failed to export blog posts: {e}")

    try:
        exports["trending_topics"] = storage.export_to_csv("trending_topics")
        print(f"✅ Exported trending topics to: {exports['trending_topics']}")
    except Exception as e:
        print(f"❌ Failed to export trending topics: {e}")

    return exports
