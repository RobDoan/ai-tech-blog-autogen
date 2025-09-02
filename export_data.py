#!/usr/bin/env python3
"""
Utility script to export all stored data to CSV files.
Usage: python export_data.py
"""
import sys
import os
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_blog.file_storage import FileStorage, export_all_data


def main():
    """Export all data to CSV files"""
    print("🚀 Starting data export process...")

    # Check if data directory exists
    storage = FileStorage()
    stats = storage.get_storage_stats()

    print(f"\n📊 Current Storage Statistics:")
    print(f"  - Blog Posts: {stats['blog_posts']} files")
    print(f"  - Trending Topics: {stats['trending_topics']} files")
    print(f"  - Content Generations: {stats['content_generations']} files")

    if sum(stats.values()) == 0:
        print("\n⚠️  No data found to export. Run the main application first to generate some data.")
        return

    print("\n📤 Exporting data to CSV files...")
    exports = export_all_data()

    print(f"\n✅ Export completed!")
    print(f"Exported files:")
    for data_type, file_path in exports.items():
        print(f"  - {data_type}: {file_path}")

    print(f"\nYou can find all your data in the './data' directory.")


if __name__ == "__main__":
    main()