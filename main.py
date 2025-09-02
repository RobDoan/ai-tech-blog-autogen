import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.topic_discovery.trend_spotter import TrendSpotter
from autogen_blog.file_storage import FileStorage, save_trending_topics

async def main():
    """Example usage of the TrendSpotter module with file export"""
    print("Welcome to the Automated Blog Project!")
    print("Initializing file storage system...")
    
    # Initialize file storage
    storage = FileStorage()
    print("✅ File storage initialized successfully")
    
    print("Testing Trend Identification Module...\n")

    # Initialize trend spotter
    spotter = TrendSpotter()

    # Get weekly trend
    trend_result = spotter.get_weekly_trend()

    if trend_result:
        print("🔥 Weekly Trend Identified:")
        print(f"Topic: {trend_result['topic']}")
        print(f"Score: {trend_result['score']:.3f}")
        print(f"Metrics:")
        print(f"  - Search Rank: {trend_result['metrics']['search_rank']}")
        print(f"  - Article Count: {trend_result['metrics']['article_count']}")
        print(f"  - Tweet Volume: {trend_result['metrics']['tweet_volume']}")
        print(f"Timestamp: {trend_result['timestamp']}")

        if trend_result.get('fallback'):
            print("\n⚠️  Note: This is a fallback result (API keys not configured)")
        
        # Save trend to file
        trend_file = save_trending_topics([trend_result])
        print(f"✅ Trend data saved to: {trend_file}")

    else:
        print("❌ Failed to identify any trends")
    
    # Display storage statistics
    stats = storage.get_storage_stats()
    print(f"\n📊 Storage Statistics:")
    print(f"  - Blog Posts: {stats['blog_posts']} files")
    print(f"  - Trending Topics: {stats['trending_topics']} files")
    print(f"  - Content Generations: {stats['content_generations']} files")

if __name__ == "__main__":
    asyncio.run(main())
