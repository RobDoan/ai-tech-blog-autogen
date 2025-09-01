import asyncio
from services.topic_discovery.trend_spotter import TrendSpotter
from src.autogen_blog.database import init_db, close_db

async def main():
    """Example usage of the TrendSpotter module with database integration"""
    print("Welcome to the Automated Blog Project!")
    print("Initializing database...")
    
    # Initialize database
    await init_db()
    print("✅ Database initialized successfully")
    
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

    else:
        print("❌ Failed to identify any trends")
    
    # Close database connection
    await close_db()
    print("✅ Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
