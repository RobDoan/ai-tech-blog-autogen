from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "blog_posts" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "title" VARCHAR(255) NOT NULL,
    "content" TEXT NOT NULL,
    "summary" TEXT,
    "tags" JSON NOT NULL,
    "trending_topics" JSON NOT NULL,
    "status" VARCHAR(20) NOT NULL DEFAULT 'draft',
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "published_at" TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "content_generations" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "agent_name" VARCHAR(100) NOT NULL,
    "generation_type" VARCHAR(50) NOT NULL,
    "input_data" JSON NOT NULL,
    "output_data" JSON NOT NULL,
    "created_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "blog_post_id" INT NOT NULL REFERENCES "blog_posts" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "trending_topics" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "topic" VARCHAR(255) NOT NULL,
    "source" VARCHAR(100) NOT NULL,
    "score" REAL NOT NULL DEFAULT 0,
    "metadata" JSON NOT NULL,
    "discovered_at" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSON NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
