from tortoise.models import Model
from tortoise import fields
from datetime import datetime


class BlogPost(Model):
    id = fields.IntField(pk=True)
    title = fields.CharField(max_length=255)
    content = fields.TextField()
    summary = fields.TextField(null=True)
    tags = fields.JSONField(default=list)
    trending_topics = fields.JSONField(default=list)
    status = fields.CharField(max_length=20, default="draft")  # draft, published, archived
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    published_at = fields.DatetimeField(null=True)
    
    class Meta:
        table = "blog_posts"
        
    def __str__(self):
        return self.title


class TrendingTopic(Model):
    id = fields.IntField(pk=True)
    topic = fields.CharField(max_length=255)
    source = fields.CharField(max_length=100)  # reddit, twitter, news, etc.
    score = fields.FloatField(default=0.0)
    metadata = fields.JSONField(default=dict)
    discovered_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "trending_topics"
        
    def __str__(self):
        return f"{self.topic} ({self.source})"


class ContentGeneration(Model):
    id = fields.IntField(pk=True)
    blog_post = fields.ForeignKeyField("models.BlogPost", related_name="generations")
    agent_name = fields.CharField(max_length=100)
    generation_type = fields.CharField(max_length=50)  # content, title, summary, etc.
    input_data = fields.JSONField(default=dict)
    output_data = fields.JSONField(default=dict)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "content_generations"