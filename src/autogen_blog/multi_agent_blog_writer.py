"""
Main script interface for the Multi-Agent Blog Writer system.

This script provides a command-line interface for generating blog posts using
the coordinated multi-agent workflow.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Load environment configuration first
sys.path.insert(0, str(Path(__file__).parent.parent))
from py_env import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, AGENT_TIMEOUT,
    MAX_ITERATIONS, ENABLE_CODE_AGENT, ENABLE_SEO_AGENT, OUTPUT_FORMAT, 
    SAVE_CONVERSATION_LOG, QUALITY_THRESHOLD, validate_config
)

from .blog_writer_orchestrator import BlogWriterOrchestrator
from .multi_agent_models import (
    AgentConfig,
    WorkflowConfig,
    BlogInput,
    TargetAudience,
    ConfigurationError
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)


def load_config() -> tuple[AgentConfig, WorkflowConfig]:
    """Load configuration from centralized environment configuration."""
    # Validate that OpenAI API key is available
    if not OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in your .env file: OPENAI_API_KEY='your-api-key-here'"
        )
    
    # Agent configuration using centralized config values
    agent_config = AgentConfig(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY,
        timeout_seconds=AGENT_TIMEOUT
    )
    
    # Workflow configuration using centralized config values
    workflow_config = WorkflowConfig(
        max_iterations=MAX_ITERATIONS,
        enable_code_agent=ENABLE_CODE_AGENT,
        enable_seo_agent=ENABLE_SEO_AGENT,
        output_format=OUTPUT_FORMAT,
        save_conversation_log=SAVE_CONVERSATION_LOG,
        quality_threshold=QUALITY_THRESHOLD
    )
    
    return agent_config, workflow_config


async def generate_blog_post(
    topic: str,
    description: Optional[str] = None,
    book_reference: Optional[str] = None,
    target_audience: str = "intermediate",
    preferred_length: int = 1500,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Generate a blog post using the multi-agent system.
    
    Args:
        topic: Main topic for the blog post
        description: Optional additional context
        book_reference: Optional reference material
        target_audience: Target audience level
        preferred_length: Preferred word count
        output_file: Optional output file path
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        agent_config, workflow_config = load_config()
        
        logger.info(f"Starting blog generation for topic: {topic}")
        logger.info(f"Target audience: {target_audience}, Length: {preferred_length} words")
        
        # Create orchestrator
        orchestrator = BlogWriterOrchestrator(agent_config, workflow_config)
        
        # Generate blog post
        result = await orchestrator.generate_blog(
            topic=topic,
            description=description,
            book_reference=book_reference
        )
        
        if result.success:
            logger.info(f"Blog generation completed successfully!")
            logger.info(f"Generated {result.metadata.get('word_count', 'unknown')} words")
            logger.info(f"Generation time: {result.generation_time_seconds:.2f} seconds")
            
            # Save or display the result
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.content)
                
                logger.info(f"Blog post saved to: {output_path}")
                
                # Save metadata and conversation log if enabled
                if workflow_config.save_conversation_log:
                    metadata_path = output_path.with_suffix('.json')
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'metadata': result.metadata,
                            'conversation_log': [msg.model_dump() for msg in result.generation_log],
                            'success': result.success,
                            'generation_time_seconds': result.generation_time_seconds
                        }, f, indent=2, default=str)
                    
                    logger.info(f"Metadata and conversation log saved to: {metadata_path}")
            
            else:
                # Display to stdout
                print("\\n" + "="*60)
                print("GENERATED BLOG POST")
                print("="*60)
                print(result.content)
                print("="*60)
                
                # Display metadata
                print("\\nMETADATA:")
                for key, value in result.metadata.items():
                    print(f"  {key}: {value}")
        
        else:
            logger.error(f"Blog generation failed: {result.error_message}")
            
            if result.content:  # Partial content available
                logger.info("Partial content was generated:")
                print("\\n" + "="*60)
                print("PARTIAL CONTENT")
                print("="*60)
                print(result.content)
                print("="*60)
            
            sys.exit(1)
    
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate blog posts using multi-agent AI collaboration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a basic blog post
  python -m src.autogen_blog.multi_agent_blog_writer "Introduction to FastAPI"
  
  # Generate with additional context
  python -m src.autogen_blog.multi_agent_blog_writer "Docker Best Practices" \\
    --description "Focus on security and performance" \\
    --audience advanced \\
    --length 2000 \\
    --output blog_post.md
  
  # Generate with book reference
  python -m src.autogen_blog.multi_agent_blog_writer "Python Async Programming" \\
    --book-reference "Fluent Python by Luciano Ramalho" \\
    --output async_python.md

Environment Variables:
  OPENAI_API_KEY       OpenAI API key (required)
  OPENAI_MODEL         Model to use (default: gpt-4)
  OPENAI_TEMPERATURE   Temperature setting (default: 0.7)
  MAX_ITERATIONS       Max refinement iterations (default: 3)
  ENABLE_CODE_AGENT    Enable code examples (default: true)
  ENABLE_SEO_AGENT     Enable SEO optimization (default: true)
  QUALITY_THRESHOLD    Minimum quality score (default: 7.0)
        """
    )
    
    # Required arguments
    parser.add_argument(
        "topic",
        help="Main topic for the blog post"
    )
    
    # Optional arguments
    parser.add_argument(
        "-d", "--description",
        help="Additional context or description for the blog post"
    )
    
    parser.add_argument(
        "-b", "--book-reference",
        help="Reference book or source material"
    )
    
    parser.add_argument(
        "-a", "--audience",
        choices=["beginner", "intermediate", "advanced", "expert"],
        default="intermediate",
        help="Target audience level (default: intermediate)"
    )
    
    parser.add_argument(
        "-l", "--length",
        type=int,
        default=1500,
        help="Preferred word count (default: 1500)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: display to stdout)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Check configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Configuration check
    if args.config_check:
        setup_logging(True)
        logger = logging.getLogger(__name__)
        
        try:
            # Validate environment configuration
            config_status = validate_config()
            
            print("🔍 Configuration Check Results:")
            print("=" * 40)
            
            for component, is_valid in config_status.items():
                status_icon = "✅" if is_valid else "❌"
                print(f"  {status_icon} {component.replace('_', ' ').title()}: {'OK' if is_valid else 'Missing/Invalid'}")
            
            print()
            
            # Load and display specific configuration
            agent_config, workflow_config = load_config()
            logger.info("Configuration check passed!")
            logger.info(f"Model: {agent_config.model}")
            logger.info(f"Temperature: {agent_config.temperature}")
            logger.info(f"Max tokens: {agent_config.max_tokens}")
            logger.info(f"Max iterations: {workflow_config.max_iterations}")
            logger.info(f"Code agent enabled: {workflow_config.enable_code_agent}")
            logger.info(f"SEO agent enabled: {workflow_config.enable_seo_agent}")
            logger.info(f"Quality threshold: {workflow_config.quality_threshold}")
            
            if all(config_status.values()):
                print("✅ All configuration is valid!")
            else:
                print("⚠️  Some configuration is missing but basic functionality will work")
                
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        return
    
    # Validate arguments
    if args.length < 300:
        print("Error: Minimum word count is 300")
        sys.exit(1)
    
    if args.length > 5000:
        print("Error: Maximum word count is 5000")
        sys.exit(1)
    
    # Run the blog generation
    asyncio.run(generate_blog_post(
        topic=args.topic,
        description=args.description,
        book_reference=args.book_reference,
        target_audience=args.audience,
        preferred_length=args.length,
        output_file=args.output,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()