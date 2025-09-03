"""
Main script interface for the Multi-Agent Blog Writer system.

This script provides a command-line interface for generating blog posts using
the coordinated multi-agent workflow.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Load environment configuration first
sys.path.insert(0, str(Path(__file__).parent.parent))
from py_env import (
    AGENT_TIMEOUT,
    ENABLE_CODE_AGENT,
    ENABLE_SEO_AGENT,
    MAX_ITERATIONS,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OUTPUT_FORMAT,
    QUALITY_THRESHOLD,
    SAVE_CONVERSATION_LOG,
    validate_config,
)

from .blog_writer_orchestrator import BlogWriterOrchestrator
from .conversational_writer_agent import ConversationalWriterAgent
from .information_synthesizer import InformationSynthesizer
from .multi_agent_models import (
    AgentConfig,
    BlogInput,
    ConfigurationError,
    TargetAudience,
    WorkflowConfig,
)
from .persona_system import (
    PersonaManager,
    create_default_persona_config,
    load_persona_config_from_file,
)

# Import conversational components
from .research_processor import ResearchProcessor


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
    description: str | None = None,
    book_reference: str | None = None,
    target_audience: str = "intermediate",
    preferred_length: int = 1500,
    output_file: str | None = None,
    verbose: bool = False,
    research_folder: str | None = None,
    conversational_mode: bool = False,
    persona_config: str | None = None
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
        research_folder: Path to folder containing research materials
        conversational_mode: Generate content in conversational format
        persona_config: Path to persona configuration JSON file
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        agent_config, workflow_config = load_config()

        logger.info(f"Starting blog generation for topic: {topic}")
        logger.info(f"Target audience: {target_audience}, Length: {preferred_length} words")
        if conversational_mode:
            logger.info("Mode: Conversational format enabled")
        if research_folder:
            logger.info(f"Research folder: {research_folder}")

        # Process research materials if provided
        research_knowledge = None
        if research_folder:
            research_path = Path(research_folder)
            if not research_path.exists():
                logger.error(f"Research folder does not exist: {research_folder}")
                sys.exit(1)

            logger.info("Processing research materials...")
            processor = ResearchProcessor()
            knowledge_base = await processor.process_folder(research_path)

            # Get research files for synthesis
            files = processor._find_supported_files(research_path, recursive=True)
            research_files = await processor._process_files_concurrently(files)

            # Synthesize knowledge
            synthesizer = InformationSynthesizer()
            research_knowledge = await synthesizer.synthesize_knowledge(knowledge_base, research_files)

            logger.info(f"Processed {len(knowledge_base.insights)} insights from research")

        # Generate blog post
        if conversational_mode:
            # Load persona configuration
            persona_config_obj = None
            if persona_config:
                persona_config_path = Path(persona_config)
                if not persona_config_path.exists():
                    logger.error(f"Persona config file does not exist: {persona_config}")
                    sys.exit(1)
                persona_config_obj = load_persona_config_from_file(persona_config_path)
            else:
                persona_config_obj = create_default_persona_config()

            # Use conversational writer
            writer_agent = ConversationalWriterAgent(agent_config)

            # Create blog input
            blog_input = BlogInput(
                topic=topic,
                description=description,
                book_reference=book_reference,
                target_audience=TargetAudience(target_audience),
                preferred_length=preferred_length
            )

            # Generate content outline (we'll need to adapt this)
            orchestrator = BlogWriterOrchestrator(agent_config, workflow_config)
            planner = orchestrator.content_planner
            outline = await planner.create_outline(blog_input)

            # Create personas
            persona_manager = PersonaManager()
            personas = persona_manager.create_personas(persona_config_obj)

            # Generate conversational content
            result_content = await writer_agent.write_conversational_content(
                outline, blog_input, research_knowledge, personas, persona_config_obj
            )

            # Create result object compatible with existing interface
            from .multi_agent_models import BlogResult
            result = BlogResult(
                content=result_content.content,
                metadata={
                    'word_count': result_content.metadata.word_count,
                    'reading_time_minutes': result_content.metadata.reading_time_minutes,
                    'conversation_flow_score': result_content.conversation_flow_score,
                    'synthesis_confidence': result_content.synthesis_confidence,
                    'personas_used': result_content.personas_used,
                    'research_sources_count': len(result_content.research_sources)
                },
                generation_log=[],  # Empty for now
                success=True,
                generation_time_seconds=0.0
            )

        else:
            # Use traditional orchestrator
            orchestrator = BlogWriterOrchestrator(agent_config, workflow_config)
            result = await orchestrator.generate_blog(
                topic=topic,
                description=description,
                book_reference=book_reference
            )

        if result.success:
            logger.info("Blog generation completed successfully!")
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
  
  # Generate conversational blog post
  python -m src.autogen_blog.multi_agent_blog_writer "React State Management" \\
    --conversational \\
    --output react_conversation.md
  
  # Generate with research materials
  python -m src.autogen_blog.multi_agent_blog_writer "Machine Learning Best Practices" \\
    --research-folder ./research_docs \\
    --conversational \\
    --output ml_conversation.md
  
  # Generate with custom personas
  python -m src.autogen_blog.multi_agent_blog_writer "DevOps Automation" \\
    --conversational \\
    --persona-config ./personas.json \\
    --research-folder ./devops_research \\
    --output devops_dialogue.md

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
        "-r", "--research-folder",
        help="Path to folder containing research materials (MD, TXT, JSON files)"
    )

    parser.add_argument(
        "-c", "--conversational",
        action="store_true",
        help="Generate content in conversational dialogue format"
    )

    parser.add_argument(
        "-p", "--persona-config",
        help="Path to JSON file containing persona configuration"
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

    # Validate conversational mode arguments
    if args.conversational and args.persona_config:
        persona_path = Path(args.persona_config)
        if not persona_path.exists():
            print(f"Error: Persona config file does not exist: {args.persona_config}")
            sys.exit(1)

    if args.research_folder:
        research_path = Path(args.research_folder)
        if not research_path.exists():
            print(f"Error: Research folder does not exist: {args.research_folder}")
            sys.exit(1)

    # Run the blog generation
    asyncio.run(generate_blog_post(
        topic=args.topic,
        description=args.description,
        book_reference=args.book_reference,
        target_audience=args.audience,
        preferred_length=args.length,
        output_file=args.output,
        verbose=args.verbose,
        research_folder=args.research_folder,
        conversational_mode=args.conversational,
        persona_config=args.persona_config
    ))


if __name__ == "__main__":
    main()
