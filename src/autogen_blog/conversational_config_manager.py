"""
Configuration Management for Conversational Blog Writer.

This module provides comprehensive configuration management for personas,
conversation styles, and conversational writing preferences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .persona_system import (
    PersonaConfig, PersonaProfile, ConversationStyle,
    CommunicationStyle, FormalityLevel, TechnicalDepth, DialoguePace,
    create_default_persona_config, save_persona_config_to_file, load_persona_config_from_file
)
from .multi_agent_models import BlogGenerationError


class ConfigurationError(BlogGenerationError):
    """Raised when configuration operations fail."""
    pass


class PresetType(str, Enum):
    """Types of configuration presets."""
    BEGINNER_FRIENDLY = "beginner_friendly"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    CASUAL_CONVERSATION = "casual_conversation"
    FORMAL_PRESENTATION = "formal_presentation"
    TUTORIAL_STYLE = "tutorial_style"
    INTERVIEW_STYLE = "interview_style"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class ConversationalPreferences:
    """User preferences for conversational content generation."""
    preferred_conversation_length: int = 12  # Number of exchanges
    max_exchanges_per_section: int = 6
    include_code_examples: bool = True
    include_practical_scenarios: bool = True
    allow_interruptions: bool = False
    emphasize_best_practices: bool = True
    include_common_pitfalls: bool = True
    use_real_world_examples: bool = True
    
    # Quality preferences
    minimum_naturalness_score: float = 0.7
    minimum_technical_accuracy: float = 0.8
    minimum_persona_consistency: float = 0.75
    
    # Content preferences
    preferred_code_languages: List[str] = None
    avoid_advanced_concepts: bool = False
    include_performance_discussions: bool = True
    include_security_considerations: bool = False
    
    def __post_init__(self):
        if self.preferred_code_languages is None:
            self.preferred_code_languages = ["python", "javascript", "typescript"]


@dataclass
class DomainConfiguration:
    """Domain-specific configuration for conversations."""
    domain_name: str
    typical_technologies: List[str]
    common_challenges: List[str]
    key_concepts: List[str]
    recommended_personas: Dict[str, Dict[str, Any]]
    conversation_patterns: List[str]
    
    # Domain-specific preferences
    technical_depth_preference: TechnicalDepth = TechnicalDepth.INTERMEDIATE
    formality_preference: FormalityLevel = FormalityLevel.PROFESSIONAL
    pace_preference: DialoguePace = DialoguePace.MODERATE


class ConversationalConfigManager:
    """Manages configuration for conversational blog generation."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Default config directory
        if config_dir is None:
            config_dir = Path.home() / ".conversational_blog_writer"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.personas_dir = self.config_dir / "personas"
        self.presets_dir = self.config_dir / "presets"
        self.domains_dir = self.config_dir / "domains"
        
        # Ensure subdirectories exist
        self.personas_dir.mkdir(exist_ok=True)
        self.presets_dir.mkdir(exist_ok=True)
        self.domains_dir.mkdir(exist_ok=True)
        
        # Initialize default configurations if they don't exist
        self._initialize_default_configs()
        
        self.logger.info(f"Configuration manager initialized at {self.config_dir}")
    
    def _initialize_default_configs(self):
        """Initialize default configurations."""
        try:
            # Create default persona configurations
            self._create_default_persona_configs()
            
            # Create preset configurations
            self._create_preset_configurations()
            
            # Create domain configurations
            self._create_domain_configurations()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default configs: {e}")
    
    def _create_default_persona_configs(self):
        """Create default persona configurations."""
        default_configs = {
            "default": create_default_persona_config(),
            "beginner_friendly": self._create_beginner_friendly_personas(),
            "expert_level": self._create_expert_level_personas(),
            "casual_chat": self._create_casual_chat_personas(),
            "formal_presentation": self._create_formal_presentation_personas()
        }
        
        for name, config in default_configs.items():
            config_file = self.personas_dir / f"{name}.json"
            if not config_file.exists():
                try:
                    save_persona_config_to_file(config, config_file)
                    self.logger.info(f"Created default persona config: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to create {name} persona config: {e}")
    
    def _create_beginner_friendly_personas(self) -> PersonaConfig:
        """Create beginner-friendly persona configuration."""
        problem_presenter = PersonaProfile(
            name="Sam",
            role="problem_presenter",
            background="Junior developer with 6 months of experience, eager to learn",
            expertise_areas=["basic programming", "web development fundamentals"],
            communication_style=CommunicationStyle.FRIENDLY,
            personality_traits=["curious", "enthusiastic", "asks clarifying questions"],
            technical_level=TechnicalDepth.BEGINNER,
            conversation_goals=[
                "Ask basic questions about concepts",
                "Request step-by-step explanations",
                "Seek practical examples"
            ],
            typical_phrases=[
                "I'm new to this, but",
                "Could you explain what",
                "I'm not sure I understand",
                "Can you walk me through",
                "What's the difference between"
            ]
        )
        
        solution_provider = PersonaProfile(
            name="Taylor",
            role="solution_provider",
            background="Patient mentor and senior developer who loves teaching",
            expertise_areas=["full-stack development", "mentoring", "best practices"],
            communication_style=CommunicationStyle.FRIENDLY,
            personality_traits=["patient", "encouraging", "uses analogies"],
            technical_level=TechnicalDepth.ADVANCED,
            conversation_goals=[
                "Explain concepts clearly and simply",
                "Use analogies and real-world examples",
                "Encourage learning and experimentation"
            ],
            typical_phrases=[
                "Great question!",
                "Think of it like",
                "Let me break that down",
                "Here's a simple way to",
                "Don't worry, this is common"
            ]
        )
        
        conversation_style = ConversationStyle(
            formality_level=FormalityLevel.CASUAL,
            technical_depth=TechnicalDepth.BEGINNER,
            dialogue_pace=DialoguePace.DETAILED,
            max_exchange_length=4,
            include_questions=True,
            context_awareness=0.9
        )
        
        return PersonaConfig(
            problem_presenter=problem_presenter,
            solution_provider=solution_provider,
            conversation_style=conversation_style,
            domain_focus="web_development",
            target_audience="beginners",
            dialogue_objective="educational"
        )
    
    def _create_expert_level_personas(self) -> PersonaConfig:
        """Create expert-level persona configuration."""
        problem_presenter = PersonaProfile(
            name="Alex",
            role="problem_presenter",
            background="Senior architect dealing with complex system challenges",
            expertise_areas=["system architecture", "performance optimization", "scalability"],
            communication_style=CommunicationStyle.TECHNICAL,
            personality_traits=["analytical", "precise", "performance-focused"],
            technical_level=TechnicalDepth.EXPERT,
            conversation_goals=[
                "Discuss complex technical challenges",
                "Explore trade-offs and alternatives",
                "Focus on scalability and performance"
            ],
            typical_phrases=[
                "In high-throughput systems",
                "We're seeing performance bottlenecks",
                "The trade-off here is",
                "At enterprise scale"
            ]
        )
        
        solution_provider = PersonaProfile(
            name="Jordan",
            role="solution_provider",
            background="Principal engineer with expertise in distributed systems",
            expertise_areas=["distributed systems", "cloud architecture", "DevOps"],
            communication_style=CommunicationStyle.TECHNICAL,
            personality_traits=["systematic", "thorough", "solution-oriented"],
            technical_level=TechnicalDepth.EXPERT,
            conversation_goals=[
                "Provide deep technical insights",
                "Explain complex architectures",
                "Share enterprise-level solutions"
            ],
            typical_phrases=[
                "From an architectural perspective",
                "The key insight is",
                "In production environments",
                "This pattern works well when"
            ]
        )
        
        conversation_style = ConversationStyle(
            formality_level=FormalityLevel.PROFESSIONAL,
            technical_depth=TechnicalDepth.EXPERT,
            dialogue_pace=DialoguePace.MODERATE,
            max_exchange_length=5,
            include_questions=True,
            context_awareness=0.95
        )
        
        return PersonaConfig(
            problem_presenter=problem_presenter,
            solution_provider=solution_provider,
            conversation_style=conversation_style,
            domain_focus="enterprise_architecture",
            target_audience="experts",
            dialogue_objective="deep_technical"
        )
    
    def _create_casual_chat_personas(self) -> PersonaConfig:
        """Create casual conversation persona configuration."""
        problem_presenter = PersonaProfile(
            name="Casey",
            role="problem_presenter",
            background="Freelance developer working on various projects",
            expertise_areas=["full-stack development", "client projects"],
            communication_style=CommunicationStyle.CASUAL,
            personality_traits=["relaxed", "conversational", "practical"],
            technical_level=TechnicalDepth.INTERMEDIATE,
            conversation_goals=[
                "Share real project experiences",
                "Discuss practical solutions",
                "Keep things relatable"
            ],
            typical_phrases=[
                "So I was working on this project",
                "You know what I mean?",
                "That's pretty cool",
                "I've been there"
            ]
        )
        
        solution_provider = PersonaProfile(
            name="Riley",
            role="solution_provider",
            background="Experienced developer who enjoys sharing knowledge casually",
            expertise_areas=["web development", "tooling", "productivity"],
            communication_style=CommunicationStyle.CASUAL,
            personality_traits=["friendly", "approachable", "practical"],
            technical_level=TechnicalDepth.ADVANCED,
            conversation_goals=[
                "Share practical tips and tricks",
                "Keep explanations accessible",
                "Make learning enjoyable"
            ],
            typical_phrases=[
                "Oh yeah, I love that approach",
                "Here's what I usually do",
                "That's a neat trick",
                "Totally, and you can also"
            ]
        )
        
        conversation_style = ConversationStyle(
            formality_level=FormalityLevel.VERY_CASUAL,
            technical_depth=TechnicalDepth.INTERMEDIATE,
            dialogue_pace=DialoguePace.QUICK,
            max_exchange_length=3,
            include_questions=True,
            allow_interruptions=True,
            context_awareness=0.8
        )
        
        return PersonaConfig(
            problem_presenter=problem_presenter,
            solution_provider=solution_provider,
            conversation_style=conversation_style,
            domain_focus="general_development",
            target_audience="developers",
            dialogue_objective="casual_learning"
        )
    
    def _create_formal_presentation_personas(self) -> PersonaConfig:
        """Create formal presentation persona configuration."""
        problem_presenter = PersonaProfile(
            name="Dr. Martinez",
            role="problem_presenter",
            background="Technical lead presenting to stakeholders and management",
            expertise_areas=["project management", "technical leadership"],
            communication_style=CommunicationStyle.FORMAL,
            personality_traits=["structured", "authoritative", "clear"],
            technical_level=TechnicalDepth.ADVANCED,
            conversation_goals=[
                "Present challenges clearly",
                "Focus on business impact",
                "Maintain professional tone"
            ]
        )
        
        solution_provider = PersonaProfile(
            name="Dr. Chen",
            role="solution_provider",
            background="Principal consultant with expertise in enterprise solutions",
            expertise_areas=["enterprise architecture", "consulting", "strategy"],
            communication_style=CommunicationStyle.FORMAL,
            personality_traits=["authoritative", "comprehensive", "strategic"],
            technical_level=TechnicalDepth.EXPERT,
            conversation_goals=[
                "Provide comprehensive solutions",
                "Explain strategic benefits",
                "Maintain professional authority"
            ]
        )
        
        conversation_style = ConversationStyle(
            formality_level=FormalityLevel.FORMAL,
            technical_depth=TechnicalDepth.ADVANCED,
            dialogue_pace=DialoguePace.THOROUGH,
            max_exchange_length=6,
            include_questions=False,
            context_awareness=0.95
        )
        
        return PersonaConfig(
            problem_presenter=problem_presenter,
            solution_provider=solution_provider,
            conversation_style=conversation_style,
            domain_focus="enterprise_solutions",
            target_audience="executives",
            dialogue_objective="formal_presentation"
        )
    
    def _create_preset_configurations(self):
        """Create preset configurations for common use cases."""
        presets = {
            PresetType.BEGINNER_FRIENDLY: {
                "description": "Friendly, patient conversation perfect for beginners",
                "persona_config": "beginner_friendly",
                "preferences": ConversationalPreferences(
                    preferred_conversation_length=10,
                    include_code_examples=True,
                    avoid_advanced_concepts=True,
                    minimum_naturalness_score=0.8,
                    preferred_code_languages=["python", "javascript"]
                )
            },
            
            PresetType.TECHNICAL_DEEP_DIVE: {
                "description": "In-depth technical discussion for experienced developers",
                "persona_config": "expert_level",
                "preferences": ConversationalPreferences(
                    preferred_conversation_length=16,
                    include_code_examples=True,
                    include_performance_discussions=True,
                    include_security_considerations=True,
                    minimum_technical_accuracy=0.9,
                    preferred_code_languages=["python", "java", "go", "rust"]
                )
            },
            
            PresetType.CASUAL_CONVERSATION: {
                "description": "Relaxed, informal conversation between developers",
                "persona_config": "casual_chat",
                "preferences": ConversationalPreferences(
                    preferred_conversation_length=8,
                    allow_interruptions=True,
                    minimum_naturalness_score=0.9,
                    include_practical_scenarios=True
                )
            },
            
            PresetType.TUTORIAL_STYLE: {
                "description": "Step-by-step tutorial conversation",
                "persona_config": "beginner_friendly",
                "preferences": ConversationalPreferences(
                    preferred_conversation_length=12,
                    max_exchanges_per_section=4,
                    include_code_examples=True,
                    emphasize_best_practices=True,
                    use_real_world_examples=True
                )
            }
        }
        
        for preset_type, config in presets.items():
            preset_file = self.presets_dir / f"{preset_type.value}.json"
            if not preset_file.exists():
                try:
                    with open(preset_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, default=str)
                    self.logger.info(f"Created preset configuration: {preset_type.value}")
                except Exception as e:
                    self.logger.error(f"Failed to create preset {preset_type.value}: {e}")
    
    def _create_domain_configurations(self):
        """Create domain-specific configurations."""
        domains = {
            "web_development": DomainConfiguration(
                domain_name="Web Development",
                typical_technologies=["HTML", "CSS", "JavaScript", "React", "Vue", "Angular", "Node.js"],
                common_challenges=["responsive design", "performance optimization", "accessibility", "SEO"],
                key_concepts=["DOM manipulation", "event handling", "API integration", "state management"],
                recommended_personas={"beginner": "beginner_friendly", "advanced": "default"},
                conversation_patterns=["problem-solution", "tutorial", "best-practices"],
                technical_depth_preference=TechnicalDepth.INTERMEDIATE
            ),
            
            "backend_development": DomainConfiguration(
                domain_name="Backend Development",
                typical_technologies=["Python", "Java", "Node.js", "Go", "PostgreSQL", "MongoDB", "Redis"],
                common_challenges=["scalability", "performance", "security", "data consistency"],
                key_concepts=["API design", "database optimization", "caching", "microservices"],
                recommended_personas={"beginner": "default", "expert": "expert_level"},
                conversation_patterns=["architecture-discussion", "problem-solution", "performance-analysis"],
                technical_depth_preference=TechnicalDepth.ADVANCED
            ),
            
            "devops": DomainConfiguration(
                domain_name="DevOps & Infrastructure",
                typical_technologies=["Docker", "Kubernetes", "AWS", "Terraform", "Jenkins", "GitLab CI"],
                common_challenges=["deployment automation", "monitoring", "scaling", "security"],
                key_concepts=["CI/CD", "infrastructure as code", "containerization", "orchestration"],
                recommended_personas={"all_levels": "expert_level"},
                conversation_patterns=["troubleshooting", "best-practices", "tool-comparison"],
                technical_depth_preference=TechnicalDepth.EXPERT
            )
        }
        
        for domain_name, config in domains.items():
            domain_file = self.domains_dir / f"{domain_name}.json"
            if not domain_file.exists():
                try:
                    with open(domain_file, 'w', encoding='utf-8') as f:
                        json.dump(asdict(config), f, indent=2, default=str)
                    self.logger.info(f"Created domain configuration: {domain_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create domain {domain_name}: {e}")
    
    # Configuration Management Methods
    
    def list_persona_configs(self) -> List[str]:
        """List available persona configurations."""
        try:
            configs = [f.stem for f in self.personas_dir.glob("*.json")]
            return sorted(configs)
        except Exception as e:
            self.logger.error(f"Failed to list persona configs: {e}")
            return []
    
    def list_presets(self) -> List[str]:
        """List available preset configurations."""
        try:
            presets = [f.stem for f in self.presets_dir.glob("*.json")]
            return sorted(presets)
        except Exception as e:
            self.logger.error(f"Failed to list presets: {e}")
            return []
    
    def list_domains(self) -> List[str]:
        """List available domain configurations."""
        try:
            domains = [f.stem for f in self.domains_dir.glob("*.json")]
            return sorted(domains)
        except Exception as e:
            self.logger.error(f"Failed to list domains: {e}")
            return []
    
    def get_persona_config(self, name: str) -> PersonaConfig:
        """Get a persona configuration by name."""
        config_file = self.personas_dir / f"{name}.json"
        
        if not config_file.exists():
            available = self.list_persona_configs()
            raise ConfigurationError(
                f"Persona configuration '{name}' not found. Available: {available}"
            )
        
        try:
            return load_persona_config_from_file(config_file)
        except Exception as e:
            raise ConfigurationError(f"Failed to load persona config '{name}': {e}")
    
    def save_persona_config(self, name: str, config: PersonaConfig) -> None:
        """Save a persona configuration."""
        config_file = self.personas_dir / f"{name}.json"
        
        try:
            save_persona_config_to_file(config, config_file)
            self.logger.info(f"Saved persona configuration: {name}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save persona config '{name}': {e}")
    
    def get_preset(self, preset_type: Union[PresetType, str]) -> Dict[str, Any]:
        """Get a preset configuration."""
        if isinstance(preset_type, PresetType):
            preset_name = preset_type.value
        else:
            preset_name = preset_type
        
        preset_file = self.presets_dir / f"{preset_name}.json"
        
        if not preset_file.exists():
            available = self.list_presets()
            raise ConfigurationError(
                f"Preset '{preset_name}' not found. Available: {available}"
            )
        
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load preset '{preset_name}': {e}")
    
    def get_domain_config(self, domain_name: str) -> DomainConfiguration:
        """Get a domain configuration."""
        domain_file = self.domains_dir / f"{domain_name}.json"
        
        if not domain_file.exists():
            available = self.list_domains()
            raise ConfigurationError(
                f"Domain configuration '{domain_name}' not found. Available: {available}"
            )
        
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return DomainConfiguration(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load domain config '{domain_name}': {e}")
    
    def create_custom_config_from_preset(
        self,
        preset_type: Union[PresetType, str],
        custom_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> PersonaConfig:
        """Create a custom configuration based on a preset."""
        try:
            preset = self.get_preset(preset_type)
            base_persona_name = preset["persona_config"]
            
            # Load base persona config
            base_config = self.get_persona_config(base_persona_name)
            
            # Apply modifications if provided
            if modifications:
                # This is a simplified approach - you might want more sophisticated merging
                config_dict = base_config.dict()
                self._deep_update(config_dict, modifications)
                base_config = PersonaConfig(**config_dict)
            
            # Save as custom configuration
            self.save_persona_config(custom_name, base_config)
            
            self.logger.info(f"Created custom configuration '{custom_name}' from preset '{preset_type}'")
            return base_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create custom config from preset: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update a dictionary with another dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def export_configuration(self, output_path: Path, include_defaults: bool = True) -> None:
        """Export all configurations to a single file."""
        try:
            export_data = {
                "personas": {},
                "presets": {},
                "domains": {},
                "export_timestamp": str(datetime.now())
            }
            
            # Export personas
            for persona_name in self.list_persona_configs():
                if include_defaults or not persona_name.startswith("default"):
                    config = self.get_persona_config(persona_name)
                    export_data["personas"][persona_name] = config.dict()
            
            # Export presets
            for preset_name in self.list_presets():
                preset = self.get_preset(preset_name)
                export_data["presets"][preset_name] = preset
            
            # Export domains
            for domain_name in self.list_domains():
                config = self.get_domain_config(domain_name)
                export_data["domains"][domain_name] = asdict(config)
            
            # Save export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to export configuration: {e}")
    
    def import_configuration(self, import_path: Path, overwrite: bool = False) -> None:
        """Import configurations from a file."""
        try:
            if not import_path.exists():
                raise ConfigurationError(f"Import file not found: {import_path}")
            
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import personas
            for persona_name, persona_data in import_data.get("personas", {}).items():
                config_file = self.personas_dir / f"{persona_name}.json"
                if config_file.exists() and not overwrite:
                    self.logger.warning(f"Skipping existing persona config: {persona_name}")
                    continue
                
                config = PersonaConfig(**persona_data)
                self.save_persona_config(persona_name, config)
            
            # Import presets
            for preset_name, preset_data in import_data.get("presets", {}).items():
                preset_file = self.presets_dir / f"{preset_name}.json"
                if preset_file.exists() and not overwrite:
                    self.logger.warning(f"Skipping existing preset: {preset_name}")
                    continue
                
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, indent=2, default=str)
            
            # Import domains
            for domain_name, domain_data in import_data.get("domains", {}).items():
                domain_file = self.domains_dir / f"{domain_name}.json"
                if domain_file.exists() and not overwrite:
                    self.logger.warning(f"Skipping existing domain: {domain_name}")
                    continue
                
                with open(domain_file, 'w', encoding='utf-8') as f:
                    json.dump(domain_data, f, indent=2, default=str)
            
            self.logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration: {e}")
    
    def validate_configuration(self, config: PersonaConfig) -> List[str]:
        """Validate a persona configuration and return issues."""
        issues = []
        
        try:
            # Basic validation
            if not config.problem_presenter.name:
                issues.append("Problem presenter name is required")
            
            if not config.solution_provider.name:
                issues.append("Solution provider name is required")
            
            if config.problem_presenter.name == config.solution_provider.name:
                issues.append("Personas must have different names")
            
            # Role validation
            if config.problem_presenter.role != "problem_presenter":
                issues.append("Problem presenter role must be 'problem_presenter'")
            
            if config.solution_provider.role != "solution_provider":
                issues.append("Solution provider role must be 'solution_provider'")
            
            # Style consistency validation
            if (config.conversation_style.technical_depth == TechnicalDepth.BEGINNER and
                config.problem_presenter.technical_level == TechnicalDepth.EXPERT):
                issues.append("Technical depth mismatch between conversation style and personas")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues


# Utility functions

def get_recommended_config_for_topic(topic: str, target_audience: str = "intermediate") -> str:
    """Get recommended configuration name for a topic and audience."""
    topic_lower = topic.lower()
    
    # Simple heuristic-based recommendations
    if any(word in topic_lower for word in ["beginner", "introduction", "basics", "getting started"]):
        return "beginner_friendly"
    
    elif any(word in topic_lower for word in ["advanced", "expert", "deep dive", "architecture"]):
        return "expert_level"
    
    elif any(word in topic_lower for word in ["chat", "discussion", "talk", "conversation"]):
        return "casual_chat"
    
    elif target_audience == "beginner":
        return "beginner_friendly"
    
    elif target_audience in ["advanced", "expert"]:
        return "expert_level"
    
    else:
        return "default"


from datetime import datetime