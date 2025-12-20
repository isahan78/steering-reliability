"""
Configuration management for steering reliability experiments.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "EleutherAI/pythia-160m"
    device: str = "auto"  # auto, cuda, cpu, mps
    dtype: str = "float32"  # float32, float16, bfloat16


@dataclass
class DataConfig:
    """Data configuration."""
    harm_train_path: str = "data/prompts/harm_train.jsonl"
    harm_test_path: str = "data/prompts/harm_test.jsonl"
    benign_path: str = "data/prompts/benign.jsonl"
    max_prompts_per_split: Optional[int] = None


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 0
    batch_size: int = 8
    num_samples_per_prompt: int = 1


@dataclass
class DirectionConfig:
    """Steering direction configuration."""
    method: str = "contrastive_prefix"
    refusal_prefix: str = " I can't help with that."
    compliance_prefix: str = " Sure, here's how:"
    hook_point: str = "resid_post"  # resid_post or resid_pre
    token_position: str = "last_prompt_token"
    normalize: bool = True


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    layers: List[int] = field(default_factory=lambda: [4, 8, 12])
    interventions: List[str] = field(default_factory=lambda: ["add", "ablate"])
    alphas: List[float] = field(default_factory=lambda: [0, 0.5, 1, 2, 4])
    output_dir: str = "artifacts/runs/run_001"
    save_completions: bool = True


@dataclass
class SanityCheckConfig:
    """Sanity check configuration."""
    enabled: bool = True
    alt_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "refusal_prefix": " Sorry, I can't do that.",
        "compliance_prefix": " Of course. Here's a step-by-step guide:"
    })
    rerun_subset_n: int = 30
    alt_seed: int = 42


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    refusal: Dict[str, float] = field(default_factory=lambda: {"threshold": 0.5})
    helpfulness: Dict[str, float] = field(default_factory=lambda: {"threshold": 0.5})


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    sanity_checks: SanityCheckConfig = field(default_factory=SanityCheckConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            direction=DirectionConfig(**config_dict.get("direction", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            sanity_checks=SanityCheckConfig(**config_dict.get("sanity_checks", {})),
            metrics=MetricsConfig(**config_dict.get("metrics", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "generation": asdict(self.generation),
            "direction": asdict(self.direction),
            "experiment": asdict(self.experiment),
            "sanity_checks": asdict(self.sanity_checks),
            "metrics": asdict(self.metrics),
        }

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML. If None, uses default config.

    Returns:
        Config object
    """
    if config_path is None:
        config_path = "configs/default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config.from_dict(config_dict)


def create_config(**kwargs) -> Config:
    """
    Create a Config programmatically (useful for notebooks).

    Example:
        config = create_config(
            model={"name": "gpt2", "device": "cuda"},
            generation={"max_new_tokens": 100}
        )

    Args:
        **kwargs: Nested dictionaries for each config section

    Returns:
        Config object
    """
    config_dict = {}
    for key, value in kwargs.items():
        if isinstance(value, dict):
            config_dict[key] = value

    return Config.from_dict(config_dict)
