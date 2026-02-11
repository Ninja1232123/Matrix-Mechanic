"""Configuration for Speed-Guardian"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ProfileConfig:
    """Profiling configuration"""
    enable_line_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_async_profiling: bool = True
    profile_depth: int = 10  # Max call stack depth
    min_time_ms: float = 1.0  # Minimum time to report
    sample_rate: int = 100  # Samples per second for sampling profilers


@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    enable_auto_fix: bool = False  # Auto-apply optimizations
    min_confidence: float = 0.7  # Minimum confidence to suggest
    min_speedup: float = 1.2  # Minimum estimated speedup to suggest
    optimization_types: List[str] = field(default_factory=lambda: [
        "caching",
        "loop_comprehension",
        "algorithm",
        "database_query",
        "io_buffering",
    ])
    max_optimizations: int = 50  # Max optimizations to apply in one run


@dataclass
class RegressionConfig:
    """Regression detection configuration"""
    enable: bool = True
    commits_to_check: int = 10  # Number of recent commits to check
    slowdown_threshold: float = 1.5  # Report if >1.5x slower
    store_benchmarks: bool = True
    benchmark_db_path: str = ".speed_guardian/benchmarks.db"


@dataclass
class UIConfig:
    """UI configuration"""
    theme: str = "dark"
    show_flamegraph: bool = True
    show_memory: bool = True
    refresh_rate: int = 1  # Seconds
    max_results: int = 50


@dataclass
class SpeedGuardianConfig:
    """Main Speed-Guardian configuration"""
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Global settings
    cache_dir: str = ".speed_guardian"
    log_level: str = "INFO"
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/tests/*",
        "*/test_*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/.venv/*",
    ])

    @classmethod
    def from_file(cls, path: Path) -> "SpeedGuardianConfig":
        """Load configuration from YAML file"""
        if not path.exists():
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            profile=ProfileConfig(**data.get('profile', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            regression=RegressionConfig(**data.get('regression', {})),
            ui=UIConfig(**data.get('ui', {})),
            cache_dir=data.get('cache_dir', '.speed_guardian'),
            log_level=data.get('log_level', 'INFO'),
            exclude_patterns=data.get('exclude_patterns', []),
        )

    def to_file(self, path: Path) -> None:
        """Save configuration to YAML file"""
        data = {
            'profile': {
                'enable_line_profiling': self.profile.enable_line_profiling,
                'enable_memory_profiling': self.profile.enable_memory_profiling,
                'enable_async_profiling': self.profile.enable_async_profiling,
                'profile_depth': self.profile.profile_depth,
                'min_time_ms': self.profile.min_time_ms,
                'sample_rate': self.profile.sample_rate,
            },
            'optimization': {
                'enable_auto_fix': self.optimization.enable_auto_fix,
                'min_confidence': self.optimization.min_confidence,
                'min_speedup': self.optimization.min_speedup,
                'optimization_types': self.optimization.optimization_types,
                'max_optimizations': self.optimization.max_optimizations,
            },
            'regression': {
                'enable': self.regression.enable,
                'commits_to_check': self.regression.commits_to_check,
                'slowdown_threshold': self.regression.slowdown_threshold,
                'store_benchmarks': self.regression.store_benchmarks,
                'benchmark_db_path': self.regression.benchmark_db_path,
            },
            'ui': {
                'theme': self.ui.theme,
                'show_flamegraph': self.ui.show_flamegraph,
                'show_memory': self.ui.show_memory,
                'refresh_rate': self.ui.refresh_rate,
                'max_results': self.ui.max_results,
            },
            'cache_dir': self.cache_dir,
            'log_level': self.log_level,
            'exclude_patterns': self.exclude_patterns,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration
DEFAULT_CONFIG = SpeedGuardianConfig()
