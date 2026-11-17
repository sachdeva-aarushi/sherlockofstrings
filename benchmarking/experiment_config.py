# Experiment configuration schema for benchmark sweeps.

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from typing import List, Optional

from generators.input_generator import InputDistribution


@dataclass
class ExperimentConfig:
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "empirical_comparison"
    distributions: List[InputDistribution] = field(
        default_factory=lambda: [InputDistribution.RANDOM]
    )
    text_lengths: List[int] = field(default_factory=lambda: [500, 1000, 2500, 5000])
    pattern_lengths: List[int] = field(default_factory=lambda: [10, 25, 50])
    alphabet_sizes: List[int] = field(default_factory=lambda: [26])
    algorithms: List[str] = field(
        default_factory=lambda: ["Naive", "KMP", "Boyer-Moore"]
    )
    repetitions: int = 30
    warmup_runs: int = 3
    seed: Optional[int] = 42
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
