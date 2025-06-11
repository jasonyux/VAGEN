from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields, field

@dataclass
class OSWorldServiceConfig(BaseServiceConfig):
    use_state_reward: bool = False
    max_workers: int = 8