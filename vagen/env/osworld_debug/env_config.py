from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields


@dataclass
class OSWorldDebugEnvConfig(BaseEnvConfig):
    env_name: str = "osworld_debug"
    observation_type: str = "screenshot"
    action_space: str = "computer_13"
    platform: str = "ubuntu"
    a11y_tree_max_tokens: int = 10000
    prerender_trajectory_fpath: str = ""
    tmp_save_path: str = ""
    max_steps: int = 100
    always_zero_reward: bool = False
    task_config: dict = {}
    
    def config_id(self) -> str:
        id_fields = [
            "observation_type",
            "platform",
            "a11y_tree_max_tokens",
            "max_steps",
            "always_zero_reward",
            "task_config",
        ]
        id_str = ",".join([
            f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields
        ])
        return f"OSWorldDebugEnvConfig({id_str})"



if __name__ == "__main__":
    config = OSWorldDebugEnvConfig()
    print(config.config_id())
   