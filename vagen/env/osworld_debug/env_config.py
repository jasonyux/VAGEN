from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
import uuid
from .prompt import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
)

@dataclass
class OSWorldDebugEnvConfig(BaseEnvConfig):
    env_name: str = "osworld_debug"
    observation_type: str = "screenshot"
    action_space: str = "pyautogui"
    platform: str = "ubuntu"
    a11y_tree_max_tokens: int = 10000
    prerender_trajectory_fpath: str = "data/osworld-debug/fixed_obs_ood-correct.pkl.xz"
    tmp_save_path: str = ""
    max_steps: int = 100
    always_zero_reward: bool = False
    task_config: dict = field(default_factory=dict)
    system_prompt_type: str = "default"
   
    def __post_init__(self):
        if self.system_prompt_type == "default":
            assert self.action_space == "pyautogui", "default system prompt only supports pyautogui action space"
            assert self.observation_type == "screenshot", "default system prompt only supports screenshot observation type"
        else:
            raise ValueError(f"Invalid system prompt type: {self.system_prompt_type}")
        return
    
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
        uuid_str = str(uuid.uuid4())
        return f"OSWorldDebugEnvConfig({id_str})({uuid_str})"
    
    def get_system_prompt(self):
        if self.system_prompt_type == "default":
            return SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
        else:
            raise ValueError(f"Invalid system prompt type: {self.system_prompt_type}")



if __name__ == "__main__":
    config = OSWorldDebugEnvConfig()
    print(config.config_id())
   