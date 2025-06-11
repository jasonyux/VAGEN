from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from .prompt import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
)

@dataclass
class OSWorldEnvConfig(BaseEnvConfig):
    env_name: str = "osworld"
    observation_type: str = "screenshot"
    action_space: str = "pyautogui"
    platform: str = "ubuntu"
    a11y_tree_max_tokens: int = 10000
    tmp_save_path: str = ""
    max_steps: int = 100
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
        ]
        config_id_data = []
        task_config = self.task_config
        domain = task_config['snapshot']
        task_id = task_config['id']
        config_id_data.append(f"domain={domain}")
        # config_id_data.append(f"task_id={task_id}")
        config_id_data.extend([
            f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields
        ])
        id_str = ",".join(config_id_data)
        return f"OSWorldEnvConfig({id_str})"
    
    def get_system_prompt(self):
        if self.system_prompt_type == "default":
            return SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
        else:
            raise ValueError(f"Invalid system prompt type: {self.system_prompt_type}")



if __name__ == "__main__":
    config = OSWorldEnvConfig()
    print(config.config_id())
   