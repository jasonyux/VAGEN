import json

from typing import Dict, List, Tuple, Optional, Any, Union
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper
from vagen.server.serial import serialize_observation
from .env import OSWorldDebugEnv, OSWorldDebugEnvConfig
from .service_config import OSWorldDebugServiceConfig


class OSWorldDebugService(BaseService):
    def __init__(self, config: OSWorldDebugServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
        print(f"[DEBUG] OSWorldDebugService init {self.config=}")
        return
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        print(f"[DEBUG] OSWorldDebugService create_environments_batch {ids2configs=}")
        for env_id, config in ids2configs.items():
            env_config_dict = config['env_config']
            if "config_str" in env_config_dict:
                env_config_dict = json.loads(env_config_dict["config_str"])
            env_config = OSWorldDebugEnvConfig(**env_config_dict)
            env = OSWorldDebugEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
        return
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        print(f"[DEBUG] OSWorldDebugService reset_batch {ids2seeds=}")
        results = {}
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        print(f"[DEBUG] OSWorldDebugService step_batch {ids2actions=}")
        results = {}
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        print(f"[DEBUG] OSWorldDebugService compute_reward_batch {env_ids=}")
        results = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()
        print(f"[DEBUG] OSWorldDebugService compute_reward_batch {results=}")
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        print(f"[DEBUG] OSWorldDebugService get_system_prompts_batch {env_ids=}")
        results = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        print(f"[DEBUG] OSWorldDebugService get_system_prompts_batch {results=}")
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        print(f"[DEBUG] OSWorldDebugService close_batch {env_ids=}")
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            env = self.environments[env_id]
            env.close()
            
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
        print(f"[DEBUG] OSWorldDebugService close_batch done {env_ids=}")
        return