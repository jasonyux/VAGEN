import json

from typing import Dict, List, Tuple, Optional, Any, Union
from vagen.env.base.base_service import BaseService
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper
from vagen.server.serial import serialize_observation
from .env import OSWorldEnv, OSWorldEnvConfig
from .service_config import OSWorldServiceConfig
import concurrent.futures


class OSWorldService(BaseService):
    def __init__(self, config: OSWorldServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
        self.max_workers = config.max_workers
        print(f"[DEBUG] OSWorldService init {self.config=}")
        return
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        print(f"[DEBUG] OSWorldService create_environments_batch {ids2configs=}")
        for env_id, config in ids2configs.items():
            env_config_dict = config['env_config']
            if "config_str" in env_config_dict:
                env_config_dict = json.loads(env_config_dict["config_str"])
            env_config = OSWorldEnvConfig(**env_config_dict)
            env = OSWorldEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
        return
    
    def _reset_single_env(self, env_id: str, seed: Any) -> Tuple[Any, Any]:
        env = self.environments[env_id]
        serialized_observation = None
        info = {}
        try:
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
        except Exception as e:
            print(f"[DEBUG] OSWorldService _reset_single_env {env_id=} {e=}")
        return env_id, (serialized_observation, info)
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        print(f"[DEBUG] OSWorldService reset_batch {ids2seeds=}")
        results = {}
        # for env_id, seed in ids2seeds.items():
        #     env = self.environments[env_id]
        #     observation, info = env.reset(seed=seed)
        #     serialized_observation = serialize_observation(observation)
        #     results[env_id] = (serialized_observation, info)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for env_id, seed in ids2seeds.items():
                future = executor.submit(self._reset_single_env, env_id, seed)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                env_id, result = future.result()
                results[env_id] = result
        return results
    
    def _step_single_env(self, env_id: str, action: Any) -> Tuple[Any, Any]:
        env = self.environments[env_id]
        serialized_observation = None
        reward = 0.0
        done = False
        info = {}
        try:
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
        except Exception as e:
            print(f"[DEBUG] OSWorldService _step_single_env {env_id=} {e=}")
        return env_id, (serialized_observation, reward, done, info)
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        print(f"[DEBUG] OSWorldService step_batch {ids2actions=}")
        results = {}
        # for env_id, action in ids2actions.items():
        #     env = self.environments[env_id]
        #     observation, reward, done, info = env.step(action)
        #     serialized_observation = serialize_observation(observation)
        #     results[env_id] = (serialized_observation, reward, done, info)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for env_id, action in ids2actions.items():
                future = executor.submit(self._step_single_env, env_id, action)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                env_id, result = future.result()
                results[env_id] = result
        return results

    def _compute_reward_single_env(self, env_id: str) -> Tuple[Any, Any]:
        env = self.environments[env_id]
        reward = env.compute_reward()
        return env_id, reward
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        print(f"[DEBUG] OSWorldService compute_reward_batch {env_ids=}")
        results = {}
        # for env_id in env_ids:
        #     env = self.environments[env_id]
        #     results[env_id] = env.compute_reward()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for env_id in env_ids:
                future = executor.submit(self._compute_reward_single_env, env_id)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                env_id, result = future.result()
                results[env_id] = result
        print(f"[DEBUG] OSWorldService compute_reward_batch {results=}")
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        print(f"[DEBUG] OSWorldService get_system_prompts_batch {env_ids=}")
        results = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        print(f"[DEBUG] OSWorldService get_system_prompts_batch {results=}")
        return results
    
    def _close_single_env(self, env_id: str) -> None:
        env = self.environments[env_id]
        env.close()
        return
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        print(f"[DEBUG] OSWorldService close_batch {env_ids=}")
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # for env_id in env_ids:
        #     env = self.environments[env_id]
        #     env.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for env_id in env_ids:
                future = executor.submit(self._close_single_env, env_id)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                future.result()
            
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
        print(f"[DEBUG] OSWorldService close_batch done {env_ids=}")
        return