import copy
import logging
import datetime
import jsonlines
import os
import hashlib
import pickle
import lzma
import json
import uuid
import subprocess
from dataclasses import asdict
from .env_utils import (
    ObsPostProcessor, encoded_img_to_pil_img,
    parse_actions_from_string, parse_code_from_string, parse_code_from_som_string
)
from .render_utils import render_train_trajectory_to_html
from .env_config import OSWorldEnvConfig
from desktop_env.desktop_env import DesktopEnv
from desktop_env.providers.docker.provider import DockerProvider
from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper


logger = logging.getLogger(__name__)


class OSWorldEnv(BaseEnv):
    def __init__(self, config: OSWorldEnvConfig):
        BaseEnv.__init__(self)
        self.config = config
        pid = os.getpid()
        unique_cache_dir = os.path.join("/tmp/osworld_cache", str(pid), str(uuid.uuid4()))
        self.env = DesktopEnv(
            action_space=config.action_space,
            provider_name='docker',
            cache_dir=unique_cache_dir,
        )

        config_str = json.dumps(asdict(self.config))
        self._env_id = hashlib.sha256(config_str.encode()).hexdigest()
        self.obs_postprocesser = ObsPostProcessor(
            observation_type=config.observation_type,
            platform=config.platform,
            a11y_tree_max_tokens=config.a11y_tree_max_tokens,
        )
        self.fail_reward = 0.0
        print('OSWorldDebugEnv got env_id:', self._env_id)
        os.makedirs(config.tmp_save_path, exist_ok=True)

        task_config = config.task_config
        self.task_config = task_config
        self.instruction = task_config['instruction']
        self._reset_env()
        
        ### init other helper env states
        self._prev_processed_obs = {}
        self._prev_processed_obs_for_logging = {}
        self._last_obs = None
        self._last_action = ""
        
        self._is_infeasible = False
        if "evaluator" in task_config:
            if "func" in task_config["evaluator"]:
                if task_config["evaluator"]["func"] == "infeasible":
                    self._is_infeasible = True
        
        self._obs_processor_for_logging = ObsPostProcessor(
            observation_type="a11y_tree",
            platform=config.platform,
            a11y_tree_max_tokens=config.a11y_tree_max_tokens,
        ) 
        self._is_last_step_terminal = False
        self._ncalls = 0
        self._nsteps = 0  # this will be maintained externally since step does not take in response str
        self._curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._chat_hist = []
        self._trajectory = []
        return
    
    def _reset_env(self):
        self.env.provider.unpause_emulator()
        obs = self.env.reset(task_config=self.task_config)
        self.env.provider.pause_emulator()
        return obs
    
    def reset(self, seed=None):
        print(f'resetting env {self._env_id=}')
        try:
            obs = self._reset_env()
        except Exception as e:
            print(f'error resetting env {self._env_id=}: {e}')
            return None, {}
        self._last_obs = obs
        
        processed_obs = self.obs_postprocesser(obs)
        processed_obs_for_logging = self._obs_processor_for_logging(obs)
        self._prev_processed_obs = processed_obs
        self._prev_processed_obs_for_logging = processed_obs_for_logging
        
        ### logging
        instruction = self.config.task_config['instruction']
        self._chat_hist = [
            {"role": "user", "content": instruction + "\n" + processed_obs_for_logging['accessibility_tree']}
        ]
        self.__log_chat(self._chat_hist)
        ## return obs
        img_placeholder= self.config.get("image_placeholder", "<image>")
        obs = {
            'obs_str': f"Given the screenshot as below. What's the next step that you will do to help with the task?\n{img_placeholder}",
            'multi_modal_data': {
                img_placeholder: [encoded_img_to_pil_img(processed_obs['screenshot'])],
            },
        }
        info = {}
        return obs, info
    
    def _evaluate(self) -> float:
        ### really call env to evaluate
        self.env.provider.unpause_emulator()
        last_action = self._last_action
        print(f'evaluating env {self._env_id=} at {self._nsteps=} with {last_action=}')
        raw_score = self.env.evaluate()
        raw_score = float(raw_score)
        success = 1.0 if raw_score == 1.0 else 0.0
        self.env.provider.pause_emulator()
        
        # #### start of normal env reward logic
        # # zero intermediate reward if: 1) episode not finished; 2) max_steps not reached
        # if not self._is_last_step_terminal and self._nsteps < self.config.max_steps:
        #     score = 0.0
        
        # # terminal reward -1 if task not successfully done
        # if score == 0.0:
        #     # bad if max_steps reached and task is still not done
        #     if self._nsteps >= self.config.max_steps:
        #         score = self.fail_reward
        #     # or if its a terminal step but is incorrect
        #     if self._is_last_step_terminal:
        #         score = self.fail_reward
        # #### end of normal env reward logic
        
        ## debugging
        self.__log_chat(self._chat_hist + [
            {
                "score": raw_score,
                "success": success,
                "is_infeasible": self._is_infeasible,
                "n_steps": self._nsteps,
                "max_steps": self.config.max_steps,
                "last_action": last_action,
                "is_last_step_terminal": self._is_last_step_terminal,
            }
        ])
        return success
    
    def parse_actions(self, response: str):
        if self.config.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.config.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.config.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.config.action_space)
            return actions
        elif self.config.observation_type in ["som"]:
            # parse from the response
            if self.config.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.config.action_space)
            elif self.config.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, None)
            else:
                raise ValueError("Invalid action space: " + self.config.action_space)
            return actions
        else:
            raise ValueError("Invalid observation_type type: " + self.config.observation_type)
    
    def __log_chat(self, messages):
        chat_fpath = os.path.join(
            self.config.tmp_save_path,
            f'chat_env_{self._env_id}_{self._ncalls}_T{self._curr_time}.jsonl'
        )
        with jsonlines.open(chat_fpath, 'w') as writer:
            writer.write_all(messages)
        self._ncalls += 1
        return
    
    def __log_trajectory(self, trajectory: list):
        html_fpath = os.path.join(
            self.config.tmp_save_path, 
            f'trajectory_env_{self._env_id}_T{self._curr_time}.html'
        )
        task_config = self.config.task_config
        render_train_trajectory_to_html(
            task_config=task_config,
            trajectory=trajectory,
            additional_actions=task_config.get("init_exec_actions", []),
            postprocesser=self.obs_postprocesser,
            output_fpath=html_fpath,
        )
        return
    
    def _action_guardrail(self, action: str):
        ## prevent wierd behaviors from training
        if self.config.action_space == "pyautogui":
            max_chars = 1000
            if len(action) > max_chars:
                logger.warning(f"action is too long, truncationg to {max_chars} chars. original action: {action}")
                action = action[:max_chars]
        else:
            raise NotImplementedError("")
        return action
    
    @env_state_reward_wrapper
    def step(self, action_str: str):
        done = False  # always set to False so this can theoretically run forever, unless model said "DONE" or "FAIL"
        parsed_actions = self.parse_actions(action_str)
        if len(parsed_actions) == 0:
            ## bad
            reward = -0.5
            action = "None"
            processed_obs = self._prev_processed_obs
            processed_obs_for_logging = self._prev_processed_obs_for_logging
            action_is_effective = False
        else:
            action = parsed_actions[0].strip()
            action = self._action_guardrail(action)
            try:
                self.env.provider.unpause_emulator()
                obs, reward, done, _ = self.env.step(action)
                self.env.provider.pause_emulator()
            except Exception as e:
                print(f'error stepping env {self._env_id=} with action {action}: {e}')
                return None, 0.0, False, {}
            ### only really evaluate once task is done. otherwise just return 0.0
            if done:
                self._last_action = action
                self._is_last_step_terminal = done
                reward = self._evaluate()
            reward = float(reward)
            
            processed_obs = self.obs_postprocesser(obs)
            processed_obs_for_logging = self._obs_processor_for_logging(obs)
            action_is_effective = True
            if not self._is_last_step_terminal:
                action_is_effective = self._prev_processed_obs_for_logging['accessibility_tree'] != processed_obs_for_logging['accessibility_tree']
            self._prev_processed_obs_for_logging = processed_obs_for_logging
        self._is_last_step_terminal = done
        self._last_action = action

        ## logging
        self._chat_hist.append({"role": "assistant", "raw_resp": action_str, "content": action})
        self._chat_hist.append({"role": "user", "content": processed_obs_for_logging['accessibility_tree']})
        self.__log_chat(self._chat_hist)
        
        # self._trajectory.append(
        #     {"raw_action": action_str, "action": action, "step_idx": self._nsteps}
        # )
        # self._trajectory.append(
        #     {"obs": obs, "info": info, "reward": reward, "done": done}
        # )
        # self.__log_trajectory(self._trajectory)
        self._nsteps += 1
        
        ### return
        info = {
            "metrics": {
                "turn_metrics": {
                    'action_is_effective': action_is_effective,
                    'action_is_valid': len(parsed_actions) > 0,
                },
                "traj_metrics": {
                    'success': reward == 1.0,
                }
            },
            "llm_raw_response": action_str,
            "parsed_action": action,
        }
        img_placeholder= self.config.get("image_placeholder", "<image>")
        obs = {
            'obs_str': f"Given the screenshot as below. What's the next step that you will do to help with the task?\n{img_placeholder}",
            'multi_modal_data': {
                img_placeholder: [encoded_img_to_pil_img(processed_obs['screenshot'])],
            },
        }
        return obs, reward, done, info

    def system_prompt(self):
        return self.config.get_system_prompt()
    
    def close(self):
        print(f'closing env {self._env_id=}')
        closed = False
        try:
            self.env.provider.unpause_emulator()
            self.env.close()
            closed = True
        except Exception as e:
            print(f'error closing env {self._env_id=}: {e}')

        assert isinstance(self.env.provider, DockerProvider), f"env {self.env.provider} is not a docker provider"
        if not closed and self.env.provider.container is not None:
            print(f"trying subprocess to close env {self._env_id=} with container {self.env.provider.container.id}")
            container_id = self.env.provider.container.id
            stop_cmd = f"docker stop {container_id}"
            subprocess.run(stop_cmd, shell=True)
            remove_cmd = f"docker rm -v {container_id}"
            subprocess.run(remove_cmd, shell=True)
        return
    
    def compute_reward(self):
        return 0.0