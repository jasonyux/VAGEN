import logging
import datetime
import jsonlines
import os
import hashlib
import pickle
import lzma
import json
from dataclasses import asdict
from .env_utils import (
    ObsPostProcessor, from_api_safe_obs,
    parse_actions_from_string, parse_code_from_string, parse_code_from_som_string
)
from .render_utils import render_train_trajectory_to_html
from .env_config import OSWorldDebugEnvConfig
from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper


logger = logging.getLogger(__name__)


class OSWorldDebugEnv(BaseEnv):
    def __init__(self, config: OSWorldDebugEnvConfig):
        BaseEnv.__init__(self)
        self.config = config
        self.env = None ## dummy env 

        config_str = json.dumps(asdict(self.config))
        self._env_id = hashlib.sha256(config_str.encode()).hexdigest()
        self.obs_postprocesser = ObsPostProcessor(
            observation_type=config.observation_type,
            platform=config.platform,
            a11y_tree_max_tokens=config.a11y_tree_max_tokens,
        )
        print('OSWorldDebugEnv got env_id:', self._env_id)

        ### load prerendered trajectories
        with lzma.open(config.prerender_trajectory_fpath, 'rb') as fread:
            fake_obs_per_instruction = pickle.load(fread)
        task_config = config.task_config
        instruction = task_config['instruction']
        self._random_traj = fake_obs_per_instruction[instruction]
        self._last_obs = self._random_traj[-1]
        
        
        self._is_infeasible = False
        if "evaluator" in task_config:
            if "func" in task_config["evaluator"]:
                if task_config["evaluator"]["func"] == "infeasible":
                    self._is_infeasible = True
        
        self._is_last_step_terminal = False
        self._ncalls = 0
        self._nsteps = 0  # this will be maintained externally since step does not take in response str
        self._curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._chat_hist = []
        self._trajectory = []
        return
    
    def reset(self, seed=None):
        print(f'resetting env {self._env_id=}')
        encoded_obs, _, _, _ = self._random_traj.pop(0)
        obs = from_api_safe_obs(encoded_obs)
        
        processed_obs = self.obs_postprocesser(obs)
        instruction = self.config.task_config['instruction']
        self._chat_hist = [
            {"role": "user", "content": instruction + "\n" + processed_obs['accessibility_tree']}
        ]
        self.__log_chat(self._chat_hist)
        
        self._trajectory = [
            {"obs": obs, "info": None, "reward": 0.0, "done": False},  # raw data
        ]
        self.__log_trajectory(self._trajectory)

        
        ## return obs
        obs = {
            'obs_str': obs['accessibility_tree'],
            'multi_modal_data': {
                'screenshot': obs['screenshot'],
            },
        }
        info = {}
        return obs, info
    
    def _evaluate(self) -> float:
        last_action = self._trajectory[-2]["action"]
        print(f'evaluating env {self._env_id=} at {self._nsteps=} with {last_action=}')
        score = 0.0
        #### semi real env logic
        if self._is_infeasible:
            if last_action == "FAIL":
                # very good
                score = 1.0
        else:
            # fail if we didn't reach the end of the trajectory
            # and success IFF we termianted
            if self._is_last_step_terminal and last_action == "DONE":
                if len(self._random_traj) != 0:
                    score = -1.0  # fail
                else:
                    score = 1.0
        
        #### start of normal env reward logic
        # zero intermediate reward if: 1) episode not finished; 2) max_steps not reached
        if not self._is_last_step_terminal and self._nsteps < self.config.max_steps:
            score = 0.0
        
        # terminal reward -1 if task not successfully done
        if score == 0.0:
            # bad if max_steps reached and task is still not done
            if self._nsteps >= self.config.max_steps:
                score = -1.0
            # or if its a terminal step but is incorrect
            if self._is_last_step_terminal:
                score = -1.0
        #### end of normal env reward logic
        
        if self.config.always_zero_reward:
            score = 0.0
        
        ## debugging
        self.__log_chat(self._chat_hist + [
            {
                "score": score,
                "is_infeasible": self._is_infeasible,
                "n_steps": self._nsteps,
                "max_steps": self.config.max_steps,
                "last_action": last_action,
                "len(self._random_traj)": len(self._random_traj),
                "is_last_step_terminal": self._is_last_step_terminal,
                "always_zero_reward": self.config.always_zero_reward
            }
        ])
        return score
    
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
    
    @env_state_reward_wrapper
    def step(self, action_str: str):
        if len(self._random_traj) == 0:
            encoded_obs, reward, done, info = self._last_obs
        else:
            encoded_obs, reward, done, info = self._random_traj.pop(0)

        done = False  # always set to False so this can theoretically run forever, unless model said "DONE" or "FAIL"
        parsed_actions = self.parse_actions(action_str)
        action = parsed_actions[0]  # for simplicity, we only support one action per response
        if action.strip() in ["DONE", "FAIL"]:
            done = True
            reward = self._evaluate()
        
        self._is_last_step_terminal = done
        
        ## debugging
        obs = from_api_safe_obs(encoded_obs)
        processed_obs = self.obs_postprocesser(obs)
        self._chat_hist.append({"role": "assistant", "raw_resp": action_str, "content": action})
        self._chat_hist.append({"role": "user", "content": processed_obs['accessibility_tree']})
        self.__log_chat(self._chat_hist)
        
        self._trajectory.append(
            {"raw_action": action_str, "action": action, "step_idx": self._nsteps}
        )
        self._trajectory.append(
            {"obs": obs, "info": info, "reward": reward, "done": done}
        )
        self.__log_trajectory(self._trajectory)
        self._nsteps += 1
        
        ### return
        info = {
            "metrics": {
                'success': False,
                'action_is_effective': False,
                'action_is_valid': len(parsed_actions) > 0,
            },
            "llm_raw_response": action_str,
            "llm_response": parsed_actions[0],
        }
        obs = {
            'obs_str': obs['accessibility_tree'],
            'multi_modal_data': {
                'screenshot': obs['screenshot'],
            },
        }
        return obs, reward, done, info

    def system_prompt(self):
        return "You are a helpful assistant."
    
    def close(self):
        print(f'closing env {self._env_id=}')
        return
    
    def compute_reward(self):
        return 0.0