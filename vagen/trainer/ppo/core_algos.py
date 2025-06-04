# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return_with_loss_mask(token_level_rewards: torch.Tensor, values: torch.Tensor, 
                                 loss_mask: torch.Tensor, gamma: float, lam: float):
    """Modified GAE calculation that handle multi-turn with loss mask
    Here we should also ensure that the trajectory score is given at the last valid token instead of last token
    Seems it's true in reward manager
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        for b in range(batch_size):
            lastgaelam = 0.0
            
            # Find the valid token positions (where loss_mask is 1)
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            
            if len(valid_positions) == 0:
                continue
                
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                # Get the next value
                if i < len(valid_positions) - 1:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            # Calculate returns for valid positions
            for i, pos in enumerate(valid_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
        

        advantages = verl_F.masked_whiten(advantages, loss_mask)
        
    return advantages, returns


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0 
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t] # TD error
            lastgaelam = delta + gamma * lam * lastgaelam # gae
            advantages_reversed.append(lastgaelam) # store the gae
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
    return advantages, returns


def compute_reinforce_plus_plus_outcome_advantage_with_loss_mask(
    token_level_rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: torch.Tensor
):
    """
    reinforce++ but gae skips non action tokens in loss mask
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        advantages = torch.zeros_like(token_level_rewards)
        bsz = token_level_rewards.shape[0]
        for b in range(bsz):
            running_return = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                raise ValueError("No action tokens found in loss mask")
           
            for t in reversed(valid_positions):
                running_return = token_level_rewards[b, t] + gamma * running_return
                returns[b, t] = running_return
                running_return = running_return * loss_mask[b, t]
        advantages = verl_F.masked_whiten(returns, loss_mask)
        advantages = advantages * loss_mask
    return advantages, returns
    