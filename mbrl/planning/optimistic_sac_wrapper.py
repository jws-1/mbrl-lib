# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac
import mbrl.models
import mbrl.models.util as model_util
from .core import Agent



class OptimisticSACAgent(Agent):
    """An Optimistic Soft-Actor Critic agent.

    This class is a wrapper for
    https://github.com/luisenp/pytorch_sac/blob/master/pytorch_sac/agent/sac.py


    Args:
        (pytorch_sac.SACAgent): the agent to wrap.
    """

    def __init__(self, sac_agent: pytorch_sac.SAC, dynamics_model: mbrl.models.OneDTransitionRewardModel, correction_model: mbrl.models.OneDTransitionRewardModel, action_low : np.ndarray, action_high: np.ndarray, lr : float, steps: int, exp_value_num_samples: int, reward_fn, gamma: float):
        self.sac_agent = sac_agent
        self.dynamics_model = dynamics_model
        self.correction_model = correction_model
        self.lr = lr
        self.steps = steps
        self.exp_value_num_samples = exp_value_num_samples
        self.action_low = torch.from_numpy(action_low)
        self.action_high = torch.from_numpy(action_high)
        self.reward_fn = reward_fn
        self.gamma = gamma

    def _expected_value(self, dist: torch.distributions.Normal) -> torch.Tensor:
        samples = dist.rsample((self.exp_value_num_samples,))
        if self.correction_model.learned_rewards:
            samples = samples[..., :-1]
        samples = samples.squeeze(dim=1)
        actions = self.sac_agent.policy.sample(samples)[0]
        q1, q2 = self.sac_agent.critic(samples, actions)
        return torch.min(q1.mean(), q2.mean())

    def _objective(self, obs, action):
        dist = self._dist(self.correction_model, obs, action)
        samples = dist.rsample((self.exp_value_num_samples,))
        if self.correction_model.learned_rewards:
            samples = samples[..., :-1]
        samples = samples.squeeze(dim=1)
        actions = self.sac_agent.policy.sample(samples)[0]
        q1, q2 = self.sac_agent.critic(samples, actions)
        v = torch.min(q1, q2)
        if self.reward_fn is not None:
            r = self.reward_fn(actions, samples)
        else:
            rew_dist = self._dist(self.dynamics_model, obs, action)
            r = rew_dist.rsample((self.exp_value_num_samples,))
            r = r[..., -1]
        return (r + self.gamma * v).mean()

    def _dist(self, model, obs, action):
        p = model.model.propagation_method
        model.model.set_propagation_method("expectation")
        dist = model.dist(obs, action)
        model.model.set_propagation_method(p)
        return dist

    def _get_optimistic_action(self, obs: np.ndarray, greedy_action: np.ndarray, action_dist: torch.distributions.Normal) -> np.ndarray:
        eps = 1e-6
        obs_tensor = torch.from_numpy(obs).float()
        greedy_action_tensor = torch.from_numpy(greedy_action).float()
        # action_low = action_dist.mean - action_dist.stddev
        # action_high = action_dist.mean + action_dist.stddev
        # action_low = torch.max(action_low.cpu(), self.action_low)
        # action_high = torch.min(action_high.cpu(), self.action_high)
        action_low = self.action_low
        action_high = self.action_high
        greedy_objective = min(self.sac_agent.critic(obs_tensor.unsqueeze(0).to(self.sac_agent.device), greedy_action_tensor.unsqueeze(0).to(self.sac_agent.device)))
        random_action = action_low + (action_high - action_low) * torch.rand_like(action_low)
        normalized_action = 2.0 * (random_action - action_low) / (action_high - action_low) - 1.0
        normalized_action = torch.clamp(normalized_action, -1 + eps, 1 - eps)
        z = torch.atanh(normalized_action.clone().detach()).requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=self.lr)

        best_val = -np.inf
        best_action = None

        for step in range(self.steps):
            optimizer.zero_grad()
            a = torch.tanh(z)
            scaled_a = self.action_low + 0.5 * (a + 1.0) * (self.action_high - self.action_low)
            objective = self._objective(obs_tensor, scaled_a)
            loss = -objective
            loss.backward()
            optimizer.step()

            if objective.item() > best_val:
                best_val = objective.item()
                best_action = scaled_a.detach().clone().requires_grad_(False)
        
        if best_val > greedy_objective:
            return best_action.cpu().numpy()
        return greedy_action
    
    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, epsilon: float = 0.5, explore: bool = True, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation (or batch of observations) for which the action
                is needed.
            sample (bool): if ``True`` the agent samples actions from its policy, otherwise it
                returns the mean policy value. Defaults to ``False``.
            batched (bool): if ``True`` signals to the agent that the obs should be interpreted
                as a batch.

        Returns:
            (np.ndarray): the action.
        """
        if explore:
            with torch.no_grad():
                greedy_action = self.sac_agent.select_action(
                    obs, batched=False, evaluate=True
                )

            if np.random.rand() < epsilon:
                return self._get_optimistic_action(obs, greedy_action, self.sac_agent.policy.get_distribution(torch.from_numpy(obs).to(self.sac_agent.device).float()))
            else:
                return greedy_action
        else:
            return self.sac_agent.select_action(obs, batched=batched, evaluate=not sample)
            