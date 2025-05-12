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

    def __init__(self, sac_agent: pytorch_sac.SAC, correction_model: mbrl.models.OneDTransitionRewardModel, dynamics_model: mbrl.models.OneDTransitionRewardModel,lr : float, steps: int, exp_value_num_samples: int):
        self.sac_agent = sac_agent
        self.dynamics_model = dynamics_model
        self.correction_model = correction_model
        self.lr = lr
        self.steps = steps
        self.exp_value_num_samples = exp_value_num_samples

    def _expected_value(self, dist: torch.distributions.Normal) -> torch.Tensor:
        with torch.no_grad():
            samples = dist.sample((self.exp_value_num_samples,))
        if self.correction_model.learned_rewards:
            samples = samples[..., :-1]
        samples = samples.squeeze(dim=1)
        actions = self.sac_agent.policy.sample(samples)[0]
        q1, q2 = self.sac_agent.critic(samples, actions)
        return min(q1.mean(), q2.mean())

    def _get_optimistic_action(self, obs: np.ndarray, greedy_action: np.ndarray) -> np.ndarray:
        eps = 1e-6
        obs_tensor = torch.from_numpy(obs)
        init_action = torch.clamp(torch.from_numpy(greedy_action), -1 + eps, 1 + eps)
        z = (
            torch.atanh(init_action.clone().detach())
            .requires_grad_(True)
        )
        optimizer = torch.optim.SGD([z], lr=self.lr)

        best_val = -float("inf")
        best_action = None

        for _ in range(self.steps):
            optimizer.zero_grad()
            a = torch.tanh(z)
            p = self.correction_model.model.propagation_method
            self.correction_model.set_propagation_method("expectation")
            with torch.no_grad():
                dist = self.correction_model.dist(obs_tensor, a)
            self.correction_model.model.set_propagation_method(p)
            value = self._expected_value(dist)
            loss = -value
            loss.backward()
            optimizer.step()

            if value.item() > best_val:
                best_action = a.detach().clone().requires_grad_(False)
                best_dist = dist

        return best_action.cpu().numpy()

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
                return self._get_optimistic_action(obs, greedy_action)
            else:
                return greedy_action
        else:
            return self.sac_agent.select_action(obs, batched=batched, evaluate=not sample)
            