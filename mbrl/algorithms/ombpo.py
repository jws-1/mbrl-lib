# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
import mbrl.models.util as model_util
from mbrl.planning.optimistic_sac_wrapper import OptimisticSACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
import wandb

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: OptimisticSACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True, explore=False)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        truncateds = np.zeros_like(pred_dones, dtype=bool)
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
            truncateds[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: OptimisticSACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
) -> float:
    avg_episode_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        video_recorder.init(enabled=(episode == 0))
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer

def maybe_add_to_correction_buffer(
    obs, action, next_obs, reward, terminated, truncated, cfg, correction_buffer, dynamics_model
):
    if not dynamics_model.trained:
        return
    obs = model_util.to_tensor(obs)
    action = model_util.to_tensor(action)
    next_obs = model_util.to_tensor(next_obs)
    low = (1.0 - cfg.algorithm.percentile) / 2
    high = (
         cfg.algorithm.percentile + low
    )
    p = dynamics_model.model.propagation_method
    dynamics_model.set_propagation_method("expectation")
    with torch.no_grad():
        next_state_dist = dynamics_model.dist(obs, action)
    if dynamics_model.learned_rewards:
        next_state_dist = torch.distributions.Normal(
            next_state_dist.loc[..., :-1].squeeze().cpu(),
            next_state_dist.scale[..., :-1].squeeze().cpu()
        )
    dynamics_model.set_propagation_method(p)
    low_percentile = next_state_dist.icdf(
        torch.full_like(next_obs, low).cpu()
    )
    high_percentile = next_state_dist.icdf(
        torch.full_like(next_obs, high).cpu()
    )
    incorrect = (low_percentile > next_obs).any() or (next_obs > high_percentile).any()
    if incorrect:
        correction_buffer.add(
            obs.cpu().numpy(),
            action.cpu().numpy(),
            next_obs.cpu().numpy(),
            reward,
            terminated,
            truncated,
        )
        if cfg.debug_mode:
            print(
                f"Transition <{obs.cpu().numpy()}, {action.cpu().numpy()}, {next_obs.cpu().numpy()}> was incorrect, low percentile: {low}, high percentile: {high} (dist: Normal({next_state_dist.mean.cpu().numpy()}, {next_state_dist.stddev.cpu().numpy()}))"
            )

def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)


    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model.trained = False
    correction_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    correction_model.trained = False
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    correction_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    correction_model_trainer= mbrl.models.ModelTrainer(
        correction_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
        log_group_name="correction_model"
    )

    # ------------------- Initialization of agent -------------------
    agent = OptimisticSACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent)), dynamics_model, correction_model, env.action_space.low, env.action_space.high, cfg.algorithm.action_optim_lr, cfg.algorithm.action_optim_steps, cfg.algorithm.exp_value_num_samples, reward_fn, cfg.overrides.sac_gamma
    )

    # -------------- Seed models --------------
    random_explore = cfg.algorithm.random_initial_explore
    if cfg.algorithm.initial_exploration_steps:
        mbrl.util.common.rollout_agent_trajectories(
            env,
            cfg.algorithm.initial_exploration_steps,
            mbrl.planning.RandomAgent(env) if random_explore else agent,
            {} if random_explore else {"sample": True, "batched": False, "explore": False},
            replay_buffer=replay_buffer,
        )

    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs = None
        terminated = False
        truncated = False
        train_reward = 0.0
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False
            # --- Doing env step and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {"epsilon": cfg.algorithm.epsilon, "explore": correction_model.trained, "sample": True}, replay_buffer, callback = lambda args: maybe_add_to_correction_buffer(
                    *args, cfg, correction_buffer, dynamics_model
                )
            )
            train_reward += reward

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
                dynamics_model.trained = True
                if len(correction_buffer):
                    mbrl.util.common.train_model_and_save_model_and_data(
                        correction_model,
                        correction_model_trainer,
                        cfg.overrides,
                        correction_buffer,
                        work_dir=work_dir
                    )
                correction_model.trained = True

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                        "dataset_size": len(replay_buffer),
                        "correction_dataset_size": len(correction_buffer),
                    },
                )
                if cfg.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "rollout_length": rollout_length,
                            "dataset_size": len(replay_buffer),
                            "correction_dataset_size": len(correction_buffer),
                        },
                        step=env_steps,
                    )
                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
