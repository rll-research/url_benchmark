import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import DDPGAgent


class ICM(nn.Module):
    """
    Same as ICM, with a trunk to save memory for KNN
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, icm_rep_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, icm_rep_dim),
                                   nn.LayerNorm(icm_rep_dim), nn.Tanh())

        self.forward_net = nn.Sequential(
            nn.Linear(icm_rep_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, icm_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * icm_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

    def get_rep(self, obs, action):
        rep = self.trunk(obs)
        return rep


class ICMAPTAgent(DDPGAgent):
    def __init__(self, icm_scale, knn_rms, knn_k, knn_avg, knn_clip,
                 update_encoder, icm_rep_dim, **kwargs):
        super().__init__(**kwargs)

        self.icm_scale = icm_scale
        self.update_encoder = update_encoder

        self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim,
                       icm_rep_dim).to(self.device)

        # optimizers
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.icm.train()

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        rep = self.icm.get_rep(obs, action)
        reward = self.pbe(rep)
        reward = reward.reshape(-1, 1)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_icm(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['intr_reward'] = intr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
