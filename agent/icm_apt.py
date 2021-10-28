import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from agent.ddpg import DDPGAgent


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, apt_rep_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, apt_rep_dim))

        self.forward_net = nn.Sequential(
            nn.Linear(apt_rep_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, apt_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * apt_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        embed_obs = self.encoder(obs)
        embed_next_obs = self.encoder(next_obs)
        next_obs_hat = self.forward_net(torch.cat([embed_obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([embed_obs, embed_next_obs], dim=-1))

        forward_error = torch.norm(embed_next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error


class ICMAPTAgent(DDPGAgent):
    def __init__(self, icm_scale, knn_rms, knn_k, knn_avg, knn_clip, apt_rep_dim, update_encoder, state_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.icm_scale = icm_scale
        self.update_encoder = update_encoder

        self.icm = ICM(self.obs_dim, self.action_dim,
                       self.hidden_dim, apt_rep_dim).to(self.device)

        # optimizers
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(),
                                              lr=self.lr)

        self.icm.train()

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        self.state_flag = state_flag

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_optimizer.zero_grad()
        loss.backward()
        self.icm_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        # entropy reward
        with torch.no_grad():
            if self.state_flag:
                rep = next_obs
            else:
                rep = self.icm.encoder(next_obs)
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
            metrics.update(
                self.update_icm(obs.detach(), action, next_obs.detach(), step))

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
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
