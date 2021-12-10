import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import DDPGAgent


class Disagreement(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, n_models=5):
        super().__init__()
        self.ensemble = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim),
                          nn.ReLU(), nn.Linear(hidden_dim, obs_dim))
            for _ in range(n_models)
        ])

    def forward(self, obs, action, next_obs):
        #import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        errors = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            model_error = torch.norm(next_obs - next_obs_hat,
                                     dim=-1,
                                     p=2,
                                     keepdim=True)
            errors.append(model_error)

        return torch.cat(errors, dim=1)

    def get_disagreement(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        preds = []
        for model in self.ensemble:
            next_obs_hat = model(torch.cat([obs, action], dim=-1))
            preds.append(next_obs_hat)
        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0).mean(dim=-1)


class DisagreementAgent(DDPGAgent):
    def __init__(self, update_encoder, **kwargs):
        super().__init__(**kwargs)
        self.update_encoder = update_encoder

        self.disagreement = Disagreement(self.obs_dim, self.action_dim,
                                         self.hidden_dim).to(self.device)

        # optimizers
        self.disagreement_opt = torch.optim.Adam(
            self.disagreement.parameters(), lr=self.lr)

        self.disagreement.train()

    def update_disagreement(self, obs, action, next_obs, step):
        metrics = dict()

        error = self.disagreement(obs, action, next_obs)

        loss = error.mean()

        self.disagreement_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.disagreement_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['disagreement_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        reward = self.disagreement.get_disagreement(obs, action,
                                                    next_obs).unsqueeze(1)
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
                self.update_disagreement(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
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
