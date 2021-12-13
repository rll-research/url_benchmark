import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.ddpg import DDPGAgent

# TODO(HL): how to include GPI for continuous domain?


class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 sf_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action, task):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        q1 = torch.einsum("bi,bi->b", task, q1).reshape(-1, 1)
        q2 = torch.einsum("bi,bi->b", task, q2).reshape(-1, 1)

        return q1, q2


class APS(nn.Module):
    def __init__(self, obs_dim, sf_dim, hidden_dim):
        super().__init__()
        self.state_feat_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, sf_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, norm=True):
        state_feat = self.state_feat_net(obs)
        state_feat = F.normalize(state_feat, dim=-1) if norm else state_feat
        return state_feat


class APSAgent(DDPGAgent):
    def __init__(self, update_task_every_step, sf_dim, knn_rms, knn_k, knn_avg,
                 knn_clip, num_init_steps, lstsq_batch_size, update_encoder,
                 **kwargs):
        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        self.lstsq_batch_size = lstsq_batch_size
        self.update_encoder = update_encoder

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(**kwargs)

        # overwrite critic with critic sf
        self.critic = CriticSF(self.obs_type, self.obs_dim, self.action_dim,
                               self.feature_dim, self.hidden_dim,
                               self.sf_dim).to(self.device)
        self.critic_target = CriticSF(self.obs_type, self.obs_dim,
                                      self.action_dim, self.feature_dim,
                                      self.hidden_dim,
                                      self.sf_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.lr)

        self.aps = APS(self.obs_dim - self.sf_dim, self.sf_dim,
                       kwargs['hidden_dim']).to(kwargs['device'])

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        # optimizers
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

        self.aps.train()

    def get_meta_specs(self):
        return (specs.Array((self.sf_dim,), np.float32, 'task'),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def update_aps(self, task, next_obs, step):
        metrics = dict()

        loss = self.compute_aps_loss(next_obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.aps_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['aps_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, task, next_obs, step):
        # maxent reward
        with torch.no_grad():
            rep = self.aps(next_obs, norm=False)
        reward = self.pbe(rep)
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward, intr_sf_reward

    def compute_aps_loss(self, next_obs, task):
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", task, self.aps(next_obs)).mean()
        return loss

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, task = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            # freeze successor features at finetuning phase
            metrics.update(self.update_aps(task, next_obs, step))

            with torch.no_grad():
                intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                    task, next_obs, step)
                intr_reward = intr_ent_reward + intr_sf_reward

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                metrics['intr_sf_reward'] = intr_sf_reward.mean().item()

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), task, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), task, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        obs, reward = [], []
        batch_size = 0
        while batch_size < self.lstsq_batch_size:
            batch = next(replay_iter)
            batch_obs, _, batch_reward, *_ = utils.to_torch(batch, self.device)
            obs.append(batch_obs)
            reward.append(batch_reward)
            batch_size += batch_obs.size(0)
        obs, reward = torch.cat(obs, 0), torch.cat(reward, 0)

        obs = self.aug_and_encode(obs)
        rep = self.aps(obs)
        task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # save for evaluation
        self.solved_meta = meta
        return meta

    def update_critic(self, obs, action, reward, discount, next_obs, task,
                      step):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,
                                                      task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, task)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
