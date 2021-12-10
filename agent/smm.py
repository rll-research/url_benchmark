import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent

"""
Reimplementation of https://github.com/RLAgent/state-marginal-matching:
 - Removed redundant forward passes
 - No updating p_z
 - Added finetuning procedure from what's described in DIAYN
 - VAE encodes and decodes from the encoding from DDPG when n > 1
   as the paper does not make it clear how to include skills with pixel input
 - When n=1, obs_type=pixel, remove the False from line 144
    to input pixels into the vae
 - TODO: when using pixel-based vae (n=1), gpu may run out of memory.
"""


class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim, code_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.make_networks(obs_dim, z_dim, code_dim)
        self.beta = vae_beta

        self.apply(utils.weight_init)
        self.device = device

    def make_networks(self, obs_dim, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Linear(obs_dim + z_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(nn.Linear(code_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU(),
                                 nn.Linear(150, obs_dim + z_dim))

    def encode(self, obs_z):
        enc_features = self.enc(obs_z)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def loss(self, obs_z):
        epsilon = torch.randn([obs_z.shape[0], self.code_dim]).to(self.device)
        obs_distr_params, (mu, logvar, stds) = self(obs_z, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                               dim=1).mean()
        log_prob = F.mse_loss(obs_z, obs_distr_params, reduction='none')

        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(
            log_prob.shape[0], 1)


class PVae(VAE):
    def make_networks(self, obs_shape, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Flatten(),
                                 nn.Linear(32 * 35 * 35, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 32 * 35 * 35), nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 35, 35)),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, obs_shape[0], 4, stride=1))


class SMM(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, z_dim))
        self.vae = VAE(obs_dim=obs_dim,
                       z_dim=z_dim,
                       code_dim=128,
                       vae_beta=vae_beta,
                       device=device)
        self.apply(utils.weight_init)

    def predict_logits(self, obs):
        z_pred_logits = self.z_pred_net(obs)
        return z_pred_logits

    def loss(self, logits, z):
        z_labels = torch.argmax(z, 1)
        return nn.CrossEntropyLoss(reduction='none')(logits, z_labels)


class PSMM(nn.Module):
    def __init__(self, obs_shape, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.vae = PVae(obs_dim=obs_shape,
                        z_dim=z_dim,
                        code_dim=128,
                        vae_beta=vae_beta,
                        device=device)
        self.apply(utils.weight_init)

    # discriminator not needed when n=1, as z is degenerate
    def predict_logits(self, obs):
        raise NotImplementedError

    def loss(self, logits, z):
        raise NotImplementedError


class SMMAgent(DDPGAgent):
    def __init__(self, z_dim, sp_lr, vae_lr, vae_beta, state_ent_coef,
                 latent_ent_coef, latent_cond_ent_coef, update_encoder,
                 **kwargs):
        self.z_dim = z_dim

        self.state_ent_coef = state_ent_coef
        self.latent_ent_coef = latent_ent_coef
        self.latent_cond_ent_coef = latent_cond_ent_coef
        self.update_encoder = update_encoder

        kwargs["meta_dim"] = self.z_dim
        super().__init__(**kwargs)
        # self.obs_dim is now the real obs_dim (or repr_dim) + z_dim
        self.smm = SMM(self.obs_dim - z_dim,
                       z_dim,
                       hidden_dim=kwargs['hidden_dim'],
                       vae_beta=vae_beta,
                       device=kwargs['device']).to(kwargs['device'])
        self.pred_optimizer = torch.optim.Adam(
            self.smm.z_pred_net.parameters(), lr=sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(),
                                              lr=vae_lr)

        self.smm.train()

        # fine tuning SMM agent
        self.ft_returns = np.zeros(z_dim, dtype=np.float32)
        self.ft_not_finished = [True for z in range(z_dim)]

    def get_meta_specs(self):
        return (specs.Array((self.z_dim,), np.float32, 'z'),)

    def init_meta(self):
        z = np.zeros(self.z_dim, dtype=np.float32)
        z[np.random.choice(self.z_dim)] = 1.0
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def update_meta(self, meta, global_step, time_step):
        # during fine-tuning, find the best skill and fine-tune that one only.
        if self.reward_free:
            return self.update_meta_ft(meta, global_step, time_step)
        # during training, change to randomly sampled z at the end of the episode
        if time_step.last():
            return self.init_meta()
        return meta

    def update_meta_ft(self, meta, global_step, time_step):
        z_ind = meta['z'].argmax()
        if any(self.ft_not_finished):
            self.ft_returns[z_ind] += time_step.reward
            if time_step.last():
                if not any(self.ft_not_finished):
                    # choose the best
                    new_z_ind = self.ft_returns.argmax()
                else:
                    # or the next z to try
                    self.ft_not_finished[z_ind] = False
                    not_tried_z = sum(self.ft_not_finished)
                    # uniformly sample from the remaining unused z
                    for i in range(self.z_dim):
                        if self.ft_not_finished[i]:
                            if np.random.random() < 1 / not_tried_z:
                                new_z_ind = i
                                break
                            not_tried_z -= 1
                new_z = np.zeros(self.z_dim, dtype=np.float32)
                new_z[new_z_ind] = 1.0
                meta['z'] = new_z
        return meta

    def update_vae(self, obs_z):
        metrics = dict()
        loss, h_s_z = self.smm.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.vae_optimizer.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        metrics['loss_vae'] = loss.cpu().item()

        return metrics, h_s_z

    def update_pred(self, obs, z):
        metrics = dict()
        logits = self.smm.predict_logits(obs)
        h_z_s = self.smm.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()

        metrics['loss_pred'] = loss.cpu().item()

        return metrics, h_z_s

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, z = utils.to_torch(
            batch, self.device)

        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
        obs_z = torch.cat([obs, z], dim=1)  # do not learn encoder in the VAE
        next_obs_z = torch.cat([next_obs, z], dim=1)

        if self.reward_free:
            vae_metrics, h_s_z = self.update_vae(obs_z)
            pred_metrics, h_z_s = self.update_pred(obs.detach(), z)

            h_z = np.log(self.z_dim)  # One-hot z encoding
            h_z *= torch.ones_like(extr_reward).to(self.device)

            pred_log_ratios = self.state_ent_coef * h_s_z.detach(
            )  # p^*(s) is ignored, as state space dimension is inaccessible from pixel input
            intr_reward = pred_log_ratios + self.latent_ent_coef * h_z + self.latent_cond_ent_coef * h_z_s.detach(
            )
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics.update(vae_metrics)
            metrics.update(pred_metrics)
            metrics['intr_reward'] = intr_reward.mean().item()
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs_z = obs_z.detach()
            next_obs_z = next_obs_z.detach()

        # update critic
        metrics.update(
            self.update_critic(obs_z.detach(), action, reward, discount,
                               next_obs_z.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs_z.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
