# https://github.com/nisheeth-golakiya/hybrid-sac/blob/main/hybrid_sac_platform.py

from numpy.random.mtrand import normal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard.writer import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
import gym_noita
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

SEED = 42

# Setting up environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('noita-v0')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

input_shape = 9
out_c = out_d = 3

TAU = 0.1
POLICY_FREQ = 1
BATCH_SIZE = 128
LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0

def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        torch.nn.init.constant_(layer.bias, bias_const) # zeros

def to_gym_action(action_c, action_d, flat_actions=True):
    if flat_actions:
        ac = action_c.tolist()[0]
    else:
        ac = action_c.unsqueeze(-1).tolist()[0]
    ad = action_d.squeeze().item()
    return [ad, ac]

def gym_to_buffer(action, flat_actions=False):
    box_a = action[0]
    box_b = action[1]
    if flat_actions:
        ac = np.hstack(action[2:])
    else:
        ac = action[2:]

    return [box_a, box_b] + np.array(ac).flatten().tolist()

def to_torch_action(actions, device):
    ad = torch.Tensor(actions[:, 0]).int().to(device)
    ac = torch.Tensor(actions[:, 1:]).to(device)

    return ac, ad

class Policy(nn.Module):

    def __init__(self, input_shape, out_c, out_d, env): 
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.mean = nn.Linear(128, out_c),
        self.logstd = nn.Linear(128, out_c)
        self.pi_d = nn.Linear(128, out_d)

        self.apply(layer_init)
    
    def foward(self, x, device):
        x = torch.Tensor(x).to(device)

        x = F.relu(self.fc1(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))
        pi_d = self.pi_d(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std, pi_d

    def get_action(self, x, device):
        mean, log_std, pi_d = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_c = torch.tanh(x_t)
        log_prob_c = normal.log_prob(x_t)
        log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d
    
    def to(self, device):
        return super(Policy, self).to(device)

class SoftQNetwork(nn.Module):

    def __init__(self, input_shape, out_c, out_d, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 128)
        self.fc2 = nn.Linear(128, out_d)
        self.apply(layer_init)
    
    def foward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ReplayBuffer():

    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s_prime_list.append(s_prime)
            done_mask_list.append(done_mask)

        return np.array(s_list), np.array(a_list), np.array(r_list), np.array(s_prime_list), np.array(done_mask_list)

rb = ReplayBuffer(10000)
pg = Policy(input_shape, out_c, out_d, env).to(device)
qf1 = SoftQNetwork(input_shape, out_c, out_d, layer_init).to(device)
qf2 = SoftQNetwork(input_shape, out_c, out_d, layer_init).to(device)
qf1_target = SoftQNetwork(input_shape, out_c, out_d, layer_init).to(device)
qf2_target = SoftQNetwork(input_shape, out_c, out_d, layer_init).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=1e-4)
loss_fn = nn.MSELoss()

# Entropy Tuning
auto_tune = True

alpha = 0.2
alpha_d = 0.2
target_entropy = -0.25
target_entropy_d = 0.25

if auto_tune:
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().detach().cpu().item()
    a_optimizer = optim.Adam([log_alpha], lr=1e-4)

    log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
    alpha_d = log_alpha_d.exp().detach().cpu().item()
    a_d_optimizer = optim.Adam([log_alpha_d], lr=1e-4)

# start the game
global_episode = 0
obs, done = env.reset(), False
episode_reward, episode_length = 0., 0

for global_step in range(1, 4000001):
    if global_step < 5e3:
        action = env.action_space.sample()
        #action_ = gym_to_buffer(action_)
        #action = [action_[0], action_[1:]]
    else:
        action_c, action_d, _, _, _ = pg.get_action([obs], device)
        action = to_gym_action(action_c, action_d)

    # step and log
    print("action: ", action)
    next_obs, reward, done, _ = env.step(action)
    rb.put((obs, gym_to_buffer(action), reward, next_obs, done))
    episode_reward += reward
    episode_length += 1
    obs = np.array(next_obs)
    
    # training.
    if len(rb.buffer) > BATCH_SIZE:  # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(BATCH_SIZE)
        with torch.no_grad():
            next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = pg.get_action(s_next_obses, device)
            qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions_c, device)
            qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions_c, device)

            min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_prob_d * next_state_log_pi_c - alpha_d * next_state_log_pi_d)
            next_q_value = torch.Tensor(s_rewards).to(
                device) + (1 - torch.Tensor(s_dones).to(device)) * 0.9 * (min_qf_next_target.sum(1)).view(-1)

        s_actions_c, s_actions_d = to_torch_action(s_actions, device)
        qf1_a_values = qf1.forward(s_obs, s_actions_c, device).gather(1, s_actions_d.long().view(-1, 1).to(device)).squeeze().view(-1)
        qf2_a_values = qf2.forward(s_obs, s_actions_c, device).gather(1, s_actions_d.long().view(-1, 1).to(device)).squeeze().view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2

        values_optimizer.zero_grad()
        qf_loss.backward()
        values_optimizer.step()

        if global_step % POLICY_FREQ == 0:  # TD 3 Delayed update support
            for _ in range(
                    POLICY_FREQ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                actions_c, actions_d, log_pi_c, log_pi_d, prob_d = pg.get_action(s_obs, device)
                qf1_pi = qf1.forward(s_obs, actions_c, device)
                qf2_pi = qf2.forward(s_obs, actions_c, device)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss_d = (prob_d * (alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                policy_loss_c = (prob_d * (alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                policy_loss = policy_loss_d + policy_loss_c

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if auto_tune:
                    with torch.no_grad():
                        a_c, a_d, lpi_c, lpi_d, p_d = pg.get_action(s_obs, device)
                    alpha_loss = (-log_alpha * p_d * (p_d * lpi_c + target_entropy)).sum(1).mean()
                    alpha_d_loss = (-log_alpha_d * p_d * (lpi_d + target_entropy_d)).sum(1).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().detach().cpu().item()

                    a_d_optimizer.zero_grad()
                    alpha_d_loss.backward()
                    a_d_optimizer.step()
                    alpha_d = log_alpha_d.exp().detach().cpu().item()

        # update the target network
        if global_step % 1 == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    if done:
        global_episode += 1

        (obs, _), done = env.reset(), False
        episode_reward, episode_length = 0.0, 0

env.close()

torch.save(pg.state_dict(), 'hybrid-sac_noita_test.pth')


