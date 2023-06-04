import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .memory import Memory


class PPOAgentNets(nn.Module):
    def __init__(self, obs_shape, act_n, embed=512):
        super(PPOAgentNets, self).__init__()

        self.conv1 = nn.Conv2d(obs_shape[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear = nn.Linear(self._get_conv_out(obs_shape), embed)
        self.linear2 = nn.Linear(embed, embed)
        
        self.actor = nn.Linear(embed, act_n)
        self.softmax = nn.Softmax(dim=-1)

        self.critic = nn.Linear(embed, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _convolutions(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.conv4(x))
    
    def _linears(self, x):
        x = F.relu(self.linear(x))
        return F.relu(self.linear2(x))
    
    def _get_conv_out(self, obs_shape):

        x = torch.zeros(1, *obs_shape)
        x = self._convolutions(x)

        return int(np.prod(x.shape))
    
    def get_value(self, x):
        num_batches = x.size(0)

        x = self._convolutions(x)
        x = x.view(num_batches, -1)
        x = self._linears(x)

        return self.critic(x)
    
    def get_act_and_value(self, x, action=None):
        num_batches = x.size(0)

        x = self._convolutions(x)
        x = x.view(num_batches, -1)
        x = self._linears(x)

        logits = self.softmax(self.actor(x))
        probs = Categorical(logits)

        if action == None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
class PPOAgent():
    # defaults to 'ideal' values from https://arxiv.org/pdf/1707.06347.pdf
    def __init__(
        self,

        obs_shape,
        act_shape,
        act_n,

        buffer_size = 2048,
        batch_size = 64,
        epochs = 4,

        lr = 2.5e-4,
        scheduler_gamma = None,

        discount = .99,
        gae_lambda = 0.95,
        policy_clip = 0.2,
        norm_advantage = False,

        entropy_coeff = 0.01,
        critic_coeff = 0.5,
        max_grad_norm = 0.5,

        early_stop_kl = None,

        embed = 512,
    ):
        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init agent memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs,
        self.mem = Memory(self.buffer_size, self.batch_size, obs_shape, act_shape)

        # init agents
        self.conv_net = PPOAgentNets(obs_shape, act_n, embed=embed).to(self.device)

        # init optimizer and optionally scheduler for descent
        self.scheduler_gamma = scheduler_gamma
        self.optimizer = torch.optim.Adam(self.conv_net.parameters(), lr=lr, eps=1e-5)
        if self.scheduler_gamma != None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.scheduler_gamma)
        
        # advantage calc hyperparams
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.norm_advantage = norm_advantage

        # loss calc hyperparams
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.max_grad_norm = max_grad_norm

        # kl div spikes can indicate a bad policy, so early 
        # stop can be beneficial to learning
        self.early_stop_kl = early_stop_kl

    def cache(self, state, action, log_prob, vals, reward, done):
        self.mem.cache(state, action, log_prob, vals, reward, done)

    def mem_full(self):
        return len(self.mem) == self.buffer_size

    def clear_mem(self):
        self.mem.clear_mem()

    def save(self, path='model.pt'):
        torch.save({
            'model': self.conv_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path='model.pt'):
        checkpoint = torch.load(path, map_location=self.device)

        self.conv_net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def act(self, obs):
        # convert state to N C W H tensor
        state = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device).unsqueeze(0)

        # only need action, logprob, value for runtime
        action, log_prob, _, value = self.conv_net.get_act_and_value(state)

        return action, log_prob, value
    
    def learn(self, next_obs):
        states, actions, log_probs, vals, rewards, dones = self.mem.get_data()

        # we'll use GAE for advantage calc
        advantages = torch.zeros(rewards.shape).to(self.device)
        # returns = torch.zeros(rewards.shape).to(self.device)
        with torch.no_grad():
            last_value = self.conv_net.get_value(next_obs.unsqueeze(0))
            if True:
                last_advantage = 0
                for t in reversed(range(self.buffer_size)):
                    mask = 1.0 - dones[t]
                    last_value = last_value * mask
                    last_advantage = last_advantage * mask

                    delta = rewards[t] + self.discount * last_value - vals[t]
                    advantages[t] = delta + self.discount * self.gae_lambda * last_advantage
                
                    last_value = vals[t]
                    last_advantage = advantages[t]
            elif False:
                last_return = last_value
                for t in reversed(range(self.buffer_size)):
                    mask = 1.0 - dones[t]

                    returns[t] = rewards[t] + self.discount * mask * last_return

                    last_return = returns[t]
            else:
                advantages = np.zeros(len(rewards), dtype=np.float32)
                rewards_mem = rewards.cpu().numpy()
                vals_mem = vals.cpu().numpy()
                dones_mem = dones.cpu().numpy()
                # calc A_t
                for t in range(len(rewards)-1):

                    discount = 1    # (gamma * delta) ^ 0
                    A_t = 0

                    # calc each A_k term
                    for k in range(t, len(rewards)-1):

                        # delta_k = r_k + gamma * V(s_k+1) - V(s_k)
                        delta_k = rewards_mem[k] + self.discount * vals_mem[k + 1] * (1 - int(dones_mem[k])) - vals_mem[k]
                        
                        # add to A_t
                        A_t += discount * delta_k

                        # discount decreases
                        discount *= self.discount * self.gae_lambda

                    # set adv
                    advantages[t] = A_t
                advantages = torch.from_numpy(advantages).to(self.device)

        # We can now begin experience replay
        clip_fracs = []
        for _ in (self.epochs):
            batch_idxs = self.mem.get_batches_idxs()
            # learn from each batch
            for batch in batch_idxs:

                # new tensors for each batch
                b_states = torch.tensor(states[batch]).to(self.device)
                b_actions = torch.tensor(actions[batch]).to(self.device)
                b_log_probs = torch.tensor(log_probs[batch]).to(self.device)
                b_vals = torch.tensor(vals[batch]).to(self.device)
                b_advantages = torch.tensor(advantages[batch]).to(self.device)
                b_returns = b_advantages + b_vals
                # b_returns = torch.tensor(returns[batch]).to(self.device)
                # b_advantages = b_returns - b_vals

                _, new_log_probs, entropy, new_vals = self.conv_net.get_act_and_value(b_states, b_actions.long())

                log_prob_ratio = new_log_probs - b_log_probs
                prob_ratio = log_prob_ratio.exp()

                with torch.no_grad():
                    # using approx kl from http://joschu.net/blog/kl-approx.html to
                    # 1. monitor policy i.e. spikes in kl div might show policy is worsening
                    # 2. early end if approx kl gets bigger than target_kl
                    old_approx_kl = (-log_prob_ratio).mean()
                    approx_kl = ((prob_ratio - 1) - log_prob_ratio).mean()
                    clip_fracs += [((prob_ratio - 1.0).abs() > self.policy_clip).float().mean().item()]

                # note sometimes advantage normalizaiton can lead to empirical benefits
                # in training but it seems it can be harmful sometimes
                if self.norm_advantage:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                
                # (p_theta / p_theta_old) * A_t
                weighted_probs = -b_advantages * prob_ratio
                # clip per paper to avoid too big a change in underlying params
                weighted_clipped_probs = -b_advantages * torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)

                # actor loss (recall loss performed using gradient ascent)
                actor_loss = torch.max(weighted_probs, weighted_clipped_probs).mean()

                # critic loss = mse(return - network_critic_val)
                critic_loss = (b_returns - torch.squeeze(new_vals)) ** 2
                critic_loss = 0.5 * critic_loss.mean()

                # entropy to encourage exploration
                entropy_loss = entropy.mean()

                # compute total loss
                loss = (actor_loss) + (self.critic_coeff * critic_loss) - (self.entropy_coeff * entropy_loss)

                # backprop and descent
                self.optimizer.zero_grad()
                loss.backward()
                if (self.max_grad_norm != None):
                    nn.utils.clip_grad_norm_(self.conv_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler_gamma != None:
                    self.scheduler.step()

            if self.early_stop_kl != None:
                if approx_kl > self.early_stop_kl:
                    break

            # explained variance measures how well the value func matches the returns
            # should be as close to 1 as possible
            y_pred, y_true = b_vals.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # return early stop condition and log metrics
        return {
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'losses/old_approx_kl': old_approx_kl.item(),
            'losses/approx_kl': approx_kl.item(),
            'losses/clip_fracs': np.mean(clip_fracs),
            'losses/critic_loss': critic_loss.item(),
            'losses/actor_loss': actor_loss.item(),
            'losses/entropy_loss': entropy_loss.item(),
            'losses/explained_variance': explained_var,
        }