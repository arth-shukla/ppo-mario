import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
import numpy as np

import os

class PPOExperience():
    def __init__(self, batch_size: int, mem_max: int = 100):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

        self.mem_max = mem_max
        self.newest_mem_idx = -1

    def gen_batches(self):
        
        mem_len = len(self.states)

        # get all possible starting indices of batch size
        # e.g. memlen 20, batch_size 5 => [0, 5, 10, 15]
        batch_starts = np.arange(0, mem_len, self.batch_size)
        
        # shuffle indices so that we get random memories in each batch
        idxs = np.arange(mem_len, dtype=np.int64)
        np.random.shuffle(idxs)

        # collections of indices w/o repeat for each starting value
        batch_idxs = [idxs[start : start + self.batch_size] for start in batch_starts]

        # return batch indices since we'll iterate over these in usage
        return  np.array(self.states), np.array(self.actions), np.array(self.probs), \
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), \
                batch_idxs
    
    def store_memory(self, state, action, prob, val, reward, done):

        mem_len = len(self.states)

        # cycle through array, replacing older memories as we go
        self.newest_mem_idx += 1
        self.newest_mem_idx %= self.mem_max
        
        # want to cap memory since too short or too long memory is bad
        if mem_len > self.mem_max:

            # place items in place of oldest memory currently stored
            self.states[self.newest_mem_idx] = state
            self.actions[self.newest_mem_idx] = action
            self.probs[self.newest_mem_idx] = prob
            self.vals[self.newest_mem_idx] = val
            self.rewards[self.newest_mem_idx] = reward
            self.dones[self.newest_mem_idx] = done
        else:

            # append memory to storage
            self.states.append(state)
            self.actions.append(action)
            self.probs.append(prob)
            self.vals.append(val)
            self.rewards.append(reward)
            self.dones.append(done)

    def clear_memory(self):

        # reset each list
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class PPOActor(nn.Module):
    def __init__(self, num_actions, input_dims, lr, fc_dims=(256, 256), checkpoint_dir='checkpoints/ppo', device='cpu'):
        super(PPOActor, self).__init__()

        # send to device (easier to handle at top level)
        self.device=device
        self.to(device)

        # save checkpoints in case smth happens or I wanna try again with a baseline
        self.checkpoint_file = os.path.join(checkpoint_dir, 'ppo_actor')


        # ------------------------------------------------------
        # define layers

        # action space to embedding space
        self.fc1 = nn.Linear(*input_dims, fc_dims[0])
        # call relu directly

        # another hidden layer
        self.fc2 = nn.Linear(*fc_dims)
        # call relu directly

        # final output layer
        self.out = nn.Linear(fc_dims[1], num_actions)
        self.softmax = nn.Softmax(dim=-1)

        # want to sample from categorical distr based on the 
        # above learned outputs
        # categorical called directly

        # -------------------------------------------------------

        # keep optimizer internal since we're keeping the
        # actor and critic separate
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, input):

        # action space to embedding
        x = nn.functional.relu(self.fc1(input))

        # hidden layers
        x = nn.functional.relu(self.fc2(x))

        # output prob for each action
        x = self.softmax(self.out(x))

        # create categorical distr to sample 
        # from for our next action
        x = Categorical(x)

        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPOCritic(nn.Module):
    def __init__(self, input_dims, lr, fc_dims=(256, 256), checkpoint_dir='checkpoints/ppo', device='cpu'):
        super(PPOCritic, self).__init__()

        # send to device (easier to handle at top level)
        self.to(device)

        # save checkpoints in case smth happens or I wanna try again with a baseline
        self.checkpoint_file = os.path.join(checkpoint_dir, 'ppo_checkpoint')
        

        # ------------------------------------------------------
        # define layers

        # action space to embedding space
        self.fc1 = nn.Linear(*input_dims, fc_dims[0])
        # call relu directly

        # another hidden layer
        self.fc2 = nn.Linear(*fc_dims)
        # call relu directly

        # final output layer
        self.out = nn.Linear(fc_dims[1], 1)

        # ------------------------------------------------------


        # keep optimizer internal since we're keeping the
        # actor and critic separate
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, input):

        # action space to embedding
        x = nn.functional.relu(self.fc1(input))

        # hidden layers
        x = nn.functional.relu(self.fc2(x))

        # output approximated value function
        # given the current policy
        x = self.out(x)

        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPOAgent():
    # defaults to 'ideal' values from https://arxiv.org/pdf/1707.06347.pdf
    def __init__(self, num_actions, input_dims, gamma=0.99, lr=0.003, smoothing_lambda=0.95, policy_clip=0.2, batch_size=64, horizon=2048, num_epochs=10, mem_max=100):
        
        # hyperparams
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs
        self.smoothing_lambda = smoothing_lambda

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # create actor, critic, and memory
        self.actor: PPOActor = PPOActor(num_actions, input_dims, lr, device=self.device)
        self.critic: PPOCritic = PPOCritic(input_dims, lr, device=self.device)
        self.memory = PPOExperience(batch_size, mem_max=mem_max)

    # boiler plate building on child functions
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def clear_memory(self):
        self.memory.clear_memory()

    def save_models(self, silent=False):
        if not silent: print('Saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, silent=False):
        if not silent: print('Loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, obs):
        state = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)

        # get action from actor
        categorical_dist = self.actor(state)
        action = categorical_dist.sample()
        
        # get approx value from critic
        val = self.critic(state)

        # get probability of action based on policy, action, and val
        probs = torch.squeeze(categorical_dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        val = torch.squeeze(val).item()

        return action, probs, val
    
    def learn(self):
        for _ in range(self.num_epochs):
            state_mem, action_mem, probs_mem, vals_mem, rewards_mem, dones_mem, batch_idxs = self.memory.gen_batches()

            advantage = np.zeros(len(rewards_mem), dtype=np.float32)
            # calc A_t
            for t in range(len(rewards_mem)-1):

                discount = 1    # (gamma * delta) ^ 0
                A_t = 0

                # calc each A_k term
                for k in range(t, len(rewards_mem)-1):

                    # delta_k = r_k + gamma * V(s_k+1) - V(s_k)
                    delta_k = rewards_mem[k] + self.gamma * vals_mem[k + 1] * (1 - int(dones_mem[k])) - vals_mem[k]
                    
                    # add to A_t
                    A_t += discount * delta_k

                    # discount decreases
                    discount *= self.gamma * self.smoothing_lambda

                # set adv
                advantage[t] = A_t

            # convert to tensors for processing
            advantage = torch.tensor(advantage).to(self.device)
            vals = torch.tensor(vals_mem).to(self.device)


            for batch in batch_idxs:

                # convert to tensors for processing
                states = torch.tensor(state_mem[batch], dtype=torch.float32).to(self.device)
                old_probs = torch.tensor(probs_mem[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_mem[batch], dtype=torch.float32).to(self.device)

                # get actor and critic outputs
                categorical_dist = self.actor(states)
                critic_val = torch.squeeze(self.critic(states))

                # get probs from vategorical distribution
                new_probs = categorical_dist.log_prob(actions)

                # (p_theta / p_theta_old) * A_t
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio

                # clip per paper to avoid too big a change in underlying params
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

                # actor loss performed using gradient ascent
                actor_loss = -1 * torch.min(weighted_probs, weighted_clipped_probs).mean()

                # return = advantage + memory_critic_val
                # critic loss = mse(return - network_critic_val)
                returns = advantage[batch] + vals[batch]
                critic_loss = (returns - critic_val) ** 2
                critic_loss = critic_loss.mean()

                # note total loss is + 0.5 since we need gradient ascent
                # also, since we're using two separate networks for
                # critic and loss, we don't need the entropy term
                total_loss = actor_loss + 0.5 * critic_loss

                # zero out grads
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # backprop
                total_loss.backward()

                # descent steps on each network
                self.actor.optimizer.step()
                self.critic.optimizer.step()