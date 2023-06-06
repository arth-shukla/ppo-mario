import torch
import numpy as np

from .memory import PPOExperience
from .model import PPOActor, PPOCritic

class PPOAgent():
    # defaults to 'ideal' values from https://arxiv.org/pdf/1707.06347.pdf
    def __init__(
            self,
            # model architecture
            obs_shape, act_shape, act_n, embed=512,
            # memory info
            buffer_size=2048, batch_size=64, epochs=10,
            # advantage calc
            discount=0.99, gae_lambda=0.95, policy_clip=0.2, norm_advantage=False,
            # loss calc
            entropy_coeff=0.01, critic_coeff=0.5, max_grad_norm=0.5, early_stop_kl=None,
            # optimizer params
            lr=2.5e-4, sch_end_f=None,

        ):
        
        # model architecture
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_n = act_n
        self.embed = embed

        # memory info
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs

        # advantage calc
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.norm_advantage = norm_advantage

        # loss calc
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.max_grad_norm = max_grad_norm
        self.early_stop_kl = early_stop_kl

        # optimizer params
        self.lr = lr
        self.sch_end_f = sch_end_f

        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create actor, critic, and memory
        self.actor = PPOActor(act_n, obs_shape, lr, embed=embed, sch_end_f=sch_end_f).to(self.device)
        self.critic = PPOCritic(obs_shape, lr, embed=embed, sch_end_f=sch_end_f).to(self.device)
        self.memory = PPOExperience(batch_size, buffer_size=buffer_size)


    def act(self, obs):
        state = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(self.device)

        # get action from actor
        categorical_dist = self.actor(state)
        action = categorical_dist.sample()
        
        # get approx value from critic
        val = self.critic(state)

        # get probability of action based on policy, action, and val
        prob = torch.squeeze(categorical_dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        val = torch.squeeze(val).item()

        return action, prob, val
    
    approx_kl = None
    entropy_loss = None
    def learn(self, next_state):
        next_state = torch.tensor(next_state).unsqueeze(0).to(self.device)
        next_state_val = self.critic(next_state)
        for _ in range(self.epochs):
            state_mem, action_mem, probs_mem, vals_mem, rewards_mem, dones_mem, batch_idxs = self.memory.gen_batches()

            advantage = np.zeros(len(rewards_mem), dtype=np.float32)
            # calc A_t
            for t in range(len(rewards_mem)):

                discount = 1    # (discount * delta) ^ 0
                A_t = 0

                # calc each A_k term
                for k in range(t, len(rewards_mem)):

                    if k == len(rewards_mem) - 1:
                        next_val = next_state_val
                    else:
                        next_val = vals_mem[k + 1]

                    # delta_k = r_k + discount * V(s_k+1) - V(s_k)
                    delta_k = rewards_mem[k] + self.discount * next_val * (1 - int(dones_mem[k])) - vals_mem[k]
                    
                    # add to A_t
                    A_t += discount * delta_k

                    # discount decreases
                    discount *= self.discount * self.gae_lambda

                # set adv
                advantage[t] = A_t

            # convert to tensors for processing
            advantage = torch.tensor(advantage).to(self.device)
            vals = torch.tensor(vals_mem).to(self.device)

            # note sometimes advantage normalizaiton can lead to empirical benefits
            # in training but it seems it can be harmful sometimes
            if self.norm_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for batch in batch_idxs:

                # convert to tensors for processing
                states = torch.tensor(state_mem[batch], dtype=torch.float32).to(self.device)
                old_probs = torch.tensor(probs_mem[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(action_mem[batch], dtype=torch.float32).to(self.device)

                # get actor and critic outputs
                categorical_dist = self.actor(states)
                critic_val = torch.squeeze(self.critic(states))

                # get probs from categorical distribution
                new_probs = categorical_dist.log_prob(actions)

                # (p_theta / p_theta_old) * A_t
                log_prob_ratio = new_probs - old_probs
                prob_ratio = log_prob_ratio.exp()
                weighted_probs = advantage[batch] * prob_ratio


                # using approx kl from http://joschu.net/blog/kl-approx.html to
                # 1. monitor policy i.e. spikes in kl div might show policy is worsening
                # 2. early end if approx kl gets bigger than target_kl
                with torch.no_grad():
                    approx_kl = ((prob_ratio - 1) - log_prob_ratio).mean()


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
                total_loss = actor_loss + self.critic_coeff * critic_loss


                # entropy can help force the model to explore more, which can help prevent the 
                # agent from converging to an unhelpful solution by encouraging exploration
                entropy = categorical_dist.entropy()
                entropy_loss = entropy.mean()
                if self.entropy_coeff != None:
                    total_loss -= self.entropy_coeff * entropy_loss


                # zero out grads
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # backprop
                total_loss.backward()

                # grad norm clip can help reduce model perturbations and avoid
                # harmful shifts in the policy space, but if too aggressive can
                # also harm learning or require smaller learning rates
                if self.max_grad_norm != None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                # descent steps on each network
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            if self.sch_end_f != None:
                self.actor.scheduler.step()
                self.critic.scheduler.step()

        log_dict = {
            'losses/actor_loss': actor_loss.item(),
            'losses/critic_loss': critic_loss.item(),
            'losses/approx_kl': approx_kl.item(),
            'losses/entropy': entropy_loss.item(),
        }
        if self.sch_end_f:
            log_dict['charts/actor_learning_rate'] = self.actor.scheduler.get_last_lr()[0]
            log_dict['charts/critic_learning_rate'] = self.critic.scheduler.get_last_lr()[0]

        return log_dict


    # ------------------------------------------------------------------------------------------
    # memory funcs 
    # --------------------------------------------------------------------------------
    def cache(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    def clear_mem(self):
        self.memory.clear_memory()
    def mem_full(self):
        return len(self.memory) == self.buffer_size
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # checkpoint funcs
    # --------------------------------------------------------------------------------
    def save(self, save_path='model_checkpoint.pt'):

        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
        }

        if self.sch_end_f:
            save_dict['actor_scheduler'] = self.actor.scheduler.state_dict()
            save_dict['critic_scheduler'] = self.critic.scheduler.state_dict()

        torch.save(save_dict, save_path)

    def load(self, load_path='model_checkpoint.pt'):
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])

        if self.sch_end_f:
            self.actor.scheduler.load_state_dict(checkpoint['actor_scheduler'])
            self.critic.scheduler.load_state_dict(checkpoint['critic_scheduler'])
    # ------------------------------------------------------------------------------------------