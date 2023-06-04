import torch
import torch.nn.functional as F
import numpy as np

class Memory:
    def __init__(self, buffer_size, batch_size, obs_shape, act_shape):

        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input dim vars
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.act_shape = act_shape

        # init mem and tracking vars
        self.clear_mem()

    # reset torch arrays for storing memories
    def clear_mem(self):
        self.state_mem = torch.zeros(self.buffer_size, *self.obs_shape)
        self.action_mem = torch.zeros(self.buffer_size, *self.act_shape)
        self.prob_mem = torch.zeros(self.buffer_size, 1)
        self.val_mem = torch.zeros(self.buffer_size, 1)
        self.reward_mem = torch.zeros(self.buffer_size, 1)
        self.done_mem = torch.zeros(self.buffer_size, 1)

        self.end_pointer = 0

    # store experiences
    def cache(self, state, action, prob, val, reward, done):

        # save torches in memory
        self.state_mem[self.end_pointer] = state.float()
        self.action_mem[self.end_pointer] = action.float()
        self.prob_mem[self.end_pointer] = prob.float()
        self.val_mem[self.end_pointer] = val.float()
        self.reward_mem[self.end_pointer] = reward.float()
        self.done_mem[self.end_pointer] = done.float()

        # don't need to worry about % since recall and clear_mem
        # should be called as soon as tensors are full anyways
        self.end_pointer += 1

    def get_data(self):

        # send to device
        self.state_mem.to(self.device)
        self.action_mem.to(self.device)
        self.prob_mem.to(self.device)
        self.val_mem.to(self.device)
        self.reward_mem.to(self.device)
        self.done_mem.to(self.device)

        return self.state_mem, self.action_mem, self.prob_mem, self.val_mem, self.reward_mem, self.done_mem

    # sample a batch of experiences from memory
    def get_batches_idxs(self):
        
        # get all possible starting indices of batch size
        # e.g. memlen 20, batch_size 5 => [0, 5, 10, 15]
        batch_starts = np.arange(0, self.buffer_size, self.batch_size)
        
        # shuffle indices so that we get random memories in each batch
        idxs = np.arange(self.buffer_size, dtype=np.int64)
        np.random.shuffle(idxs)

        # collections of indices w/o repeat for each starting value
        batch_idxs = [idxs[start : start + self.batch_size] for start in batch_starts]
        
        return batch_idxs
    
    def __len__(self):
        return len(self.state_mem)


if __name__ == '__main__':
    obs_shape = (4, 84, 84)
    act_shape = (1,)
    act_n = 12
    mem = Memory(2048, 32, (4, 84, 84), act_shape)

    state = torch.rand(*(4, 84, 84))
    action = torch.randint(low=0, high=act_n, size=(1,))
    prob = torch.rand(1)
    val = torch.rand(1)
    reward = torch.rand(1)
    done = torch.rand(1) > 0.1

    mem.cache(state, action, prob, val, reward, done)

    state_mem, action_mem, prob_mem, val_mem, reward_mem, done_mem, batch_idxs = mem.recall()

    for x in [state_mem, action_mem, prob_mem, val_mem, reward_mem, done_mem]:
        print(x.shape)
    print(len(batch_idxs))

    print(state_mem[0].shape, action_mem[0])
