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
        self.state_mem = []
        self.action_mem = []
        self.prob_mem = []
        self.val_mem = []
        self.reward_mem = []
        self.done_mem = []

        self.end_pointer = 0

    # store experiences
    def cache(self, state, action, prob, val, reward, done):

        # save torches in memory
        self.state_mem.append(state)
        self.action_mem.append(action)
        self.prob_mem.append(prob)
        self.val_mem.append(val)
        self.reward_mem.append(reward)
        self.done_mem.append(done)

        # don't need to worry about % since recall and clear_mem
        # should be called as soon as tensors are full anyways
        self.end_pointer += 1

    def get_data(self):
        return np.array(self.state_mem), np.array(self.action_mem), \
            np.array(self.prob_mem), np.array(self.val_mem), \
            np.array(self.reward_mem), np.array(self.done_mem)

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
