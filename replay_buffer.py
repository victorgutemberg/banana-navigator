from abc import abstractmethod
from collections import namedtuple, deque

import numpy as np
import random
import torch

from sum_tree import SumTree

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

to_experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority):
        '''Add a new experience to memory.'''
        experience = to_experience(state, action, reward, next_state, done, priority)
        self.memory.append(experience)

    def sample(self):
        '''Sample a batch of experiences from memory.'''

        indexes, experiences, weights = self._get_experices_sample()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        indexes = torch.from_numpy(np.vstack(indexes)).int().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        return (states, actions, rewards, next_states, dones, indexes, weights)

    @abstractmethod
    def _get_experices_sample(self,):
        raise NotImplementedError()

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, buffer_size, seed):
        '''Initialize a UniformReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        '''
        super().__init__(action_size, batch_size, seed)
        self.memory = deque(maxlen=buffer_size)

    def _get_experices_sample(self):
        indexes = np.random.choice(range(len(self.memory)), size=self.batch_size)
        experiences = [self.memory[i] for i in indexes]
        weights = np.ones(self.batch_size)

        return (indexes, experiences, weights)

    def __len__(self):
        return super().__len__()


class ProportionalReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, buffer_size, alpha, beta0, beta_inc, seed):
        '''Initialize a ProportionalReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            alpha (float): determines how much prioritization is used. 0 = uniform, 1 = fully prioritized.
            beta0 (float): initial beta value. Used to calculate the importance sampling (IS) weights.
            beta_inc (float): value to increment beta0 on after each sampling.
            seed (int): random seed
        '''
        super().__init__(action_size, batch_size, seed)
        self.memory = SumTree(buffer_size)
        self.alpha = alpha
        self.beta = beta0
        self.beta_inc = beta_inc

    def _get_experices_sample(self):
        range_size = self.memory.sum / self.batch_size
        total = self.memory.sum
        N = len(self.memory)

        indexes = np.zeros(self.batch_size)
        experiences = np.zeros(self.batch_size)
        weights = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            range_start = i * range_size
            range_end = (i + 1) * range_size
            priority = np.random.uniform(range_start, range_end)

            index, experience = self.memory.sample(priority)
            probability = (experience.priority / total) ** self.alpha

            indexes[i] = index
            experiences[i] = experience
            weights[i] = ((1 / N) * (1 / probability)) ** self.beta

        # normalize weights so they have range [0, 1].
        max_weight = weights.max()
        weights /= max_weight

        # increment value of beta after each sampling up to a maximum of 1.
        self.beta = max(self.beta + self.beta_inc, 1.0)

        return (indexes, experiences, weights)
