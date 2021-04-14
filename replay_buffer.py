from abc import abstractmethod
from collections import namedtuple, deque

import numpy as np
import random
import torch

from array_local import Array
from sum_tree import SumTree

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


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

    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        '''Sample a batch of experiences from memory.'''

        indexes, experiences, is_weights = self._get_experices_sample()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(device)

        return indexes, (states, actions, rewards, next_states, dones), is_weights

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

    def batch_update(self, indexes, priorities):
        pass

    def _get_experices_sample(self):
        indexes = np.random.choice(range(len(self.memory)), size=self.batch_size)
        experiences = [self.memory[i] for i in indexes]
        is_weights = np.ones(self.batch_size)

        return (indexes, experiences, is_weights)

    def __len__(self):
        return super().__len__()


class ProportionalReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, buffer_size, e, alpha, beta0, beta_inc, seed):
        '''Initialize a ProportionalReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            e (float): small amount to be added to the priority
            alpha (float): determines how much prioritization is used. 0 = uniform, 1 = fully prioritized.
            beta0 (float): initial beta value. Used to calculate the importance sampling (IS) weights.
            beta_inc (float): value to increment beta0 on after each sampling.
            seed (int): random seed
        '''
        super().__init__(action_size, batch_size, seed)
        self.memory = SumTree(buffer_size, e)
        self.alpha = alpha
        self.beta = beta0
        self.beta_inc = beta_inc

    def batch_update(self, indexes, priorities):
        self.memory.batch_update(indexes, priorities)

    def _get_experices_sample(self):
        range_size = self.memory.sum / self.batch_size
        N = len(self.memory)

        indexes = np.zeros(self.batch_size, dtype=int)
        experiences = np.zeros(self.batch_size, dtype=tuple)
        probabilities = np.zeros(self.batch_size)
        total = 0
        # Importance Sampling (IS) weights.
        is_weights = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            range_start = i * range_size
            range_end = (i + 1) * range_size
            priority_query = np.random.uniform(range_start, range_end)

            index, priority, experience = self.memory.sample(priority_query)
            probability = priority ** self.alpha
            probabilities[i] = probability
            total += probability

            indexes[i] = index
            experiences[i] = experience

        for i in range(self.batch_size):
            probability = probabilities[i] / total
            is_weights[i] = ((1 / N) * (1 / probability)) ** self.beta

        # normalize is_weights so they have range [0, 1].
        is_weights /= is_weights.max()

        # increment value of beta after each sampling up to a maximum of 1.
        self.beta = max(self.beta + self.beta_inc, 1.0)

        return (indexes, experiences, is_weights)


class RankBasedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, buffer_size, buffer_type, alpha, beta0, beta_inc, seed):
        '''Initialize a ProportionalReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            buffer_type (str): determines how the buffer is stored. It can be either 'heap' or 'array'.
            alpha (float): determines how much prioritization is used. 0 = uniform, 1 = fully prioritized.
            beta0 (float): initial beta value. Used to calculate the importance sampling (IS) weights.
            beta_inc (float): value to increment beta0 on after each sampling.
            seed (int): random seed
        '''
        super().__init__(action_size, batch_size, seed)
        self.alpha = alpha
        self.beta = beta0
        self.beta_inc = beta_inc

        if buffer_type == 'array':
            self.memory = Array(buffer_size)
        else:
            raise NotImplementedError()

        self._segments_cache = {}
    
    def batch_update(self, indexes, priorities):
        self.memory.batch_update(indexes, priorities)

    def _get_experices_sample(self):
        segments = self._get_sample_ranges()
        N = len(self.memory)

        indexes = np.zeros(self.batch_size, dtype=int)
        experiences = np.zeros(self.batch_size, dtype=tuple)
        probabilities = np.zeros(self.batch_size)
        # Importance Sampling (IS) weights.
        is_weights = np.zeros(self.batch_size)
        total = 0
        for i in range(self.batch_size):
            segment_start, segment_end = segments[i]
            index = np.random.randint(segment_start, segment_end + 1)
            _, experience = self.memory[index]
            rank = index + 1
            probability = rank ** self.alpha
            probabilities[i] = probability
            total += probability

            indexes[i] = index
            experiences[i] = experience

        for i in range(self.batch_size):
            probability = probabilities[i] / total
            is_weights[i] = ((1 / N) * (1 / probability)) ** self.beta

        # normalize is_weights so they have range [0, 1].
        is_weights /= is_weights.max()

        # increment value of beta after each sampling up to a maximum of 1.
        self.beta = max(self.beta + self.beta_inc, 1.0)

        return (indexes, experiences, is_weights)

    def _get_sample_ranges(self):
        N = len(self.memory)
        cache_key = (N, self.batch_size, self.alpha)

        segments = self._segments_cache.get(cache_key)
        if segments:
            return segments
        else:
            # resets cache since both N and alpha will likely only increase
            # so previous computed values won't be useful.
            self._segments_cache = {}


        total = 0
        probabilities = np.zeros(N)
        for i in range(1, N + 1):
            probabilities[i - 1] = (1 / i) ** self.alpha
            total += probabilities[i - 1]

        probabilities /= total
        cumulative_probs = np.cumsum(probabilities)
        segment_size = 1 / self.batch_size

        segment_start = 0 # the first segment always start at 0
        segment_end_prob = segment_size
        segments = []
        for i in range(len(cumulative_probs)):
            if cumulative_probs[i] > segment_end_prob:
                segments.append((segment_start, i))
                segment_start = i + 1
                segment_end_prob += segment_size

                if len(segments) == self.batch_size - 1:
                    # the last segment will always go until the last element
                    segments.append((segment_start, N - 1))
                    break
        
        assert len(segments) == self.batch_size

        self._segments_cache[cache_key] = segments

        return segments
