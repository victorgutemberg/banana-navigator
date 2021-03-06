import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    '''Interacts with and learns from the environment.'''

    def __init__(self, state_size, action_size, model, memory, seed):
        '''Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network.
        self.qnetwork_local = model(state_size, action_size, seed).to(device)
        self.qnetwork_target = model(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # replay memory.
        self.memory = memory
        # initialize time step (for updating every UPDATE_EVERY steps).
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory.
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn.
            if len(self.memory) > BATCH_SIZE:
                indexes, experiences, is_weights = self.memory.sample()
                self.learn(indexes, experiences, is_weights)

    def act(self, state, eps=0.):
        '''Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon-greedy action selection.
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, indexes, experiences, is_weights, gamma=GAMMA):
        '''Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        # get max predicted Q values (for next states) from target model.
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # compute Q targets for current states.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # get expected Q values from local model.
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # compute new priorities.
        priorities = abs(Q_expected - Q_targets).detach().numpy()
        self.memory.batch_update(indexes, priorities)

        # compute loss.
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        # minimize the loss.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
