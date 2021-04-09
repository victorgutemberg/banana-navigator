from models.q_network import QNetwork as BaseQNetwork

class QNetwork(BaseQNetwork):
    '''Actor (Policy) Model.'''

    def __init__(self, state_size, action_size, seed):
        '''Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        '''
        super().__init__(state_size, action_size, seed, fc1_units=64, fc2_units=128)