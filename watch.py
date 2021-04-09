import argparse
import numpy as np

from unityagents import UnityEnvironment

from agent import Agent
from models.utils import get_model

available_models = ['dqn_32', 'dqn_64']

parser = argparse.ArgumentParser(description='Watch a trained model running on the environment.')
parser.add_argument('model', choices=available_models, type=str,
                   help='Which model to use to choose the actions of the agent')
parser.add_argument('-g', '--goal', action='store_true',
                   help='Use the trained model at the moment that it reached the goal instead of after all the episodes')

args = parser.parse_args()

model = get_model(args.model)

env = UnityEnvironment(file_name='Banana_Windows_x86_64/Banana.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
action_size = brain.vector_action_space_size
state_size = len(state)

agent = Agent(state_size, action_size, model, None, seed=0)
agent.qnetwork_local.load_checkpoint(args.model, at_goal=args.goal)

def step_env(action):
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    return (next_state, reward, done)

score = 0                                          # initialize the score
done = False
while not done:
    action = agent.act(state)                      # select an action
    env_info = env.step(action)[brain_name]        # execute the chosen action
    state = env_info.vector_observations[0]        # update current state
    reward = env_info.rewards[0]                   # the reward (s, a) => r
    done = env_info.local_done[0]                  # if the episode is done
    score += reward                                # update the score

print('Score: {}'.format(score))