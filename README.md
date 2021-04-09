[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/3620840/114130087-ae419180-98b4-11eb-97d6-e3469d99ee5a.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Project description

This project uses deep reinforcement learning to teach an agent to navigate on a virtual world filled with good (yellow) and rotten (blue) bananas. 

This repo is a fork of the Udacity's deep reinforcement learning [`p1_navigation`](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) repo.

To train a new model, use the `Navigation.ipynb` notebook. There is another notebook file named `Report.ipybn` where a comparison between trained models is done.

## Environment

The environment consists of a large, square world filled with bananas. The agent should learn how to collect the yellow bananas (reward of +1) while avoiding the blue bananas (reward of -1).

![Trained Agent][image1]

### State

The agent's state has 37 dimentions and it is composed of its speed along with ray-based perception of objects around the agent's forward direction.

### Actions

At any given state, the agent has the four discrete actions to take. It can move forward, backward, left or right.

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Rewards

The agent receives a reward of +1 for every yellow banana collected and -1 if a blue banana is collected.

## Running the project

### Installing dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:

    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

    - __Windows__:

    ```bash
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. This project uses Unity to emulate the environment in which the agent takes actions. To run it, you will need to download the environment that matches the operating system you are using.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Linuz (no visualization): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

 After downloading the environment, extract the zip file anywhere and update the `Navigation.ipynb` file with the path to the environment folder.

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

### Trained models

There are three Neural Networks available in the `models` folder. The input layer of all of the models has the same size of the state space and the ouput layer has the size of the actions space. All of them have 2 hidden layers with sizes described by `model_name` below:

- dqn_32: 64x32
- dqn_64: 64x64
- dqn_128: 128x64

In the `checkpoints` folder, you can find a trained checkpoint for all models. For each model there are three files saved:

- checkpoint_{model_name}_goal.pth: a pytorch checkpoint that can be loaded into the corresponding model. This checkpoint is saved the first time the model reaches the goal.
- checkpoint_{model_name}.pth: same as the above but saved at the end of the training. The models available were trained for 2000 episodes.
- score_{model_name}.pkl: a pickled array containing the scores reached by the model while training per episode.

To watch a trained model executing actions on the environment use the `watch.py` script. Use `-h` or `--help` to see how to use the tool.

usage: watch.py [-h] [-g] {dqn_32,dqn_64}

Watch a trained model running on the environment.

positional arguments:
  {dqn_32,dqn_64}  Which model to use to choose the actions of the agent

optional arguments:
  -h, --help       show this help message and exit
  -g, --goal       Use the trained model at the moment that it reached the
                   goal instead of after all the episodes