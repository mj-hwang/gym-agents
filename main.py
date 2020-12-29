### Training script for DQN Agent in Open AI Atari Environment

import gym
import time
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import tensorflow as tf
from rl.dqn_agent import DQNAgent
from misc.random_agent import RandomAgent
from misc.atari_utils import *


env = gym.make("BreakoutNoFrameskip-v4")
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
env = ClipRewardEnv(env)
env = FrameStack(env, 4)


agent = DQNAgent(env)
agent.train()
agent.save_model(filename="my_model.h5")
