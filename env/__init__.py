from gym.envs.registration import register
import numpy as np


register(
    id='CartPoleCont-v0',
    entry_point='env.cartpolecont:CartPoleContEnv')
