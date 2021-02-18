""" 
Author : goji .
Date : 29/01/2021 .
File : population.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
from AC import AC
from behavior import *
# =============

class Individual:

    def __init__(self, state_shape, action_dim, sub_goals, epsilon=0, lr=0.001, gamma=0.99, entropy_scale=0,
                 gae_lambda=0, traj_length=1, batch_size=1, neg_scale=1.0):
        self.pi = AC(state_shape, action_dim, epsilon, lr, gamma, entropy_scale, gae_lambda,
                 traj_length, batch_size, neg_scale)
        self.rewardFunc = Reward(sub_goals)

        self.behaviour_stats = {}

        self.age = 0

    def evaluate(self):
        return self.rewardFunc.total()

    def get_weights(self):
        return {'pi': self.pi.get_weights(),
                'r': self.rewardFunc.weights}

    def set_weights(self, w):
        self.pi.set_weights(w['pi'])
        self.rewardFunc.weights = w['r']

class Population:

    def __init__(self, state_shape, action_dim, sub_goals, size):
        self.size = size
        self.individuals = np.empty((size,), dtype=Individual)
        for i in range(size):
            self.individuals[i] = Individual(state_shape, action_dim, sub_goals)
