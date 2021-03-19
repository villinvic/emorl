"""
Author : goji .
Date : 18/03/2021 .
File : observe.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
import fire
import gym
from time import sleep
from pprint import pprint

from serializer import Serializer
from population import Individual, Population
from env_utils import name2class
# =============


class PopulationObserver:
    def __init__(self, ckpt, path='checkpoint/', env_id='Boxing-ramDeterministic-v4', slow_factor=0.01):

        self.slow_factor = slow_factor
        self.util = name2class[env_id]
        self.serializer = Serializer(path)
        self.to_observe: Population = self.serializer.load(ckpt)
        self.env = gym.make(env_id)
        self.util = name2class[env_id]
        self.state_shape = (self.util.state_dim * 2,)
        self.action_dim = self.env.action_space.n
        self.player = Individual(self.state_shape, self.action_dim, self.util.goal_dim)

    def observe(self):
        print('--------Population Observation Stared--------')
        for index, individual in enumerate(self.to_observe.individuals):
            print('Individual %d stats :' % index)
            pprint(individual.behavior_stats)
            self.play(index, individual.get_weights())

    def play(self, index, player_weights):
        self.player.set_weights(player_weights)
        try:
            done = False
            observation = self.util.preprocess(self.env.reset())
            observation = np.concatenate([observation, observation])
            while not done:
                self.env.render()
                action = self.player.pi.policy.get_action(observation, eval=True)
                observation_, _, done, _ = self.env.step(action)
                observation_ = self.util.preprocess(observation_)
                observation = np.concatenate([observation[len(observation) // 2:], observation_])
                sleep(self.slow_factor)
        except KeyboardInterrupt:
            print('individual %d skipped' % index)


if __name__ == '__main__':
    fire.Fire(PopulationObserver)



