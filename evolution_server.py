""" 
Author : goji .
Date : 29/01/2021 .
File : evolution_server.py .

Description : None

Observations : None
"""

# == Imports ==
from population import Individual
from behavior import *

import zmq
import numpy as np
import signal
import sys
from copy import deepcopy
import gym
import fire
# =============

class EvolutionServer:

    def __init__(self, ID, env_id='Assault-ram-v0', traj_length=10000, batch_size=1):
        self.ID = ID
        self.env = gym.make(env_id)
        self.state_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.sub_goals = ['win', 'destructed']
        self.indicators = Indicator(dummy)
        self.player = Individual(self.state_shape, self.action_dim, self.sub_goals, traj_length=self.traj_length)

        context = zmq.Context()
        self.mating_pipe = context.socket(zmq.PULL)
        self.mating_pipe.connect("ipc://MATING")
        self.evolved_pipe = context.socket(zmq.PUSH)
        self.evolved_pipe.connect("ipc://EVOLVED")

        self.traj_length = traj_length
        self.batch_size = batch_size

        self.trajectory = {
            'state': np.zeros((batch_size,traj_length)+self.env.observation_space.shape, dtype=np.float32),
            'action': np.zeros((batch_size, self.traj_length), dtype=np.int32),
            'rew': np.zeros((batch_size, self.traj_length,), dtype=np.float32),
        }


        signal.signal(signal.SIGINT, self.exit)

    def exit(self, signal_num, frame):
        self.mating_pipe.close()
        self.evolved_pipe.close()
        sys.exit(0)

    def recv_mating(self):
        return self.mating_pipe.recv_pyobj()

    def send_evolved(self, q):
        self.evolved_pipe.send_pyobj(q)

    def SBX_beta(self, n):
        beta = np.random.randint(0 , 5)
        if beta <= 1.0:
            return 0.5 * (n+1) * (beta ** n)
        else:
            return 0.5 * (n+1) / (beta ** (n+2))

    def crossover(self, mating):
        offspring = np.empty_like(mating)
        for i in range(len(mating)-1, step=2):
            p1 = mating[i]
            p2 = mating[i+1]


            # SPX for NN
            q1 = deepcopy(p1)
            q2 = deepcopy(p1)
            for j in range(len(p1['pi'])):
                point = np.random.randint(0, len(p1['pi'][j]))
                q1['pi'][j][:point] = p2['pi'][j][:point]
                q2['pi'][j][point:] = p2['pi'][j][point:]
            # SBX for reward
            beta = self.SBX_beta(5)
            x = 0.5 * (p1['r'] + p2['r'])
            q1['r'] = x - 0.5 * beta * (np.abs(p1['r'] - p2['r']))
            q2['r'] = x + 0.5 * beta * (np.abs(p1['r'] - p2['r']))

            offspring[i] = q1
            offspring[i+1] = q2

        return offspring

    def mutate(self, offspring):
        for q in offspring:
            for i in range(len(q['pi'])):
                gaussian_noise = np.random.normal(size=q['pi'][i])
                q['pi'][i] += gaussian_noise

            gaussian_noise = np.random.normal(size=q['r'])
            q['r'] += gaussian_noise

    def play(self, player:Individual, max_frame):
        r = {
            'game_reward': 0,
            'avg_length': np.nan,
            'total_punition': 0,
            'info': {}
        }
        frame_count = 0
        n_games = 0
        while frame_count < max_frame:
            done = False
            obs = self.env.reset()
            while not done:
                action = player.pi.get_action(obs)
                observation, reward, done, info = self.env.step(action)
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward
                self.trajectory['state'][0, frame_count] = observation
                self.trajectory['action'][0, frame_count] = action
                self.trajectory['rew'][0, frame_count] = reward
                frame_count += 1
            n_games += 1

        r['avg_length'] = max_frame / float(n_games+1)
        return r

    def DRL(self, offspring):
        trained = np.empty_like(offspring)
        for i, q in enumerate(offspring):
            self.player.set_weights(q) # sets nn and r weights
            self.play(self.player, self.traj_length)
            self.player.pi.train(self.trajectory['state'], self.trajectory['action'][:, :-1], self.trajectory['rew'][:, 1:])
            trained[i] = {'weights': self.player.get_weights()}
        return trained

    def evaluate(self, trained):
        for individual in trained:
            self.player.set_weights(individual)  # sets nn and r weights
            trained['eval'] = self.play(self.player, self.traj_length)


    def run(self):
        while True:
            mating = self.recv_mating()
            qs = self.crossover(mating)
            self.mutate(qs)

            trained = self.DRL(qs)
            self.evaluate(trained)

            self.send_evolved(trained)

def RUN(ID, env_id='Assault-ram-v0'):
    server = EvolutionServer(ID, env_id)
    server.run()
    sys.exit(0)

if __name__ == '__main__':
    fire.Fire(RUN)