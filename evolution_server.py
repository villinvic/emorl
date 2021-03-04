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
from env_utils import *

import zmq
import numpy as np
import signal
import sys
from copy import deepcopy
import gym
from time import time
import matplotlib.pyplot as plt
# =============

class EvolutionServer:

    def __init__(self, ID, env_id='Pong-ram-v0', traj_length=256, batch_size=1, max_train=5, early_stop=7,
                 round_length=100, eval_length=2000, subprocess=True, mutation_rate=0.01):
        self.ID = ID
        self.env = gym.make(env_id)
        self.env.frame_skip = 4
        self.util = name2class[env_id]()
        self.state_shape = (self.util.state_dim*2,)
        self.action_dim = self.env.action_space.n
        self.indicators = Indicator(dummy)
        self.mutation_rate = mutation_rate
        self.player = Individual(self.state_shape, self.action_dim, self.util.goal_dim, traj_length=traj_length)

        if subprocess:
            context = zmq.Context()
            self.mating_pipe = context.socket(zmq.PULL)
            self.mating_pipe.connect("ipc://MATING")
            self.evolved_pipe = context.socket(zmq.PUSH)
            self.evolved_pipe.connect("ipc://EVOLVED")

        self.traj_length = traj_length
        self.max_train = max_train
        self.early_stop = early_stop
        self.round_length = round_length
        self.batch_size = batch_size
        self.eval_length = eval_length

        self.trajectory = {
            'state': np.zeros((batch_size, traj_length)+self.state_shape, dtype=np.float32),
            'action': np.zeros((batch_size, self.traj_length), dtype=np.int32),
            'rew': np.zeros((batch_size, self.traj_length), dtype=np.float32),
        }

        if subprocess:
            signal.signal(signal.SIGINT, self.exit)

    def exit(self, signal_num, frame):
        self.mating_pipe.close()
        self.evolved_pipe.close()
        print('[%d] closed' % self.ID)
        sys.exit(0)

    def recv_mating(self):
        return self.mating_pipe.recv_pyobj()

    def send_evolved(self, q):
        self.evolved_pipe.send_pyobj(q)

    def SBX_beta(self, n):
        beta = np.random.uniform(0, 5)
        if beta <= 1.0:
            return 0.5 * (n+1) * (beta ** n)
        else:
            return 0.5 * (n+1) / (beta ** (n+2))

    def crossover(self, mating):
        offspring = np.empty_like(mating)
        for i in range(0, len(mating)-1, 2):
            p1 = mating[i]
            p2 = mating[i+1]


            # SPX for NN
            q1 = deepcopy(p1)
            q2 = deepcopy(p1)
            s = 32*32 * 2 +32*2 + 32*6 + 6 + 32 + 1 # 33927 128×128 × 2 +128×2 + 128×6 + 6 + 128 + 1
            c = 0
            point = np.random.randint(0, s)
            for j in range(len(p1['pi'])):
                if isinstance(p1['pi'][j], np.ndarray) and len(p1['pi'][j] > 0):
                    if isinstance(p1['pi'][j][0], np.ndarray) and len(p1['pi'][j] > 0):
                        for k in range(len(p1['pi'][j])):
                            if c + len(p1['pi'][j][k]) > point and c < point:
                                q1['pi'][j][k][:point-c] = p2['pi'][j][k][:point-c]
                                q2['pi'][j][k][point-c:] = p2['pi'][j][k][point-c:]
                            elif c < point :
                                q1['pi'][j][k] = p2['pi'][j][k]
                            else:
                                q2['pi'][j][k] = p2['pi'][j][k]
                            c += len(p1['pi'][j][k])
                    else:
                        if c + len(p1['pi'][j]) > point and c < point:
                            q1['pi'][j][:point-c] = p2['pi'][j][:point-c]
                            q2['pi'][j][point-c:] = p2['pi'][j][point-c:]
                        elif c < point :
                            q1['pi'][j] = p2['pi'][j]
                        else:
                            q2['pi'][j] = p2['pi'][j]
                        c += len(p1['pi'][j])
               
            # SBX for reward
            beta = self.SBX_beta(2)
            x = 0.5 * (p1['r'] + p2['r'])
            q1['r'] = x - 0.5 * beta * (np.abs(p1['r'] - p2['r']))
            q2['r'] = x + 0.5 * beta * (np.abs(p1['r'] - p2['r']))

            offspring[i] = q1
            offspring[i+1] = q2
        return offspring

    def mutate(self, offspring, intensity=0.005):
        for q in offspring:
            if np.random.random() < self.mutation_rate:
                for i in range(len(q['pi'])):
                    if isinstance(q['pi'][i], np.ndarray) and len(q['pi'][i] > 0):
                        gaussian_noise = np.random.normal(loc=0, scale=intensity, size=q['pi'][i].shape)
                        q['pi'][i] += gaussian_noise

                gaussian_noise = np.random.normal(loc=0, scale=0.02, size=q['r'].shape)
                q['r'] = np.clip(q['r'] + gaussian_noise, 0, 1)# np.clip(q['r'] * (1  + gaussian_noise), 0, 1)

    def eval(self, player: Individual, max_frame):
        r = {
            'game_reward': 0.0,
            'avg_length': 0.0,
            'total_punition': 0.0,
            'no_op_rate': 0.0,
            'move_rate': 0.0,
            'win_rate': 0.0,
            'entropy': 0.0,
        }
        frame_count = 0
        n_games = 0
        last_pos = 0.0
        actions = [0] * 6
        dist = np.zeros((self.action_dim,), dtype=np.float32)
        while frame_count < max_frame:
            done = False
            observation = self.util.preprocess(self.env.reset())
            observation = np.concatenate([observation, observation])
            while not done and frame_count < max_frame:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                observation_, reward, done, info = self.env.step(action)
                observation_ = self.util.preprocess(observation_)
                observation = np.concatenate([observation[len(observation)//2:],observation_])
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward
                r['no_op_rate'] += int(self.util.is_no_op(action))
                distance_moved = self.util.pad_move(observation_, last_pos)

                moved = int(distance_moved > 1e-5)

                r['move_rate'] += moved
                last_pos = observation_[4]

                frame_count += 1
            if done:
                n_games += 1

        print(actions)
        r['avg_length'] = max_frame / float(n_games + 1)
        r['win_rate'] = (np.abs(r['game_reward'] - r['total_punition'])) / float(np.abs(r['game_reward'] - 2 * r['total_punition']))
        r['no_op_rate'] = r['no_op_rate'] / float(max_frame)
        r['move_rate'] = r['move_rate'] / float(max_frame)
        dist /= float(max_frame)
        r['entropy'] = -np.sum(np.log(dist+1e-8) * dist)
        return r

    def play(self, player: Individual, max_frame, observation=None):
        frame_count = 0
        n_games = 0
        last_pos = 0.0
        last_score_delta = 0
        actions = [0]*6
        while frame_count < max_frame:
            done = False
            if observation is None:
                observation = self.util.preprocess(self.env.reset())
                observation = np.concatenate([observation, observation])
                
            while not done and frame_count < max_frame:
                action = player.pi.policy.get_action(observation)
                actions[action] += 1
                observation_, reward, done, info = self.env.step(action)
                observation_ = self.util.preprocess(observation_)
                distance_moved = self.util.pad_move(observation_, last_pos)

                moved = int(distance_moved > 1e-5)
                last_pos = observation_[4]
                delta_score = self.util.score_delta(observation_)
                # win = delta_score - last_score_delta
                # last_score_delta = delta_score
                act = (int(self.util.is_no_op(action)) - 1)

                self.trajectory['state'][0, frame_count] = observation
                self.trajectory['action'][0, frame_count] = action
                self.trajectory['rew'][0, frame_count] = reward * player.reward_weight[0] +\
                                                         0.1 * moved * player.reward_weight[1] +\
                                                         0.1 * act * player.reward_weight[2]
                
                observation = np.concatenate([observation[len(observation)//2:],observation_])
                frame_count += 1

            if done:
                observation = self.util.preprocess(self.env.reset())
                observation = np.concatenate([observation, observation])
        return observation

    def DRL(self, offspring):
        trained = np.empty_like(offspring)
        for i, q in enumerate(offspring):
            self.player.set_weights(q) # sets nn and r weights
            obs = None
            # x = np.arange(self.n_play)
            # y = np.empty((self.n_play,))
            y = []
            start_time = time()
            no_progress_counter = 0
            best_score = -np.inf
            rew = 0
            training_step = 0
            while time() - start_time < self.max_train * 60:
                obs = self.play(self.player, self.traj_length, obs)
                self.player.pi.train(self.trajectory['state'], self.trajectory['action'][:, :-1], self.trajectory['rew'][:, :-1], -1)
                training_step += 1
                rew += np.sum(self.trajectory['rew'][:, :-1])

                if not training_step % self.round_length:
                    y.append(rew)
                    if rew >= best_score:
                        best_score = rew
                        rew = 0
                        no_progress_counter = 0
                    else:
                        no_progress_counter += 1
                        if no_progress_counter == self.early_stop:
                            print('[%d] DRL break at %d' % (self.ID, training_step))
                            break

            plt.plot(np.arange(len(y)), y)
            plt.draw()
            plt.show()
            trained[i] = {'weights': self.player.get_weights()}
        return trained

    def evaluate(self, trained):
        for individual in trained:
            self.player.set_weights(individual['weights'])
            individual['eval'] = self.eval(self.player, self.eval_length)

    def run(self):
        print('[%d] started' % self.ID)

        while True:
            print('[%d] receiving mating' % self.ID)
            mating = self.recv_mating()
            print('[%d] received all' % self.ID)
            qs = self.crossover(mating)
            print('[%d] crossover ok' % self.ID)
            self.mutate(qs)
            print('[%d] mutating ok' % self.ID)
            trained = self.DRL(qs)
            print('[%d] DRL ok' % self.ID)
            self.evaluate(trained)
            print('[%d] eval ok' % self.ID)
            self.send_evolved(trained)
            print('[%d] sent evolved' % self.ID)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth