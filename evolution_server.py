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
import socket
# =============

class EvolutionServer:

    def __init__(self, ID, env_id='Pong-ram-v0', collector_ip=None, traj_length=16, batch_size=4, max_train=1000,
                 early_stop=100, round_length=300, min_eval=50000, min_games=2, subprocess=True, mutation_rate=0.5):
        if collector_ip is None:
            self.ip = socket.gethostbyname(socket.gethostname())
        else:
            self.ip = collector_ip

        self.ID = ID
        self.env = gym.make(env_id)
        self.util = name2class[env_id]
        self.action_dim = self.util.action_space_dim
        self.state_shape = (self.util.state_dim*4,)
        self.mutation_rate = mutation_rate
        self.player = Individual(self.state_shape, self.action_dim, self.util.goal_dim, traj_length=traj_length, batch_size=batch_size)
        self.frame_skip = 4

        if subprocess:
            context = zmq.Context()
            self.mating_pipe = context.socket(zmq.PULL)
            self.mating_pipe.connect("tcp://%s:5655" % self.ip)
            self.evolved_pipe = context.socket(zmq.PUSH)
            self.evolved_pipe.connect("tcp://%s:5656" % self.ip)

        self.traj_length = traj_length
        self.max_train = max_train
        self.early_stop = early_stop
        self.round_length = round_length
        self.batch_size = batch_size
        self.min_eval = min_eval
        self.min_games = min_games

        self.trajectory = {
            'state': np.zeros((batch_size, traj_length)+self.state_shape, dtype=np.float32),
            'action': np.zeros((batch_size, self.traj_length), dtype=np.int32),
            'rew': np.zeros((batch_size, self.traj_length), dtype=np.float32),
            'base_rew': np.zeros((batch_size, self.traj_length), dtype=np.float32),
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

    def SBX_beta(self, n, p1, p2, distance):
        if distance < 1e-5:
            return 0, 0
        spread_factor_lower = 1 + 2 * np.clip((min(p1, p2) - 0) / distance, 0, np.inf)  # sometimes negative ?
        spread_factor_upper = 1 + 2 * np.clip((1 - max(p1, p2)) / distance, 0, np.inf)
        amplification_lower = self.amplification_factor(spread_factor_lower, n)
        amplification_upper = self.amplification_factor(spread_factor_upper, n)

        return self.compute_spread_factor(amplification_lower, n), self.compute_spread_factor(amplification_upper, n)

    @staticmethod
    def amplification_factor(spread_factor, distribution_index):
        assert spread_factor >= 0, spread_factor
        assert distribution_index >= 0
        return 2 / (2 - np.power(spread_factor, -(distribution_index + 1)))

    @staticmethod
    def compute_spread_factor(amplification_factor, distribution_index):
        assert amplification_factor >= 1, amplification_factor
        assert distribution_index >= 0, distribution_index
        u = np.random.random()
        if u < amplification_factor / 2:
            return np.power(2 * u / amplification_factor, 1. / (distribution_index + 1))
        else:
            return np.power(1 / (2 - 2 * u / amplification_factor), 1. / (distribution_index + 1))

    def crossover(self, mating):
        offspring = np.empty((len(mating)//2,), dtype=dict)
        offspring_index = 0
        for i in range(0, len(mating)-1, 2):
            p1 = mating[i]
            p2 = mating[i+1]


            # SPX for NN
            q1 = deepcopy(p1)
            # q2 = deepcopy(p1)
            s = 25 * 64 + 64*65 + 65 * 6 + 65 * 1  # 5,447 33927 128×128 × 2 +128×2 + 128×6 + 6 + 128 + 1
            c = 0
            point = np.random.randint(0, s)
            for j in range(len(p1['pi'])):
                if isinstance(p1['pi'][j], np.ndarray) and len(p1['pi'][j] > 0):
                    if isinstance(p1['pi'][j][0], np.ndarray) and len(p1['pi'][j] > 0):
                        for k in range(len(p1['pi'][j])):
                            if c + len(p1['pi'][j][k]) > point and c < point:
                                q1['pi'][j][k][:point-c] = p2['pi'][j][k][:point-c]
                                # q2['pi'][j][k][point-c:] = p2['pi'][j][k][point-c:]
                            elif c < point :
                                q1['pi'][j][k] = p2['pi'][j][k]
                            else:
                                pass
                                # q2['pi'][j][k] = p2['pi'][j][k]
                            c += len(p1['pi'][j][k])
                    else:
                        if c + len(p1['pi'][j]) > point > c:
                            q1['pi'][j][:point-c] = p2['pi'][j][:point-c]
                            # q2['pi'][j][point-c:] = p2['pi'][j][point-c:]
                        elif c < point:
                            q1['pi'][j] = p2['pi'][j]
                        else:
                            # q2['pi'][j] = p2['pi'][j]
                            pass
                        c += len(p1['pi'][j])
            print(c)

            # SBX for reward
            distance = np.fabs(p1['r'] - p2['r'])
            x = 0.5 * (p1['r'] + p2['r'])
            for j in range(len(p1['r'])):
                beta1, beta2 = self.SBX_beta(20, p1['r'][j], p2['r'][j], distance[j])
                if np.random.random() < 0.5:
                    q1['r'] = np.clip(x - 0.5 * beta1 * distance, 0, np.inf)
                else:
                    q1['r'] = np.clip(x + 0.5 * beta2 * distance, 0, np.inf)
            # q2['r'] = x + 0.5 * beta * (np.abs(p1['r'] - p2['r']))


            offspring[offspring_index] = q1
            offspring_index += 1
            # offspring[i+1] = q2
        return offspring

    def mutate(self, offspring, intensity=0.005):
        for q in offspring:
            if np.random.random() < self.mutation_rate:
                for i in range(len(q['pi'])):
                    if isinstance(q['pi'][i], np.ndarray) and len(q['pi'][i] > 0):
                        gaussian_noise = np.random.normal(loc=0, scale=intensity, size=q['pi'][i].shape)
                        q['pi'][i] += gaussian_noise

                gaussian_noised = np.random.normal(loc=q['r'], scale=q['r']*0.6, size=q['r'].shape)
                q['r'] = np.clip(gaussian_noised, 1e-4, np.inf)

    """
    def eval(self, player: Individual, min_frame):
        r = {
            'game_reward': 0.0,
            'avg_length': 0.0,
            'total_punition': 0.0,
            'no_op_rate': 0.0,
            'move_rate': 0.0,
            'mean_distance': 0.0,
            'win_rate': 0.0,
            'entropy': 0.0,
            'eval_length': 0,
        }
        frame_count = 0
        n_games = 0
        actions = [0] * self.env.action_space.n
        dist = np.zeros((self.action_dim,), dtype=np.float32)
        while frame_count < min_frame or n_games < self.min_games:
            done = False
            observation = self.util.preprocess(self.env.reset())
            observation = np.concatenate([observation, observation, observation, observation])
            # last_pos = observation[self.util.state_dim*3 + 4]
            while not done:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                for _ in range(self.frame_skip):
                    observation_, rr, done, info = self.env.step(self.util.action_to_id(action))  # players pad only moves every two frames
                    reward += rr

                observation_ = self.util.preprocess(observation_)
                observation = np.concatenate([observation[len(observation) //4:], observation_])
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward

                r['mean_distance'] += self.util.distance(observation_)
                r['win_rate'] += int(self.util.win(done, observation_) > 30)

                # distance_moved = self.util.pad_move(observation_, last_pos)
                # last_pos = observation_[4]

                # moved = int(distance_moved > 1e-5)

                #r['move_rate'] += moved
                #r['no_op_rate'] += int(self.util.is_no_op(action))

                frame_count += 1

            n_games += 1

        print(actions)
        r['avg_length'] = frame_count / float(n_games)
        r['win_rate'] = r['win_rate'] / float(n_games) # r['win_rate'] = r['game_reward'] / float(n_games) #(np.abs(r['game_reward'] - r['total_punition'])) / float(np.abs(r['game_reward'] - 2 * r['total_punition']))
        # r['no_op_rate'] = r['no_op_rate'] / float(frame_count)
        # r['move_rate'] = r['move_rate'] / float(frame_count)
        r['mean_distance'] = r['mean_distance'] / float(frame_count)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist+1e-8) * dist)
        r['eval_length'] = frame_count
        return r

    def play(self, player: Individual, observation=None):
        actions = [0]*self.action_dim

        if observation is None:
            observation = self.util.preprocess(self.env.reset())
            observation = np.concatenate([observation, observation, observation, observation])
            
        # last_pos = observation[self.util.state_dim*3 + 4]

        for batch_index in range(self.batch_size):
            for frame_count in range(self.traj_length):
                action = player.pi.policy.get_action(observation)
                actions[action] += 1
                reward = 0
                for _ in range(self.frame_skip):
                    observation_, rr, done, info = self.env.step(self.util.action_to_id(action))  # players pad only moves every two frames
                    reward += rr
                observation_ = self.util.preprocess(observation_)
                # distance_moved = self.util.pad_move(observation_, last_pos)
                #last_pos = observation_[4]

                #moved = int(distance_moved > 1e-5)
                #  delta_score = self.util.score_delta(observation_)
                # win = delta_score - last_score_delta
                # last_score_delta = delta_score
                # act = (int(self.util.is_no_op(action)) - 1)
                win = int(self.util.win(done, observation_) > 0)
                dmg, injury = self.util.compute_damage(observation)

                self.trajectory['state'][batch_index, frame_count] = observation
                self.trajectory['action'][batch_index, frame_count] = action

                self.trajectory['rew'][batch_index, frame_count] = 100 * win * player.reward_weight[0] +\
                                                         dmg * player.reward_weight[1] +\
                                                         -injury * player.reward_weight[2]

                self.trajectory['base_rew'][batch_index, frame_count] = reward

                if done:
                    observation = self.util.preprocess(self.env.reset())
                    observation = np.concatenate([observation, observation, observation, observation])
                    # last_pos = observation[self.util.state_dim*3 + 4]
                else:
                    observation = np.concatenate([observation[len(observation) // 4:], observation_])

        return observation

    """
    def DRL(self, offspring):
        trained = np.empty_like(offspring)
        for i, q in enumerate(offspring):
            self.player.set_weights(q) # sets nn and r weights
            obs = None
            # x = np.arange(self.n_play)
            # y = np.empty((self.n_play,))
            # y = []
            start_time = time()
            rew = 0
            top = -np.inf
            training_step = 0
            no_improvement_counter = 0
            # self.player.pi.reset_optim()
            while training_step < self.max_train:#time() - start_time < self.max_train * 60:
                obs = self.util.play(self.player,
                                     self.env,
                                     self.batch_size,
                                     self.traj_length,
                                     self.frame_skip,
                                     self.trajectory,
                                     self.action_dim,
                                     obs)

                self.player.pi.train(self.trajectory['state'], self.trajectory['action'][:, :-1], self.trajectory['rew'][:, :-1], -1)
                training_step += 1
                rew += np.sum(self.trajectory['base_rew'][:, :-1])
                # y.append(rew)
                if not training_step % self.round_length:
                    if rew > top:
                        top = rew
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1
                        if no_improvement_counter == self.early_stop:
                            print('[%d] early stop DRL at %d' % (self.ID, training_step))
                            break
                    rew = 0


            # y.append(np.nan)
            # plt.plot(np.arange(len(y)), smooth(y, 100))
            # plt.draw()
            # plt.show()
            trained[i] = {'weights': self.player.get_weights()}
        return trained

    def evaluate(self, trained):
        for individual in trained:
            self.player.set_weights(individual['weights'])
            individual['eval'] = self.util.eval(self.player,
                                                self.env,
                                                self.action_dim,
                                                self.frame_skip,
                                                self.min_eval,
                                                self.min_games)

            print(individual['eval'])

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
