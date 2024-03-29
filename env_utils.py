import gym
from copy import copy
import numpy as np
from population import Individual
import time
import tensorflow as tf
from PIL import Image
import gym_super_mario_bros


class Objective:
    def __init__(self, name, nature=1, domain=(0., 1.)):
        self.name = name
        self.nature = nature
        self.domain = domain
        self.norm = domain[1] - domain[0]

    def make(self, x):
        val = self.nature * (x.behavior_stats[self.name] - self.domain[0]) / self.norm
        if self.nature == -1:
            val += 1
        return val

    def __repr__(self):
        return self.name


class EnvUtil(dict):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(EnvUtil, self).__init__(*args, **kwargs)

        problems = ['SOP1', 'SOP2', 'MOP1', 'MOP2', 'MOP3']
        self['problems'] = dict()

        for p in problems:
            self['problems'].update({
                p: {
                    'is_single'         : 'S' in p,
                    'complexity'        : 1,
                    'behavior_functions': None,
                }
            })

    @staticmethod
    def build_objective_func(*args, prioritized=None, sum_=False):
        funcs = []
        if prioritized is not None:
            for i in range(len(args)):
                funcs = [lambda x, arg=arg: np.array([prioritized.make(x), arg.make(x)]) for arg in args]
        elif sum_:
            funcs.append(lambda x: np.sum(np.array([arg.make(x) for arg in args])))
        else:
            funcs = [lambda x, arg=arg: np.array([arg.make(x)]) for arg in args]

        return funcs

    def play(self, player: Individual,
             env,
             batch_size,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None):
        return observation

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games):
        return {}


class Pong(dict):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(Pong, self).__init__(*args, **kwargs)
        self['player_y'] = 51
        self['player_x'] = 46
        self['enemy_y'] = 21
        self['enemy_x'] = 45
        self['ball_x'] = 49
        self['ball_y'] = 54
        self['enemy_score'] = 13
        self['player_score'] = 14
        self.indexes = np.array([13, 14, 21, 49, 51, 54], dtype=np.int32)
        self.centers = np.array([0, 0, 127, 127, 127, 127], dtype=np.float32)
        self.scales = np.array([0.05, 0.05, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        self.state_dim = len(self.indexes)

        self.action_space_dim = 3

        self['objectives'] = ['win_rate', 'move_rate', 'no_op_rate']

        self.goal_dim = len(self['objectives'])

        self.behavior_functions = [
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], x.behavior_stats[self['objectives'][1]]]),
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], x.behavior_stats[self['objectives'][2]]]),
        ]

    def preprocess(self, obs):
        return (obs[self.indexes] - self.centers) * self.scales

    def score_delta(self, obs):
        return np.float32(obs[1] - obs[0])

    def pad_move(self, obs, last_pos):
        return np.abs(np.float32(obs[4]) - last_pos)

    def is_no_op(self, action_id):
        return action_id == 0

    def action_to_id(self, action_id):
        if action_id == 0:
            return 0
        elif action_id == 1:
            return 2
        else:
            return 3


class Boxing(EnvUtil):
    def __init__(self, name):
        self.name = name
        super(Boxing, self).__init__(name)

        self['ram_locations'] = dict(player_x=32,
                                     player_y=34,
                                     enemy_x=33,
                                     enemy_y=35,
                                     player_score=18,
                                     enemy_score=19,
                                     clock=17)

        self.indexes = np.array([value for value in self['ram_locations'].values()], dtype=np.int32)
        self.centers = np.array([55, 45, 55, 45, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01], dtype=np.float32)
        self.state_dim = len(self.indexes)

        self['objectives'] = [
            Objective('win_rate'),
            Objective('avg_length', nature=-1, domain=(220.0, 1786.0)),
            Objective('mean_distance', domain=(0.8, 2.5))
        ]

        self.action_space_dim = 18
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

    def action_to_id(self, action_id):
        return action_id

    def preprocess(self, obs):
        return (obs[self.indexes] - self.centers) * self.scales

    def distance(self, obs):
        return np.sqrt(np.square(obs[0] - obs[2]) + np.square(obs[1] - obs[3]))

    def win(self, done, obs, eval=False):
        if done:
            return obs[4] / self.scales[4] - obs[5] / self.scales[5]

        return 0

    def compute_damage(self, obs):
        injury = obs[5 + self.state_dim] - obs[5]
        damage = obs[4 + self.state_dim] - obs[4]

        return np.clip(damage / self.scales[4], 0, 2), np.clip(injury / self.scales[5], 0, 2)

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games,
             render=False,
             slow_factor=0.017):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.0,
            'total_punition': 0.0,
            'no_op_rate'    : 0.0,
            'move_rate'     : 0.0,
            'mean_distance' : 0.0,
            'win_rate'      : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
        }
        frame_count = 0
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)
        while frame_count < min_frame or n_games < min_games:
            done = False
            observation = self.preprocess(env.reset())
            observation = np.concatenate([observation, observation, observation, observation])
            while not done:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                
                if render:
                    env.render()
                    time.sleep(slow_factor)
                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr

                observation_ = self.preprocess(observation_)
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward

                r['mean_distance'] += self.distance(observation_)
                r['win_rate'] += int(self.win(done, observation_) > 30)

                frame_count += 1

            n_games += 1

        print(actions)
        r['avg_length'] = frame_count / float(n_games)
        print(r['win_rate'])
        r['win_rate'] = r['win_rate'] / float(n_games)
        r['mean_distance'] = r['mean_distance'] / float(frame_count)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count

        return r

    def play(self, player: Individual,
             env,
             batch_size,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None):

        actions = [0] * action_dim

        if observation is None:
            observation = self.preprocess(env.reset())
            observation = np.concatenate([observation, observation, observation, observation])

        for batch_index in range(batch_size):
            for frame_count in range(traj_length):
                action = player.pi.policy.get_action(observation)
                actions[action] += 1
                reward = 0
                for _ in range(frame_skip):

                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr
                observation_ = self.preprocess(observation_)
                win = int(self.win(done, observation_) > 0)
                dmg, injury = self.compute_damage(observation)

                trajectory['state'][batch_index, frame_count] = observation
                trajectory['action'][batch_index, frame_count] = action

                trajectory['rew'][batch_index, frame_count] = 100 * win * player.reward_weight[0] + \
                                                              dmg * player.reward_weight[1] + \
                                                              -injury * player.reward_weight[2]

                trajectory['base_rew'][batch_index, frame_count] = reward

                if done:
                    observation = self.preprocess(env.reset())
                    observation = np.concatenate([observation, observation, observation, observation])
                else:
                    observation = np.concatenate([observation[len(observation) // 4:], observation_])

        return observation

class TennisImage(EnvUtil):
    def __init__(self, name):
        super(TennisImage, self).__init__(name)
        self['ram_locations'] = dict(enemy_x=27,
                                     enemy_y=25,
                                     enemy_score=70,
                                     ball_x=16,
                                     ball_y=15,
                                     player_x=26,
                                     player_y=24,
                                     player_score=69,
                                     ball_height=17)

        self.indexes = np.array([value for value in self['ram_locations'].values()], dtype=np.int32)
        self.reversed_indexes = np.array([26, 24, 70, 16, 15, 27, 25, 69, 17], dtype=np.int32)
        self.centers = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.007, 0.007, 0.2, 0.007, 0.007, 0.007, 0.007, 0.2, 0.025], dtype=np.float32)

        self.state_dim = (80, 70, 4)
        self.full_state_dim = (80, 70, 4)
        self.ram_state_dim = len(self.indexes) + 1
        self.y_bounds = (0.91, 1.48)
        # 2 - 74 75 - 148
        self.side = True
        self.max_frames = 15000
        self.frames_since_point = 0
        self.opposite_action_space = {
            0 : 0,
            1 : 1,
            2 : 5,
            3 : 3,
            4 : 4,
            5 : 2,
            6 : 8,
            7 : 9,
            8 : 6,
            9 : 7,
            10: 13,
            11: 11,
            12: 12,
            13: 10,
            14: 16,
            15: 17,
            16: 14,
            17: 15
        }

        self.points = np.array([71, 72], dtype=np.int32)
        self.top_side_points = np.array([0, 3, 4, 7, 8, 11])

        self['objectives'] = [
            Objective('game_score'),
            Objective('aim_quality', domain=(0., 0.55)),
            Objective('mobility', domain=(0., 0.06)),
        ]

        self.action_space_dim = 18
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems'].update({
            'SOP3': {
                'is_single'         : True,
                'complexity'        : 1,
                'behavior_functions': self.build_objective_func(self['objectives'][0],
                                                                self['objectives'][1],
                                                                sum_=True),
            }})

        self['problems'].update({
            'MOP4': {
                'is_single'         : True,
                'complexity'        : 1,
                'behavior_functions': self.build_objective_func(self['objectives'][0], self['objectives'][1]),
            }})

        self.mins = [np.inf, np.inf]
        self.maxs = [-np.inf, -np.inf]
        self.ball_max = [-np.inf, -np.inf]
        self.ball_min = [np.inf, np.inf]

    def action_to_id(self, action_id):
        return action_id


    def preprocess(self, obs):

        if self.side:
            indexes = self.indexes
        else:
            indexes = self.reversed_indexes
        return np.concatenate([(obs[indexes] - self.centers) * self.scales, [np.float32(self.side)]])

    def is_back(self, obs):
        # print(np.sqrt((obs[5]-obs[3])**2+(obs[6]-obs[4])**2))
        # for ob, k in zip(obs, self['ram_locations'].keys()):
        #    print(k, ob)
        # print()
        if self.side:
            return obs[6] > 1.022
        else:
            return obs[6] < 0.028

    def is_front(self, obs):
        # print(obs[3]*100, obs[4]*100)
        if self.side:
            return obs[6] < 0.756
        else:
            return obs[6] > 0.294

    def distance_ran(self, obs, obs_):
        d = np.sqrt((obs[1] - obs_[1]) ** 2 + (obs[0] - obs_[0]) ** 2)
        if d > 20 * 0.007:
            d = 0
        return d

    def self_dy(self, full_obs):
        return np.abs(full_obs[6] - full_obs[-self.ram_state_dim + 6])

    def aim_quality(self, full_obs):
        ball_x = full_obs[-self.ram_state_dim + 3]
        ball_y = full_obs[-self.ram_state_dim + 4]
        # print('1', ball_x, ball_y)
        # print('2', full_obs[-2*self.state_dim+3], full_obs[-2*self.state_dim+4])
        vector = complex(ball_y - full_obs[-2 * self.ram_state_dim + 4], ball_x - full_obs[-2 * self.ram_state_dim + 3])
        angle = np.angle(vector)

        opp_x = full_obs[-self.ram_state_dim]
        opp_y = full_obs[-self.ram_state_dim + 1]
        dY = opp_y - ball_y

        scale = (0.5 + 0.2 * np.abs(dY)) * np.sign(dY)
        deviation = np.tan(angle) * scale

        quality = np.clip(np.abs(ball_x + deviation - opp_x), 0, 1) + 0.2

        return quality

    def proximity_to_front(self, obs):
        if self.side:
            return (np.abs(0.406 - (obs[6] - 0.63)) / 0.406) ** 2
        else:
            return ((obs[6] - 0.014) / 0.406) ** 2

    def proximity_to_back(self, obs):
        if self.side:
            return (np.abs(0.406 - (1.036 - obs[6])) / 0.406) ** 2
        else:
            return (np.abs(0.406 - (obs[6] - 0.014)) / 0.406) ** 2

    def is_returning(self, preprocessed_obs, opp=False):
        if opp:
            side = not self.side
        else:
            side = self.side
        d1 = preprocessed_obs[4 + self.ram_state_dim] - preprocessed_obs[4]
        d2 = preprocessed_obs[4 - self.ram_state_dim * 2] - preprocessed_obs[4 - self.ram_state_dim * 3]
        d2x = preprocessed_obs[3 - self.ram_state_dim * 2] - preprocessed_obs[3 - self.ram_state_dim * 3]
        if abs(d2) + abs(d2x) > 0.1:
            return False

        d = d1 * d2

        if not side:
            d2 = -d2

        return (d <= 0 and d2 < 0)

    def win(self, obs, last_obs, eval=False):
        dself = obs[7] - last_obs[7]
        dopp = obs[2] - last_obs[2]
        dscore = np.clip(dself, 0, 1) - np.clip(dopp, 0, 1)
        return dscore

    def swap_court(self, full_obs):
        total = np.sum(full_obs[self.points])
        if total in self.top_side_points:
            self.side = True
        else:
            self.side = False

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games,
             render=False,
             render_test=False,
             slow_factor=0.017,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'game_score'    : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'mobility'      : 0.,
            'n_shoots'      : 0.,
            'aim_quality'   : 0.,
            'opp_shoots'    : 0,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)

        while frame_count < min_frame or n_games < min_games:
            done = False
            observation = env.reset()
            self.frames_since_point = 0

            ram_data = env.unwrapped._get_ram()
            self.swap_court(observation)
            ram_data  = self.preprocess(ram_data)
            ram_data = np.concatenate([ram_data, ram_data, ram_data, ram_data])
            while not done or frame_count < min_frame:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                if render:
                    env.render()
                    if render_test:
                        time.sleep(slow_factor)

                observation, reward, done, info = env.step(self.action_to_id(action))

                if reward == 0:
                    if abs(ram_data[3] - ram_data[3 + 3 * self.ram_state_dim]) < 1e-4 and \
                            abs(ram_data[4] - ram_data[4 + 3 * self.ram_state_dim]) < 1e-4:
                        self.frames_since_point += 1
                        if self.frames_since_point > 300 // frame_skip:
                            r['game_score'] = -np.inf
                            r['mobility'] = -np.inf
                            r['aim_quality'] = -np.inf
                            return r
                else:
                    self.frames_since_point = 0
                ram_data_ = env.unwrapped._get_ram()
                self.swap_court(ram_data_)
                ram_data_ = self.preprocess(ram_data_)
                r['game_score'] += reward  # self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
                # r['opponent_run_distance'] += self.distance_ran(observation[3 * len(observation) // 4:], observation_)
                ram_data = np.concatenate([ram_data[len(ram_data) // 4:], ram_data_])
                r['mobility'] += self.self_dy(ram_data)

                is_returning = self.is_returning(ram_data)
                if is_returning:
                    r['n_shoots'] += 1
                    r['aim_quality'] += self.aim_quality(ram_data)
                r['opp_shoots'] += int(self.is_returning(ram_data, True))
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward

                frame_count += 1
            n_games += 1

        print(actions)
        r['avg_length'] = frame_count / float(n_games)
        r['game_score'] = (r['game_score'] + 24. * n_games) / (48. * n_games)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count
        r['aim_quality'] /= np.clip(r['n_shoots'], 120, np.inf)
        r['mobility'] /= frame_count

        return r

    def play(self, player: Individual,
             env,
             batch_size,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             data=None):

        actions = [0] * action_dim
        force_reset = False

        if data is None:
            observation = env.reset()
            self.frames_since_point = 0
            ram_data = env.unwrapped._get_ram()
            self.swap_court(ram_data)
            ram_data = self.preprocess(ram_data)
            ram_data = np.concatenate([ram_data, ram_data, ram_data, ram_data])
        else:
            observation, ram_data = data

        for batch_index in range(batch_size):
            for frame_count in range(traj_length):
                action = player.pi.policy.get_action(observation)
                actions[action] += 1

                # env.render()
                # time.sleep(0.5)
                observation_, reward, done, info = env.step(self.action_to_id(action))

                # print(observation_)
                ram_data_ = env.unwrapped._get_ram()
                self.swap_court(ram_data_)
                ram_data_ = self.preprocess(ram_data_)
                punish = 0
                if reward == 0:
                    if abs(ram_data[3] - ram_data[3 + 3 * self.ram_state_dim]) < 1e-4 and \
                            abs(ram_data[4] - ram_data[4 + 3 * self.ram_state_dim]) < 1e-4:
                        self.frames_since_point += 1
                        if self.frames_since_point > 300 // frame_skip:
                            punish -= 1.5
                            force_reset = True
                else:
                    self.frames_since_point = 0

                # win = self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
                # front = np.clip(self.proximity_to_front(observation_) - 0.25, 0, 1)
                back = self.proximity_to_back(ram_data_)

                trajectory['state'][batch_index, frame_count] = observation
                trajectory['action'][batch_index, frame_count] = action
                trajectory['rew'][batch_index, frame_count] = reward * player.reward_weight[0] + punish
                # -0.5 * front * player.reward_weight[2]
                observation[:] = observation_

                trajectory['base_rew'][batch_index, frame_count] = reward

                if done or force_reset:
                    force_reset = False
                    self.frames_since_point = 0
                    observation = env.reset()
                    ram_data = self.preprocess(env.unwrapped._get_ram())
                    ram_data = np.concatenate([ram_data, ram_data, ram_data, ram_data])
                else:
                    ram_data = np.concatenate([ram_data[len(ram_data) // 4:], ram_data_])
                    trajectory['rew'][batch_index, frame_count] += \
                        self.aim_quality(ram_data) * np.float32(self.is_returning(ram_data)) * \
                        player.reward_weight[1] \
                        + (1 + back) * self.self_dy(ram_data) * player.reward_weight[2]
        return observation, ram_data


class Tennis(EnvUtil):
    def __init__(self, name):
        self.name = name
        super(Tennis, self).__init__(name)

        self['ram_locations'] = dict(enemy_x=27,
                                     enemy_y=25,
                                     enemy_score=70,
                                     ball_x=16,
                                     ball_y=15,
                                     player_x=26,
                                     player_y=24,
                                     player_score=69,
                                     ball_height=17)

        self.indexes = np.array([value for value in self['ram_locations'].values()], dtype=np.int32)
        self.reversed_indexes = np.array([26, 24, 70, 16, 15, 27, 25, 69, 17], dtype=np.int32)
        self.centers = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.2, 0.025], dtype=np.float32)
        self.state_dim = len(self.indexes) + 2
        self.full_state_dim = self.state_dim * 4
        self.y_bounds = (0.91, 1.48)
        # 2 - 74 75 - 148
        self.side = True
        self.max_frames = 15000
        self.frames_since_point = 0
        self.opposite_action_space = {
                0:0,
                1:1,
                2:5,
                3:3,
                4:4,
                5:2,
                6:8,
                7:9,
                8:6,
                9:7,
                10:13,
                11:11,
                12:12,
                13:10,
                14:16,
                15:17,
                16:14,
                17:15
        }

        self.points = np.array([71, 72], dtype=np.int32)
        self.top_side_points = np.array([0, 3, 4, 7, 8, 11])

        self.max_quality = 0

        self['objectives'] = [
            Objective('game_score'),
            Objective('aim_quality', domain=(0., 0.6)),
            Objective('mobility', domain=(0., 0.09)),
        ]

        self.action_space_dim = 9
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems'].update({
                'SOP3': {
                    'is_single'         : True,
                    'complexity'        : 1,
                    'behavior_functions': self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   sum_=True),
                }})

        self['problems'].update({
                'MOP4': {
                    'is_single'         : True,
                    'complexity'        : 1,
                    'behavior_functions': self.build_objective_func(self['objectives'][0], self['objectives'][1]),
                }})


        self.mins = [np.inf, np.inf]
        self.maxs = [-np.inf, -np.inf]
        self.ball_max = [-np.inf, -np.inf]
        self.ball_min = [np.inf, np.inf]

    def action_to_id(self, action_id):
        # ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
        # 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
        if action_id > 1:
            action_id += 8
        else:
            action_id = 0

        return action_id

    def preprocess(self, obs):

        if self.side:
            indexes = self.indexes
        else:
            indexes = self.reversed_indexes

        reduced = (obs[indexes] - self.centers) * self.scales
        distance_from_ball = self.distance_from_ball(reduced)
        return np.concatenate([(obs[indexes] - self.centers) * self.scales, [np.float32(self.side), distance_from_ball]])

    def distance_from_ball(self, obs):
        return np.sqrt((obs[3]-obs[5])**2+(obs[4]-obs[6])**2) * 0.9

    def is_back(self, obs):
        #print(np.sqrt((obs[5]-obs[3])**2+(obs[6]-obs[4])**2))
        #for ob, k in zip(obs, self['ram_locations'].keys()):
        #    print(k, ob)
        #print()
        if self.side:
            return obs[6] > 1.022
        else:
            return obs[6] < 0.028

    def is_front(self, obs):
        # print(obs[3]*100, obs[4]*100)
        if self.side:
            return obs[6] < 0.756
        else:
            return obs[6] > 0.294

    def distance_ran(self, obs, obs_):
        d = np.sqrt((obs[1] - obs_[1])**2 + (obs[0] - obs_[0])**2)
        if d > 20 * 0.01:
            d = 0
        return d

    def self_dy(self, full_obs):
        dx = np.abs(full_obs[5] - full_obs[-self.state_dim+5])
        dy = np.abs(full_obs[6] - full_obs[-self.state_dim+6])
        d = dx+dy
        if d > 30 * 0.01:
            dy = 0
        return dy

    def aim_quality(self, full_obs):
        ball_x = full_obs[-self.state_dim+3]
        ball_y = full_obs[-self.state_dim+4]
        dplayer_y = full_obs[-self.state_dim+1]-full_obs[-self.state_dim+6]
        #print('1', ball_x, ball_y)
        #print('2', full_obs[-2*self.state_dim+3], full_obs[-2*self.state_dim+4])
        vector = complex(ball_y - full_obs[-2*self.state_dim+4], ball_x - full_obs[-2*self.state_dim+3])
        vector_p2p = complex(dplayer_y, full_obs[-self.state_dim+0]-full_obs[-self.state_dim+5])
        if self.side :
            vector *= -1
            vector_p2p *= -1
        angle = np.angle(vector)
        angle_2 = np.angle(vector_p2p)
        #print(full_obs[-self.state_dim+1]-full_obs[-self.state_dim+6])

        quality = np.clip((angle-angle_2) ** 2 * 2 + 0.1, 0, 2.25)
        if np.abs(dplayer_y) < 0.35:
            quality *= 0.2

        #opp_x = full_obs[-self.state_dim]
        #opp_y = full_obs[-self.state_dim+1]
        #dY = opp_y - ball_y

        #scale = (0.5 + 0.2 * np.abs(dY)) * np.sign(dY)
        #deviation = np.tan(angle) * scale

        #quality = np.clip(np.abs(ball_x + deviation - opp_x), 0, 1) + 0.2

        return quality


            
    def proximity_to_front(self, obs):
        if self.side:
            return (np.abs(0.406 - (obs[6] - 0.63))/ 0.406)**2
        else:
            return ((obs[6]-0.014)/0.406)**2
      
    
    def proximity_to_back(self, obs):
        if self.side:
            return (np.abs(0.406 - (1.036 - obs[6]))/ 0.406)**2
        else:
            return (np.abs(0.406 - (obs[6]-0.014))/ 0.406)**2
            

    def is_returning(self, preprocessed_obs, opp=False):
        if opp:
            side = not self.side
        else:
            side = self.side
        d1 = preprocessed_obs[4+self.state_dim] - preprocessed_obs[4]
        d2 = preprocessed_obs[4-self.state_dim*2] - preprocessed_obs[4-self.state_dim*3]
        d2x = preprocessed_obs[3-self.state_dim*2] - preprocessed_obs[3-self.state_dim*3]
        if abs(d2)+abs(d2x) > 0.12 or abs(d2)+abs(d2x) < 1e-6:
            return False

        d = d1 * d2

        if not side:
            d2 = -d2

        return (d <= 0 and d2<0)


    def win(self, obs, last_obs, eval=False):
        dself = obs[7] - last_obs[7]
        dopp = obs[2] - last_obs[2]
        dscore = np.clip(dself, 0, 1) - np.clip(dopp, 0, 1)
        return dscore

    def swap_court(self, full_obs):
        total = np.sum(full_obs[self.points])
        if total in self.top_side_points:
            self.side = True
        else:
            self.side = False

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             max_frames,
             min_games,
             render=False,
             slow_factor=0.04,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'game_score'      : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'mobility'         : 0.,
            'n_shoots'          : 0.,
            'aim_quality': 0.,
            'opp_shoots'    : 0,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)


        while  n_games < min_games:
            done = False
            observation = env.reset()
            self.frames_since_point = 0

            self.swap_court(observation)
            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation, observation, observation])
            while not done and frame_count < max_frames:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                if render:
                    env.render()
                    time.sleep(slow_factor)

                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr
                    if done:
                        break

                while not done and np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5:
                    for _ in range(frame_skip):
                        observation_, rr, done, info = env.step(
                            self.action_to_id(action))
                        reward += rr
                        if done:
                            break

                if reward == 0:
                    if abs(observation[3]-observation[3+3*self.state_dim])<1e-4 and\
                            abs(observation[4]-observation[4+3*self.state_dim])<1e-4 :
                        self.frames_since_point += 1
                        if self.frames_since_point > 300//frame_skip:
                            r['game_score'] = -np.inf
                            r['mobility'] = -np.inf
                            r['aim_quality'] = -np.inf
                            return r
                else:
                    self.frames_since_point = 0





                self.swap_court(observation_)
                observation_ = self.preprocess(observation_)
                r['game_score'] += reward  # self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
                #r['opponent_run_distance'] += self.distance_ran(observation[3 * len(observation) // 4:], observation_)
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                r['mobility'] += self.self_dy(observation)

                is_returning = self.is_returning(observation)
                if is_returning:
                    r['n_shoots'] += 1
                    r['aim_quality'] += self.aim_quality(observation)
                r['opp_shoots'] += int(self.is_returning(observation, True))
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward
                    r['aim_quality'] += reward * 0.05


                frame_count += 1

            n_games += 1

        print(actions)
        r['avg_length'] = frame_count / float(n_games)
        r['game_score'] = (r['game_score'] + 24. * n_games) / (48. * n_games)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count
        r['aim_quality'] = np.clip(r['aim_quality'], 0, np.inf)
        r['aim_quality'] /= np.clip(r['n_shoots'], 48*n_games, np.inf)
        r['mobility'] /= frame_count

        return r

    def play(self, player: Individual,
             env,
             batch_index,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None,
             gpu=-1):

        actions = [0] * action_dim
        force_reset = False

        if observation is None:
            observation = env.reset()
            self.frames_since_point = 0
            self.swap_court(observation)
            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation, observation, observation])

        for frame_count in range(traj_length):
            action = player.pi.policy.get_action(observation, gpu=gpu)
            actions[action] += 1
            reward = 0

            #env.render()
            #time.sleep(0.5)
            for _ in range(frame_skip):
                observation_, rr, done, info = env.step(
                    self.action_to_id(action))
                reward += rr
                if done:
                    break
            while not done and np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5:
                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr
                    if done:
                        break

                #print(np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5,
                #      np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))))

            # print(observation_)
            self.swap_court(observation_)
            punish = 0
            if reward == 0:
                if abs(observation[3]-observation[3+3*self.state_dim])<1e-4 and\
                        abs(observation[4]-observation[4+3*self.state_dim])<1e-4 :
                    self.frames_since_point += 1
                    if self.frames_since_point > 200//frame_skip:
                        punish -= 5
                        force_reset = True
            else:
                self.frames_since_point = 0

            observation_ = self.preprocess(observation_)
            # win = self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
            # front = np.clip(self.proximity_to_front(observation_) - 0.25, 0, 1)
            back = self.proximity_to_back(observation_)

            trajectory['state'][batch_index, frame_count] = observation
            trajectory['action'][batch_index, frame_count] = action
            trajectory['rew'][batch_index, frame_count] = reward * player.reward_weight[0] + punish
                                                          #-0.5 * front * player.reward_weight[2]

            trajectory['base_rew'][batch_index, frame_count] = reward

            if done or force_reset:
                force_reset = False
                self.frames_since_point = 0
                observation = self.preprocess(env.reset())
                observation = np.concatenate([observation, observation, observation, observation])
            else:
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                if self.is_returning(observation):
                    aim_quality = self.aim_quality(observation)
                else:
                    aim_quality = 0
                trajectory['rew'][batch_index, frame_count] +=\
                    aim_quality * player.reward_weight[1] \
                    + (1 + 2*back) * self.self_dy(observation) * player.reward_weight[2] * 0.03

        return observation


class Breakout(EnvUtil):
    def __init__(self, name):
        self.name = name
        super(Breakout, self).__init__(name)

        self['ram_locations'] = dict(ball_x=99,
                     ball_y=101,
                     player_x=72,
                     blocks_hit_count=77,
                     score=84,
                     lives=57)
        self['ram_locations'].update({
            'block_bit_map' + str(i): i for i in range(30)
        })

        # px 55 191
        # bx 55 200

        self.indexes = np.array([value for value in self['ram_locations'].values()], dtype=np.int32)
        self.centers = np.array([0 for _ in range(len(self.indexes))], dtype=np.float32)
        self.scales = np.array([0.004 for _ in range(len(self.indexes))], dtype=np.float32)
        self.scales[:3] *= 1.2
        self.state_dim = len(self.indexes)
        self.full_state_dim = 2 * len(self.indexes)
        self.block_hit_combo = 0
        self.hit_cooldown = 0
        self.hit_max_cooldown = 18
        self.n_hit_max = 200.
        self.on_sides_max_count = 1
        self.on_sides_count = 0

        self.max_pos_history = 60*20
        self.pos_history = np.random.uniform(55, 191, self.max_pos_history) * 0.004 * 1.2
        self.pos_history_index = 0

        self['objectives'] = [
            Objective('game_score', domain=(0., 200.)),
            Objective('best_shot', nature=1, domain=(0., 25.)),
            Objective('n_hits', nature=1, domain=(0., self.n_hit_max)),
        ]

        self.action_space_dim = 3
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

    def punish_pos(self, full_obs):
        self.pos_history[self.pos_history_index % self.max_pos_history] = full_obs[2]
        mean_dist = np.nanmean(np.abs(np.random.choice(self.pos_history, int(self.max_pos_history*0.1))-full_obs[2]))
        self.pos_history_index += 1
        punish = mean_dist < (25 * 0.004* 1.2)
        if punish:
            print('punish')
        return punish


    def preprocess(self, obs):
        return (obs[self.indexes] - self.centers) * self.scales

    def is_hit(self, full_obs):
        #is_hit =  full_obs[1-2*self.state_dim] > 170 * 0.005 \
        #       and  0.2 > full_obs[1-3*self.state_dim]-full_obs[1-self.state_dim] > 1e-8 \
        #          and 0.2 > full_obs[1+self.state_dim]-full_obs[1] > 1e-8
        is_hit = (self.hit_cooldown == 0) \
                 and full_obs[1-self.state_dim] > 165 * 0.004*1.2 \
                 and 0.2 > full_obs[1-2*self.state_dim]-full_obs[1-self.state_dim] > 1e-8
        if is_hit:
            #print('is_hit !', full_obs[1-2*self.state_dim]/0.004)
            self.hit_cooldown = self.hit_max_cooldown
            self.block_hit_combo = 0
        elif self.hit_cooldown > 0:
            self.hit_cooldown -= 1

        return is_hit

    def d_lives(self, full_obs):
        return int((full_obs[5-2*self.state_dim]-full_obs[5-self.state_dim])>1e-6)

    def action_to_id(self, action):
        return action+1

    def on_sides(self, full_obs):

        on_sides = (full_obs[2 - self.state_dim]/ (0.004*1.2) < 60 or full_obs[2 - self.state_dim]/ (0.004*1.2) > 186) and \
               (full_obs[2]/ (0.004*1.2) < 57 or full_obs[2] / (0.004*1.2) > 190)


        if on_sides:
            self.on_sides_count += 1
        else:
            self.on_sides_count = 0
        return (self.on_sides_count>self.on_sides_max_count) and on_sides

    def combo_bonus(self, reward):
        if reward > 0:
            bonus = np.clip(self.block_hit_combo, 0, 10)
            self.block_hit_combo += 1
            #print(self.block_hit_combo)
            return bonus
        return 0

    def moving(self, full_obs):
        return np.abs(full_obs[2]-full_obs[2-self.state_dim]) > 1e-5

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games,
             render=False,
             slow_factor=0.04,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'game_score'      : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'best_shot'         : 0.,
            'n_hits'          : 0.,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)
        best_shot = 0
        hits = 0

        while frame_count < min_frame or n_games < min_games:
            done = False
            observation = env.reset()
            self.frames_since_point = 0

            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation])
            self.block_hit_combo = 0
            while not done or frame_count < min_frame:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                if render:
                    env.render()
                    time.sleep(slow_factor)

                if (observation[1] != 0 or observation[5] > 4.5 * 0.004) and observation[1-self.state_dim] == 0:
                    env.step(1)
                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(self.action_to_id(action))
                    self.combo_bonus(rr)
                    r['game_score'] += int(rr > 0)
                    best_shot = self.block_hit_combo if self.block_hit_combo > best_shot else best_shot
                    reward += rr

                #print(observation_)
                observation_ = self.preprocess(observation_)

                observation = np.concatenate([observation[len(observation) // 2:], observation_])
                #print(observation)

                hits += int(self.is_hit(observation))
                if hits >= self.n_hit_max:
                    print('Max n_hit reached')
                    hits = 0

                    break
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward


                frame_count += 1
            r['best_shot'] += best_shot
            r['n_hits'] += hits
            best_shot = 0
            hits = 0
            n_games += 1

        # print(actions)
        r['avg_length'] = frame_count / float(n_games)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count
        r['game_score'] /= float(n_games)
        r['best_shot'] /= float(n_games)
        r['n_hits'] /= float(n_games)


        return r

    def play(self, player: Individual,
             env,
             batch_size,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None):

        actions = [0] * action_dim
        force_reset = False

        if observation is None:
            observation = env.reset()
            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation])
            self.block_hit_combo = 0

        for batch_index in range(batch_size):
            for frame_count in range(traj_length):
                action = player.pi.policy.get_action(observation)
                actions[action] += 1
                reward = 0
                combo_bonus = 0

                #env.render()
                #time.sleep(0.5)
                if (observation[1] != 0 or observation[5] > 4.5 * 0.004) and observation[1-self.state_dim] == 0:
                    env.step(1)

                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(self.action_to_id(action))
                    reward += np.float32(rr>0)
                    combo_bonus += self.combo_bonus(rr)


                observation_ = self.preprocess(observation_)
                # win = self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
                # front = np.clip(self.proximity_to_front(observation_) - 0.25, 0, 1)

                trajectory['state'][batch_index, frame_count] = observation
                trajectory['action'][batch_index, frame_count] = action

                trajectory['base_rew'][batch_index, frame_count] = reward

                if done or force_reset:
                    force_reset = False
                    self.block_hit_combo = 0
                    observation = self.preprocess(env.reset())
                    observation = np.concatenate([observation, observation])
                else:
                    observation = np.concatenate([observation[len(observation) // 2:], observation_])
                    is_hit = self.is_hit(observation)
                    trajectory['rew'][batch_index, frame_count] = (reward)* player.reward_weight[0] \
                                                              + combo_bonus * player.reward_weight[1] \
                                                              + np.float32(is_hit) * player.reward_weight[2] \
                                                              -(np.float32(self.on_sides(observation))*0.1 + self.d_lives(observation)) * player.reward_weight[0] * 2

        return observation


class Mario(EnvUtil):
    def __init__(self, name):
        self.name = name
        super(Mario, self).__init__(name)
        self.max_time = 400.
        self['objectives'] = [
            Objective('final_x', domain=(0., 3200.)),
            Objective('time_left', nature=1, domain=(0., self.max_time)),
            Objective('game_score', nature=1, domain=(0., 10000.)),
        ]

        self.action_space_dim = 7

        self.state_dim = (70, 70, 2)
        self.full_state_dim = (70, 70, 2)
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

    def inertia(self, current, previous):
        return np.clip((current['x_pos']-previous['x_pos']),0, np.inf) * 0.01

    def d_score(self, current, previous):
        return np.clip(current['score']-previous['score'], 0, np.inf) * 0.001


    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games,
             render=False,
             slow_factor=0.,
             render_test=False,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'final_x'    : 0.,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'time_left'     : 0.,
            'game_score'        : 0.,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)

        while frame_count < min_frame or n_games < min_games:
            done = False
            observation = env.reset()

            while not done or frame_count < min_frame:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1

                observation_, reward, done, info = env.step(action)
                reward /= 25.

                if render:
                    time.sleep(slow_factor)
                    if render_test:
                        env.render()


                observation[:] = observation_
                # print(observation)

                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward

                frame_count += 1
            r['final_x'] += info['x_pos']
            r['time_left'] += info['time'] if info['flag_get'] else 0
            r['game_score'] += info['score']
            n_games += 1

        # print(actions)
        r['avg_length'] = frame_count / float(n_games)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count
        r['final_x'] /= float(n_games)

        if r['time_left'] < n_games//10 * self.max_time*0.9:
            r['time_left'] = 0
        else:
            r['time_left'] /= float(n_games)
        r['game_score'] /= float(n_games)

        return r

    def play(self, player: Individual,
             env,
             batch_size,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None):

        actions = [0] * action_dim
        info_ = None

        if observation is None:
            observation = env.reset()

        for batch_index in range(batch_size):
            for frame_count in range(traj_length):
                action = player.pi.policy.get_action(observation)
                actions[action] += 1

                observation_, reward, done, info = env.step(action)
                if info_ is None:
                    info_ = dict(info)

                trajectory['state'][batch_index, frame_count] = observation
                trajectory['action'][batch_index, frame_count] = action
                #trajectory['x'][batch_index, frame_count] = info['x_pos']

                trajectory['base_rew'][batch_index, frame_count] = reward

                if done :
                    observation = env.reset()
                else:
                    observation[:] = observation_

                trajectory['rew'][batch_index, frame_count] = 3 * reward * player.reward_weight[0] \
                                                              + self.inertia(info, info_) * player.reward_weight[1] \
                                                              + self.d_score(info, info_) * player.reward_weight[2]
                info_.update(info)



        return observation


class Lander(EnvUtil):
    def __init__(self, name):
        super(Lander, self).__init__(name)

        self['objectives'] = [
            Objective('game_reward', domain=(0., 250.)),
            Objective('time_left', nature=1, domain=(0., 100)),
            Objective('game_score', nature=1, domain=(0., 10000.)),
        ]

        self.action_space_dim = 4

        self.state_dim = 8
        self.full_state_dim = (self.state_dim*4,)
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])


    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             min_frame,
             min_games,
             render=False,
             slow_factor=0.04,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'game_score'      : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'best_shot'         : 0.,
            'n_hits'          : 0.,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)

        while frame_count < min_frame or n_games < min_games:
            done = False
            observation = env.reset()

            observation = np.concatenate([observation, observation,observation, observation])
            while not done or frame_count < min_frame:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                if render:
                    env.render()
                    time.sleep(slow_factor)

                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(action)
                    reward += rr
                    if done:
                        break

                observation = np.concatenate([observation[len(observation) // 4:], observation_])

                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward


                frame_count += 1
            n_games += 1

        return r

    def play(self, player: Individual,
             env,
             batch_index,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None,
             gpu=-1):

        actions = [0] * action_dim

        if observation is None:
            observation = env.reset()
            observation = np.concatenate([observation, observation, observation, observation])

        for frame_count in range(traj_length):
            action = player.pi.policy.get_action(observation, gpu=gpu)
            actions[action] += 1

            reward = 0

            for _ in range(frame_skip):
                observation_, rr, done, info = env.step(action)
                reward += rr
                if done:
                    break

            trajectory['state'][batch_index, frame_count] = observation
            trajectory['action'][batch_index, frame_count] = action

            trajectory['base_rew'][batch_index, frame_count] = reward

            if done:
                observation = env.reset()
                observation = np.concatenate([observation, observation, observation, observation])
            else:
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                trajectory['rew'][batch_index, frame_count] = reward* player.reward_weight[0] \

        return observation


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, render=False):
        super(SkipEnv, self).__init__(env)
        self.skip =skip
        self._render = render

    def step(self, action):
        t_reward = 0.
        done = False
        for _ in range(self.skip):
            if self._render:
                self.env.render()
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done :
                break
        return obs, t_reward, done, info


class MarioPreProcess(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MarioPreProcess, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(70,70,1), dtype=np.uint8)

    def observation(self, observation):
        return MarioPreProcess.process(observation)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1]+ 0.114*new_frame[:,:,2]
        img =  tf.image.resize(new_frame[20:220,64:230][:, :, np.newaxis], (70, 70)).numpy().astype(np.uint8)
        return img
        #return new_frame[10:210:4, 46:246:4][np.newaxis]

class GymPreProcess(gym.ObservationWrapper):
    # 210, 160, 3
    def __init__(self, env=None):
        super(GymPreProcess, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80,70,1), dtype=np.uint8)

    def observation(self, observation):
        return GymPreProcess.process(observation)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.999*new_frame[:,:,0] + 0.*new_frame[:,:,1]+ 0.001*new_frame[:,:,2] # 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1]+ 0.114*new_frame[:,:,2]
        img = new_frame[50:210:2,10:150:2][:, :, np.newaxis]
        #img = tf.image.resize(new_frame[50:210,10:150][:, :, np.newaxis], (80, 70), method='nearest').numpy().astype(np.uint8)
        #if np.random.random() < 0.0005:
        #    i = Image.fromarray(img[:,:,0], 'L')
        #    i.save('state.png')
        #time.sleep(15)
        return img

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                            shape = (self.observation_space.shape[-1],
                                                     self.observation_space.shape[0],
                                                     self.observation_space.shape[1]),
                                                dtype=np.float32)
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        self.n_steps = n_steps
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32
        )

    def reset(self):
        self.buffer = np.zeros((80, 70, self.n_steps), dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[:,:, -1:] = observation
        return self.buffer

def make_env_mario(env_name, n_mem, frame_skip, render=False):
    env = gym_super_mario_bros.make(env_name)
    return ScaleFrame(BufferWrapper(MarioPreProcess(SkipEnv(env, frame_skip, render)), n_mem))

def make_img_env_gym(env_name, n_mem, frame_skip, render=False):
    env = gym.make(env_name)
    return ScaleFrame(BufferWrapper(GymPreProcess(SkipEnv(env, frame_skip, render)), n_mem))



name2class = {'Pong-ramNoFrameskip-v4'    : Pong('Pong-ramNoFrameskip-v4'),
              'Pong-ram-v0'               : Pong('Pong-ram-v0'),
              'Pong-ramDeterministic-v4'  : Pong('Pong-ramDeterministic-v4'),
              'Boxing-ramNoFrameskip-v4'  : Boxing('Boxing-ramNoFrameskip-v4'),
              'Boxing-ramDeterministic-v4': Boxing('Boxing-ramDeterministic-v4'),
              'Tennis-ramNoFrameskip-v4'  : Tennis('Tennis-ramNoFrameskip-v4'),
              'TennisNoFrameskip-v4'     : TennisImage('TennisNoFrameskip-v4'),
              'Breakout-ramNoFrameskip-v4': Breakout('Breakout-ramNoFrameskip-v4'),
              'Mario'                     : Mario('SuperMarioBros-1-1-v1'),
              'test'                      : Lander('LunarLander-v2')
              }