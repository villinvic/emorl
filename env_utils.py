import gym
from copy import copy
import numpy as np
from population import Individual

class Objective:
    def __init__(self, name, nature=1, domain=(0.,1.)):
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
                p : {
                    'is_single' : 'S' in p,
                    'complexity': 1,
                    'behavior_functions' : None,
                }
            })

    @staticmethod
    def build_objective_func(*args, prioritized=None, sum=False):
        funcs = []
        if prioritized is not None:
            for i in range(len(args)):
                funcs = [lambda x, arg=arg: np.array([prioritized.make(x), arg.make(x)]) for arg in args]
        elif sum:
            funcs.append(lambda x: np.sum(np.array([arg.make(x) for arg in args])))
        else :
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
        self.indexes = np.array([13,14,21,49,51,54], dtype=np.int32)
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

        self['ram_locations']= dict(player_x=32,
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
            Objective('avg_length', nature=-1, domain=(400.0, 1786.0)),
            Objective('mean_distance', domain=(0.8, 2.))
        ]

        self.action_space_dim = 18
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum=True)

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
            return obs[4]/self.scales[4] - obs[5]/self.scales[5]
                
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
             min_games):

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



name2class = {'Pong-ramNoFrameskip-v4': Pong('Pong-ramNoFrameskip-v4'),
              'Pong-ram-v0': Pong('Pong-ram-v0'),
              'Pong-ramDeterministic-v4': Pong('Pong-ramDeterministic-v4'),
              'Boxing-ramNoFrameskip-v4': Boxing('Boxing-ramNoFrameskip-v4'),
              'Boxing-ramDeterministic-v4': Boxing('Boxing-ramDeterministic-v4'),
              }


# TODO
# play loop
# eval loop
# plotting
# auto change plotting, optimization etc...
# SOP1 MOP1 SOP2 MOP2

