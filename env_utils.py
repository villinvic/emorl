import gym
import numpy as np

class Pong(dict):
    def __init__(self, name):
        self.name = name
        super(Pong, self).__init__()
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

class Boxing(dict):
    def __init__(self, name, player_x=32,
                   player_y=34,
                   enemy_x=33,
                   enemy_y=35,
                   enemy_score=19,
                   clock=17,
                   player_score=18):
        self.name = name
        super(Boxing, self).__init__(player_y=player_y, player_x=player_x, enemy_y=enemy_y, enemy_x=enemy_x,
                                     player_score=player_score, enemy_score=enemy_score, clock=clock)


        self.indexes = np.array([value for value in self.values()], dtype=np.int32)
        self.centers = np.array([55, 45, 55, 45, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01], dtype=np.float32)
        self.state_dim = len(self.indexes)

        self['objectives'] = ['win_rate', 'avg_length', 'mean_distance']

        self.action_space_dim = 18



        self.goal_dim = len(self['objectives'])

        self.behavior_functions = [
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], -x.behavior_stats[self['objectives'][1]]]),
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], x.behavior_stats[self['objectives'][2]]]),
        ]

    def action_to_id(self, action_id):
        return action_id

    def preprocess(self, obs):
        return (obs[self.indexes] - self.centers) * self.scales

    def distance(self, obs):
        return np.sqrt(np.square(obs[0] - obs[2]) + np.square(obs[1] - obs[3]))

    def win(self, done, obs):
        if done:
            d = obs[4]/self.scales[4] - obs[5]/self.scales[5]
            if d > 30:
                return 1
            elif d < 0:
                return -1
                
        return 0


    def compute_damage(self, obs):
        injury = obs[5 + self.state_dim] - obs[5]
        damage = obs[4 + self.state_dim] - obs[4]

        return np.clip(damage / self.scales[4], 0, 2), np.clip(injury / self.scales[5], 0, 2)


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

