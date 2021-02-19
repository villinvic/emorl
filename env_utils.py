import gym
import numpy as np

class Pong(dict):
    def __init__(self):
        super(Pong, self).__init__()
        self['player_y'] = 51
        self['player_x'] = 46
        self['enemy_y'] = 50
        self['enemy_x'] = 45
        self['ball_x'] = 49
        self['ball_y'] = 54
        self['enemy_score'] = 13
        self['player_score'] = 14

        self['objectives'] = ['win_rate', 'move_rate', 'no_op_rate']

        self.name = 'Pong-ram-v0'

        self.goal_dim = 3

        self.behavior_functions = [
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], x.behavior_stats[self['objectives'][1]]]),
            lambda x: np.array([x.behavior_stats[self['objectives'][0]], x.behavior_stats[self['objectives'][2]]]),
        ]

    def score_delta(self, obs):
        return np.float32(obs[self['player_score']] - obs[self['enemy_score']])

    def pad_move(self, obs, last_pos):
        return np.abs(np.float32(obs[self['player_y']]) - last_pos)

    def is_no_op(self, action_id):
        return action_id == 0


name2class = {'Pong-ram-v0': Pong}


