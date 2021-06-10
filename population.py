""" 
Author : goji .
Date : 29/01/2021 .
File : population.py .

Description : None

Observations : None
"""

# == Imports ==
from AC import AC
from behavior import *
# =============

class Individual:

    def __init__(self, state_shape, action_dim, goal_dim, epsilon=0.01, lr=0.001, gamma=0.99, entropy_scale=0.0005,
                 gae_lambda=1.0, traj_length=10, batch_size=16, neg_scale=1.0, generation=1):
        self.pi = AC(state_shape, action_dim, epsilon, lr, gamma, entropy_scale, gae_lambda,
                     traj_length, batch_size, neg_scale)
        self.reward_weight = np.random.uniform(0.1, 0.5, size=(goal_dim,))

        dummy_obs = np.ones(state_shape, dtype=np.float32)
        self.pi.policy.get_action(dummy_obs)
        self.pi.policy.value(dummy_obs[np.newaxis])

        self.behavior_stats = {}

        self.gen = generation

    def get_weights(self):
        return {'pi': self.pi.policy.get_weights(),
                'r': self.reward_weight}

    def set_weights(self, w):
        self.pi.policy.set_weights(w['pi'])
        self.reward_weight = w['r']


class LightIndividual:
    def __init__(self, goal_dim, generation=1):

        self.reward_weight = log_uniform(0, 4.1, (goal_dim,), base=10) / 1e4
        self.behavior_stats = {}
        self.gen = generation
        self.model_weights = None

    def get_weights(self):
        return {'pi': self.model_weights,
                'r': self.reward_weight}

    def set_weights(self, w):
        self.model_weights = w['pi']
        self.reward_weight = w['r']


class Population:

    def __init__(self, state_shape, action_dim, sub_goals, size, objectives):
        self.size = size
        self.individuals = np.empty((size,), dtype=LightIndividual)
        for i in range(size):
            dummy_random = Individual(state_shape, action_dim, sub_goals)
            w = dummy_random.get_weights()
            self.individuals[i] = LightIndividual(sub_goals)
            self.individuals[i].behavior_stats = {o.name: -np.inf * o.nature for o in objectives}
            self.individuals[i].behavior_stats.update({'entropy': np.inf })

            self.individuals[i].model_weights = w['pi']

        self.history = {}

    def __repr__(self):
        x = ""
        for i in range(self.size):
            x += "-----Individual %d-----\n" % i +\
                 str(self.individuals[i].behavior_stats) + "\n" +\
                 'gen:%d\n' % self.individuals[i].gen + \
                 'reward weights :' + str(self.individuals[i].reward_weight) + "\n"
        return x

    def track_evolution(self, gen):
        data = {
            'mean_entropy': np.mean([x.behavior_stats['entropy'] for x in self.individuals]),
            'mean_weights': np.mean(np.array([list(x.reward_weight) for x in self.individuals]), axis=-1),
            'min_weights': np.min(np.array([list(x.reward_weight) for x in self.individuals]), axis=-1),
            'max_weights': np.max(np.array([list(x.reward_weight) for x in self.individuals]), axis=-1),
            'avg_score': np.mean([x.behavior_stats['game_reward'] for x in self.individuals]),
            'worst_score': np.min([x.behavior_stats['game_reward'] for x in self.individuals]),
            'best_score': np.max([x.behavior_stats['game_reward'] for x in self.individuals]),
        }
        self.history.update({gen: data})


def log_uniform(low=0., high=1., size=None, base=np.e):
    return np.power(base, np.random.uniform(low, high, size))
