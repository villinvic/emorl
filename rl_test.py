from population import Individual
import fire
import matplotlib.pyplot as plt
from env_utils import *
import gym
from copy import deepcopy

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class RLtest:
    def __init__(self, alpha=0.001, gamma=0.993, traj=10, batch=16, env_id='Tennis-ramNoFrameskip-v4', plot_freq=100):
        self.alpha = alpha
        self.gamma = gamma
        self.traj = traj
        self.batch = batch
        self.plot_freq = plot_freq


        self.N_ENVS = 10
        self.util = name2class[env_id]
        self.envs = [gym.make(self.util.name) for _ in range(self.N_ENVS)]
        self.observations = [None] * self.N_ENVS
        self.state_shape = (self.util.full_state_dim,)
        self.action_dim = self.util.action_space_dim
        self.player = Individual(self.state_shape, self.action_dim, 3, 0.01, alpha, gamma, 0.0005, 1, traj, batch, 1)

        self.trajectory = {
            'state': np.zeros((batch, traj) + self.state_shape, dtype=np.float32),
            'action': np.zeros((batch, traj), dtype=np.int32),
            'rew': np.zeros((batch, traj), dtype=np.float32),
            'base_rew': np.zeros((batch, traj), dtype=np.float32),
        }

        self.last_obs = None

        self.plot_max = 50000
        self.reward = np.full((self.plot_max, 2), np.nan)
        self.range = np.arange(self.plot_max)
        self.plot_index = 0

    def train(self):
        self.player.pi.train(self.trajectory['state'], self.trajectory['action'][:, :-1],
                             self.trajectory['rew'][:, :-1], 0)

    def plot(self):

        plt.clf()
        plt.plot(self.range, smooth(self.reward[:, 0], 100), label='total')
        plt.plot(self.range, smooth(self.reward[:, 1], 100), label='game_score')

        plt.savefig('rl/' + str(self.plot_index) + '.png')

    def train_loop(self):
        c = 1
        try:
            self.player.reward_weight[:] = 1., 0.2, 0.1

            while True:

                for batch_index in range(self.batch):
                    env_id = np.random.randint(0, self.N_ENVS)
                    self.observations[env_id] = self.util.play(self.player,
                                         self.envs[env_id],
                                         batch_index,
                                         self.traj,
                                         4,
                                         self.trajectory,
                                         self.action_dim,
                                         observation=self.observations[env_id],
                                         gpu=0)
                self.reward[self.plot_index, 0] = np.mean(self.trajectory['rew'])
                self.reward[self.plot_index, 1] = np.mean(self.trajectory['base_rew'])
                self.plot_index += 1
                self.train()
                if not c % self.plot_freq:

                    self.plot()

                c += 1
        except KeyboardInterrupt:
            pass
        try:
            while True:
                r = self.util.eval(self.player,
                                     self.envs[0],
                                     self.action_dim,
                                     4,
                                     min_frame=1,
                                     min_games=1,
                                     render=True)
                print(r)

        except KeyboardInterrupt:
            pass

        print('done')

def TEST(alpha=0.001, gamma=0.98, traj=20, batch=16):
    tester = RLtest(alpha, gamma, traj, batch)
    tester.train_loop()


if __name__ == '__main__':
    fire.Fire(TEST)
