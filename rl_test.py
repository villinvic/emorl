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
    def __init__(self, alpha=0.001, gamma=0.993, traj=10, batch=16, env_id='Breakout-ramNoFrameskip-v4', plot_freq=100):
        self.alpha = alpha
        self.gamma = gamma
        self.traj = traj
        self.batch = batch
        self.plot_freq = plot_freq

        self.env = gym.make(env_id)
        self.util = name2class[env_id]
        self.state_shape = (self.util.state_dim * 2,)
        self.action_dim = self.util.action_space_dim
        self.player = Individual(self.state_shape, self.action_dim, 3, 0.01, alpha, gamma, 0.0001, 1, traj, batch, 1)

        self.trajectory = {
            'state': np.zeros((batch, traj) + self.state_shape, dtype=np.float32),
            'action': np.zeros((batch, traj), dtype=np.int32),
            'rew': np.zeros((batch, traj), dtype=np.float32),
            'base_rew': np.zeros((batch, traj), dtype=np.float32),
        }

        self.last_obs = None

        self.plot_max = 50000
        self.reward = np.full((self.plot_max,), np.nan)
        self.ent = np.full((self.plot_max,), np.nan)
        self.range = np.arange(self.plot_max)
        self.plot_index = 0

    def train(self):
        self.player.pi.train(self.trajectory['state'], self.trajectory['action'][:, :-1],
                             self.trajectory['rew'][:, :-1], 0)

    def plot(self):

        plt.clf()
        plt.plot(self.range, smooth(self.reward, 100))
        plt.savefig('rl/' + str(self.plot_index) + '.png')

    def train_loop(self):
        c = 1
        obs = None
        try:
            self.player.reward_weight[:] = 1., 0.01, 0.8

            while True:
                obs = self.util.play(self.player,
                                     self.env,
                                     self.batch,
                                     self.traj,
                                     5,
                                     self.trajectory,
                                     self.action_dim,
                                     observation=obs)
                self.reward[self.plot_index] = np.mean(self.trajectory['rew'])
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
                                     self.env,
                                     self.action_dim,
                                     5,
                                     min_frame=1,
                                     min_games=1,
                                     render=True)
                print(r)

        except KeyboardInterrupt:
            pass

        print('done')

def TEST(alpha=0.001, gamma=0.99, traj=32, batch=8):
    tester = RLtest(alpha, gamma, traj, batch)
    tester.train_loop()


if __name__ == '__main__':
    fire.Fire(TEST)
