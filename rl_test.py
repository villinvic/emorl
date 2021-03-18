import AC
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
    def __init__(self, alpha=0.001, gamma=0.99, traj=200, batch=1, env_id='Pong-ram-v0', plot_freq=100):
        self.alpha = alpha
        self.gamma = gamma
        self.traj = traj
        self.batch = batch
        self.plot_freq = plot_freq

        self.env = gym.make(env_id)
        self.env.frame_skip = 4
        self.util = name2class[env_id]
        self.state_shape = (self.util.state_dim * 2,)
        self.action_dim = self.env.action_space.n
        self.nn = AC.AC(self.state_shape, self.action_dim, 0.01, alpha, gamma, 0, 1, traj_length=traj, batch_size=batch)

        self.trajectory = {
            'state': np.zeros((batch, traj) + self.state_shape, dtype=np.float32),
            'action': np.zeros((batch, traj), dtype=np.int32),
            'rew': np.zeros((batch, traj), dtype=np.float32),
        }

        self.last_obs = None

        self.plot_max = 10000
        self.reward = np.full((self.plot_max,), np.nan)
        self.ent = np.full((self.plot_max,), np.nan)
        self.range = np.arange(self.plot_max)
        self.plot_index = 0

    def play(self):
        if self.last_obs is None:
            observation = self.util.preprocess(self.env.reset())
            observation = np.concatenate([observation, observation])
        else:
            observation = self.last_obs
        batch_total = 0

        actions = [0] * self.action_dim

        for b in range(self.batch):
            for i in range(self.traj):
                action = self.nn.policy.get_action(observation)
                observation_, reward, done, info = self.env.step(action)
                observation_ = self.util.preprocess(observation_)
                actions[action] += 1
                batch_total += reward

                self.trajectory['state'][b, i] = deepcopy(observation)
                self.trajectory['action'][b, i] = action
                self.trajectory['rew'][b, i] = reward

                if done:
                    observation = self.util.preprocess(self.env.reset())
                    observation = np.concatenate([observation, observation])
                else:
                    observation = np.concatenate([observation[len(observation) // 2:], observation_])

        self.reward[self.plot_index] = batch_total
        self.plot_index += 1
        self.last_obs = observation
        if not self.plot_index % 10:
            print(actions)


    def train(self):
        self.nn.train(self.trajectory['state'], self.trajectory['action'][:, :-1],
                             self.trajectory['rew'][:, :-1], -1)

    def plot(self):
        plt.clf()
        plt.plot(self.range, smooth(self.reward, 100))
        plt.savefig('rl/' + str(self.plot_index) + '.png')

    def train_loop(self):
        c = 1
        try:
            while True:
                self.play()
                self.train()
                if not c % self.plot_freq:
                    self.plot()
                c += 1
        except KeyboardInterrupt:
            pass

        print('done')


def TEST(alpha=0.002, gamma=0.98, traj=256, batch=1):
    tester = RLtest(alpha, gamma, traj, batch)
    tester.train_loop()


if __name__ == '__main__':
    fire.Fire(TEST)
