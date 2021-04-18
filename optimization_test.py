import numpy as np
import optimization
import matplotlib.pyplot as plt

num_objectives = 2
pop_size = 60
sub_objective_dim = 2
scores = np.empty((num_objectives, pop_size, sub_objective_dim), dtype=np.float32)
scores[:, :, 0] = np.zeros((num_objectives, pop_size))
scores[0, :, 1] = np.random.uniform(0, 1 , (pop_size,)) * 10
scores[1, :, 1] = np.random.uniform(0, 1 , (pop_size,)) * 0.01

frontiers = optimization.nd_sort(scores, num_objectives, 0)
print(len(frontiers), pop_size)
color = np.ones(3)
i = 0

optimization.cd_select(scores, frontiers[0], 3)

for f in frontiers:
    i+=1
    color = np.ones(3) * 0.96**i
    for individual in f:
        plt.plot(*scores[:, individual, 1], marker='o', color=color[:])

plt.draw()
plt.show()
