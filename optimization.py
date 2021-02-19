""" 
Author : goji .
Date : 15/02/2021 .
File : optimization.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
# =============


def rankmin(x):
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]


def nd_sort(scores, n_objectives ):
    """
    builds frontiers, descending sort
    """
    frontiers = [[]]
    assert n_objectives > 1
    indexes = np.array(list(reversed(np.argsort(scores[0, :, 0]))))
    weighted_score = scores[:, :, 0]*100 + scores[:, :, 1]

    for index in indexes:
        x = len(frontiers)
        k = 0
        while True:
            dominated = False
            for solution in frontiers[k]:
                tmp = True
                for objective_num in range(1, n_objectives):
                    if weighted_score[objective_num][index] > weighted_score[objective_num][solution]:
                        tmp = False
                        break
                dominated = tmp
                if dominated:
                    break
            if dominated:
                k += 1
                if k >= x:
                    frontiers.append([index])
                    break
            else:
                frontiers[k].append(index)
                break

    return frontiers


def cd_select(scores, indexes, size):
    indexes = np.array(indexes)
    distances_left = [0] * len(indexes)
    distances_right = [0] * len(indexes)
    distances = [0] * len(indexes)
    comparing_score = scores[:, :, 1]

    for i, index in enumerate(indexes):
        for index2 in indexes[:index:]:
            dist = distance(comparing_score[:, index], comparing_score[:, index2])
            if dist > 0 and dist > distances_left[i]:
                distances_left[i] = dist
            elif dist < 0 and dist < distances_right[i]:
                distances_right[i] = dist
        distances[i] = distances_left[i] - distances_right[i]

    return indexes[np.argsort(np.array(distances))][-size:]


def distance(x1, x2):
    d = 0
    for i in range(len(x1)):
        d += (x2[i] - x1[i]) * (-1)**i
    return d








def complex2simple(complex_objective):
    """
    complex_objective: (n_sub, individual scores)
    """
    total = 0
    for i, sub in enumerate(reversed(complex_objective)):
        total += sub * 10 ** (i*3)
