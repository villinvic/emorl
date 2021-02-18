""" 
Author : goji .
Date : 15/02/2021 .
File : utils.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    indexes = np.array(list(reversed(np.argsort(scores[0]))))

    for index in indexes:
        x = len(frontiers)
        k = 0
        while True:
            dominated = False
            for solution in frontiers[k]:
                tmp = True
                for objective_num in range(1, n_objectives):
                    if scores[objective_num][index] > scores[objective_num][solution]:
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
    distances_left = [0] * len(indexes)
    distances_right = [0] * len(indexes)
    distances = [0] * len(indexes)

    for i, index in enumerate(indexes):
        for index2 in indexes[:index:]:
            dist = distance( scores[:][index], scores[:][index2])
            if dist > 0 and dist > distances_left[i]:
                distances_left[i] = dist
            elif dist < 0 and dist < distances_right[i]:
                distances_right[i] = dist
        distances[i] = distances_left[i] - distances_right[i]
    return indexes[np.argsort(distances)][-size:]

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
