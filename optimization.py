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

def is_dominated(x_scores, y_scores):
    assert len(x_scores) == len(y_scores)

    for i in range(len(x_scores)):
        if x_scores[i] > y_scores[i]:
            return True
        elif x_scores[i] < y_scores[i]:
            return False
    return False


def nd_sort(scores, n_objectives ):
    """
    builds frontiers, descending sort
    """
    frontiers = [[]]
    assert n_objectives > 1
    indexes = np.array(list(reversed(np.argsort(scores[0, :, 0]))))

    for index in indexes:
        x = len(frontiers)
        k = 0
        while True:
            dominated = False
            for solution in frontiers[k]:
                tmp = True
                for objective_num in range(1, n_objectives):
                    if is_dominated(scores[objective_num][index], scores[objective_num][solution]):
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
    distances = [np.inf] * len(indexes)
    distances_2 = [np.inf] * len(indexes)
    comparing_score = scores[:, :, -1]

    for i, index in enumerate(indexes):
        for index2 in indexes[:index:]:
            dist = distance(comparing_score[:, index], comparing_score[:, index2])
            if dist < distances[i]:
                distances_2[i] = distances[i]
                distances[i] = dist
            elif dist < distances_2[i]:
                distances_2[i] = dist
        distances[i] += distances_2[i]

    return indexes[np.argsort(np.array(distances))][-size:]


def distance(x1, x2):
    return np.sum(np.absolute(x2 - x1))




def complex2simple(complex_objective):
    """
    complex_objective: (n_sub, individual scores)
    """
    total = 0
    for i, sub in enumerate(reversed(complex_objective)):
        total += sub * 10 ** (i*3)
