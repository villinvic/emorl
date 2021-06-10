""" 
Author : goji .
Date : 15/02/2021 .
File : optimization.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
from copy import deepcopy
# =============


def rankmin(x):
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]


def is_dominated(x_scores, y_scores, epsilon):
    assert len(x_scores) == len(y_scores)

    for i in range(len(x_scores)):
        if i != 0:
            eps = 0
        else:
            eps = epsilon

        if x_scores[i] > y_scores[i] + eps:
            return True
        elif x_scores[i] < y_scores[i]:
            return False

    return False


def argsort_with_order(seq):

    #seqq = np.concatenate([seq[0,:,0][:, np.newaxis]] + [seqq[:, np.newaxis] for seqq in seq[:,:,1]], axis=1)
    seqq = np.concatenate([seqq[:, np.newaxis] for seqq in seq[:,:,0]], axis=1)

    names_l = [str(i) for i in range(len(seq))] # +1
    f = ', '.join(['f8' for _ in range(len(seq))]) # +1
    names = ', '.join(names_l)


    with_fields = np.core.records.fromarrays(seqq.transpose(), names=names, formats=f)
    return list(np.argsort(with_fields, order=tuple(names_l)))


def nd_sort(scores, n_objectives, epsilon=0):
    """
    builds frontiers, descending sort
    """
    frontiers = [[]]
    assert n_objectives > 1
    indexes = np.array(list(reversed(argsort_with_order(scores))))
    for index in indexes:
        x = len(frontiers)
        k = 0
        while True:
            dominated = False
            for solution in frontiers[k]:
                tmp = True
                for objective_num in range(1, n_objectives):
                    if is_dominated(scores[objective_num][index], scores[objective_num][solution], epsilon):
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

    for i, index in enumerate(indexes):
        for index2 in indexes[:index:]:
            dist = distance(scores[:, index], scores[:, index2])
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
