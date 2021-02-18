""" 
Author : goji .
Date : 29/01/2021 .
File : behavior.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
# =============


class Reward(dict):
    def __init__(self, sub_goals):
        super().__init__()
        for sub in sub_goals:
            self[sub] = 0.0

        self.weights = np.random.normal(size=(len(sub_goals),))

    def total(self):
        t = 0.0
        for v, weight in zip(self.values(), self.weights):
            t += v * weight

        return t


class Indicator:
    def __init__(self, *functions):
        self.functions = []
        for f in functions:
            self.functions.append(f)

    def compute_behavior(self, data):
        return [f(data) for f in self.functions]


# indicator functions

def dummy():
    pass
