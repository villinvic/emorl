""" 
Author : goji .
Date : 29/01/2021 .
File : main.py .

Description : None

Observations : None
"""

# == Imports ==
import fire
from collector import Collector
from env_utils import *
import sys
import gym
# =============


def RUN(env='Pong-ram-v0', size=15, n_server=4, n_send=1):

    collector = Collector(env, size, n_server, n_send)

    collector.main_loop()


if __name__ == '__main__':
    sys.exit(fire.Fire(RUN))
