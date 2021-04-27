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
import os
# =============

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def RUN(env='Boxing-ramNoFrameskip-v4', client_mode=False, collector_ip=None, size=15, n_server=4, n_send=1,
        epsilon=0.0, checkpoint_dir='checkpoint/', problem ='MOP3', start_from=None):

    if start_from == 'latest':
        pass

    collector = Collector(env, size, n_server, n_send, epsilon, checkpoint_dir, problem,
                          start_from, client_mode, collector_ip)

    collector.main_loop()


if __name__ == '__main__':
    sys.exit(fire.Fire(RUN))
