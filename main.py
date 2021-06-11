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
import time
# =============

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def RUN(env='Tennis-ramNoFrameskip-v4', client_mode=False, collector_ip=None, size=15, n_server=4, n_send=1,
        epsilon=0.0, checkpoint_dir='checkpoint/', problem ='MOP3', start_from=None, max_gen=1e10):

    if start_from == 'latest':
        pass

    if not client_mode and problem =='ALL':

        if start_from is not None:
            _, _, filenames = next(os.walk(start_from))
        for p in ['SOP1','SOP2', 'MOP1', 'MOP2', 'MOP3']:

            target = None
            if start_from is not None:
                for f in filenames:
                    if p in f:
                        target = f
                        break

                if target is not None:
                    collector = Collector(env, size, n_server, n_send, epsilon, checkpoint_dir, p,
                                      start_from+target, client_mode, collector_ip, max_gen)
                else:
                    collector = Collector(env, size, n_server, n_send, epsilon, checkpoint_dir, p,
                                          None, client_mode, collector_ip, max_gen)
            else:
                collector = Collector(env, size, n_server, n_send, epsilon, checkpoint_dir, p,
                                      None, client_mode, collector_ip, max_gen)
            collector.main_loop()
            time.sleep(1)

    else:
        collector = Collector(env, size, n_server, n_send, epsilon, checkpoint_dir, problem,
                              start_from, client_mode, collector_ip, max_gen)
        collector.main_loop()


if __name__ == '__main__':
    sys.exit(fire.Fire(RUN))
