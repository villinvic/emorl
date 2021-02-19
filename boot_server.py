"""
Author : goji .
Date : 19/02/2021 .
File : evolution_server.py .

Description : None

Observations : None
"""

# == Imports ==
from evolution_server import EvolutionServer
import sys
import fire
# =============


def RUN(ID, env_id):
    print(env_id)
    server = EvolutionServer(int(ID), env_id)
    server.run()
    sys.exit(0)


if __name__ == '__main__':
    fire.Fire(RUN)
