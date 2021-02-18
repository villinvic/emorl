""" 
Author : goji .
Date : 29/01/2021 .
File : collector.py .

Description : None

Observations : None
"""

# == Imports ==
from population import Population, Individual
from env_utils import *


import numpy as np
import zmq
import subprocess
import signal
from optimization import nd_sort, cd_select
import sys
import time
# =============

class EXIT(Exception) : pass

class Collector:
    def __init__(self, env_id, size, n_server, n_send):

        self.util = name2class[env_id]()
        dummy = gym.make(self.util.name)

        self.state_shape = dummy.observation_space.shape
        self.action_dim = dummy.action_space.n
        self.goal_dim = self.util.goal_dim
        self.population = Population(self.state_shape, self.action_dim, self.goal_dim, size, self.util['objectives'])
        self.n_server = n_server
        self.servers = [None] * n_server
        self.n_send = n_send
        self.env_id = env_id
        self.behavior_functions = self.util.behavior_functions

        context = zmq.Context()
        self.mating_pipe = context.socket(zmq.PUSH)
        self.mating_pipe.bind("ipc://MATING")
        self.evolved_pipe = context.socket(zmq.PULL)
        self.evolved_pipe.bind("ipc://EVOLVED")

        self.generation = 1

    def start_servers(self):
        for i in range(self.n_server):
            cmd = "python3 evolution_server.py %d %s" % (i, self.env_id)
            self.servers[i] = subprocess.Popen(cmd.split())

    def send_mating(self):
        # replacement ?
        p = [None] * 2 * self.n_send
        parents = np.random.choice(self.population.individuals, (self.n_server, self.n_send*2), replace=True)
        for i in range(self.n_server):
            for j in range(self.n_send*2):
                p[j] = parents[i, j].get_weights()
            self.mating_pipe.send_pyobj(p)

    def receive_evolved(self):
        offspring = np.empty((2 * self.n_send * self.n_server,), dtype=Individual)
        print('Receiving...')
        for i in range(self.n_server):
            print(i)
            try:
                p = self.evolved_pipe.recv_pyobj()
            except KeyboardInterrupt:
                self.exit()
                break

            for j in range(self.n_send*2):
                new = Individual(self.state_shape, self.action_dim, self.goal_dim, generation=self.generation)
                new.set_weights(p[j]['weights'])
                new.behavior_stats = p[j]['eval']
                offspring[i * self.n_send*2 + j] = new
        print('done receiving')
        return offspring

    def select(self, offspring):
        n_behavior = len(self.behavior_functions)
        scores = np.empty((n_behavior, len(self.population.individuals) + len(offspring)), dtype=np.float32)
        for objective_num, function in enumerate(self.behavior_functions):
            for index in range(self.population.size):
                scores[objective_num, index] = function(self.population.individuals[index])
            for index in range(len(offspring)):
                scores[objective_num, index+self.population.size] = function(offspring[index])

        frontiers = nd_sort(scores, n_behavior)
        selected = []
        i = 0
        while len(selected) < self.population.size:
            if len(selected) + len(frontiers[i]) <= self.population.size:
                selected.extend(frontiers[i])

            else:
                selected.extend(cd_select(scores, frontiers[i], self.population.size-len(selected)))
            i += 1
        new_pop = np.empty((self.population.size,), dtype=Individual)
        for i, index in enumerate(selected):
            if index < self.population.size:
                new_pop[i] = self.population.individuals[index]
            else:
                new_pop[i] = offspring[index - self.population.size]

        self.population.individuals = new_pop

    def exit(self):
        print('Exiting...')
        for i in range(self.n_server):
            self.servers[i].send_signal(signal.SIGINT)
        print('Done.')
        raise EXIT

    def main_loop(self):
        try:
            self.start_servers()
            time.sleep(5)
            while True:
                self.generation += 1
                self.send_mating()
                offspring = self.receive_evolved()
                self.select(offspring)

                print(self.population)

        except (KeyboardInterrupt, EXIT):
            self.exit()


