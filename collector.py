""" 
Author : goji .
Date : 29/01/2021 .
File : collector.py .

Description : None

Observations : None
"""

# == Imports ==
from population import Population, Individual
import numpy as np
import zmq
import subprocess
import signal
from utils import nd_sort, cd_select
import sys
# =============

class Collector:
    def __init__(self, state_shape, action_dim, sub_goals, size, behaviour_functions, n_server, n_send):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.sub_goals = sub_goals
        self.population = Population(state_shape, action_dim, sub_goals, size)
        self.behaviour_functions = behaviour_functions
        self.n_server = n_server
        self.servers = [None] * n_server
        self.n_send = n_send


        context = zmq.Context()
        self.mating_pipe = context.socket(zmq.PUSH)
        self.mating_pipe.bind("ipc://MATING")
        self.evolved_pipe = context.socket(zmq.PULL)
        self.evolved_pipe.bind("ipc://EVOLVED")


    def start_servers(self):
        for i in range(self.n_server):
            cmd = "python3 evolution_server.py ID %d" %(i)
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
        for i in range(self.n_server):
            for j in range(self.n_send*2):
                try:
                    p = self.evolved_pipe.recv_pyobj()
                except KeyboardInterrupt:
                    self.exit()
                    break
                new = Individual(self.state_shape, self.action_dim, self.sub_goals)
                new.set_weights(p['weights'])
                new.behaviour_stats = p['eval']
                offspring[i * self.n_server + j] = new
        return offspring

    def select(self, offspring):
        n_objectives = len(self.behaviour_functions)
        scores = np.empty((n_objectives, len(self.population.individuals) + len(offspring)), dtype=np.float32)
        for objective_num in range(n_objectives):
            for index in range(self.population.size):
                scores[objective_num, index] = self.behaviour_functions[objective_num](self.population.individuals[index])
            for index in range(len(offspring)):
                scores[objective_num, index] = self.behaviour_functions[objective_num](offspring[index])
        frontiers = nd_sort(scores, n_objectives)

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
        sys.exit(0)

    def main_loop(self):
        try:
            while True:
                self.send_mating()
                offspring = self.receive_evolved()
                self.select(offspring)

                # TODO plotting mean and pop score, population objective score...

        except KeyboardInterrupt:
            self.exit()


