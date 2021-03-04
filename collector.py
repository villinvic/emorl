""" 
Author : goji .
Date : 29/01/2021 .
File : collector.py .

Description : None

Observations : None

"""

# == Imports =
from population import Population, LightIndividual
from env_utils import *
from plotting import PlotterV2
from evolution_server import EvolutionServer
from serializer import Serializer

import numpy as np
import zmq
import subprocess
import signal
from optimization import nd_sort, cd_select
import sys
import time
import p5
from datetime import datetime
# =============

class EXIT(Exception) : pass

class Collector:
    def __init__(self, env_id, size, n_server, n_send, checkpoint_dir='checkpoint/', start_from=None):

        self.util = name2class[env_id]()
        dummy = gym.make(self.util.name)


        self.state_shape = (self.util.state_dim*2,)
        self.action_dim = dummy.action_space.n
        self.goal_dim = self.util.goal_dim
        self.n_server = n_server
        self.servers = [None] * n_server
        self.n_send = n_send
        self.env_id = env_id
        self.behavior_functions = self.util.behavior_functions
        self.size = size
        self.ckpt_dir = checkpoint_dir

        self.population = Population(self.state_shape, self.action_dim, self.goal_dim, self.size,
                                     self.util['objectives'])

        context = zmq.Context()
        self.mating_pipe = context.socket(zmq.PUSH)
        self.mating_pipe.bind("ipc://MATING")
        self.evolved_pipe = context.socket(zmq.PULL)
        self.evolved_pipe.bind("ipc://EVOLVED")

        self.plotter = PlotterV2()
        self.serializer = Serializer(checkpoint_dir)

        self.generation = 1

        self.start_from = start_from

        self.evaluation = None

    def init_pop(self):
        print('Population initialization...')
        self.evaluation = EvolutionServer(-1, self.env_id, subprocess=False)

        for i in range(self.population.size):
            self.evaluation.player.set_weights(self.population.individuals[i].get_weights())
            self.population.individuals[i].behavior_stats = \
                self.evaluation.eval(self.evaluation.player, self.evaluation.eval_length / 4.0)
        print('OK.')

    def start_servers(self):
        for i in range(self.n_server):
            cmd = "python3 boot_server.py %d %s" % (i, self.env_id)
            self.servers[i] = subprocess.Popen(cmd.split())

    def tournament(self, k=1, key='win_rate'):
        p = np.random.choice(np.arange(self.population.size), (k,), replace=False)
        best_index = p[0]
        best_score = -np.inf
        for i in p:
            score = self.population.individuals[i].behavior_stats[key]
            if score > best_score:
                best_index = i
                best_score = score
        # print('tournament:', best_index)
        return best_index

    def send_mating(self):
        p = [None] * 2 * self.n_send
        for i in range(self.n_server):
            for j in range(self.n_send*2):
                p[j] = self.population.individuals[self.tournament()].get_weights()
            self.mating_pipe.send_pyobj(p)

    def receive_evolved(self):
        offspring = np.empty((2 * self.n_send * self.n_server,), dtype=LightIndividual)
        print('Collector receiving...')
        for i in range(self.n_server):
            try:
                p = self.evolved_pipe.recv_pyobj()
            except KeyboardInterrupt:
                raise EXIT

            for j in range(self.n_send*2):
                new = LightIndividual(self.goal_dim, generation=self.generation)
                new.set_weights(p[j]['weights'])
                new.behavior_stats = p[j]['eval']
                offspring[i * self.n_send*2 + j] = new
        print('Done receiving')
        return offspring

    def select(self, offspring):
        n_behavior = len(self.behavior_functions)
        scores = np.empty((n_behavior, len(self.population.individuals) + len(offspring), 2), dtype=np.float32)
        for objective_num, function in enumerate(self.behavior_functions):
            for index in range(self.population.size):
                scores[objective_num, index, :] = function(self.population.individuals[index])
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
        new_pop = np.empty((self.population.size,), dtype=LightIndividual)
        for i, index in enumerate(selected):
            if index < self.population.size:
                new_pop[i] = self.population.individuals[index]
            else:
                new_pop[i] = offspring[index - self.population.size]

        self.population.individuals = new_pop

        return scores, selected

    def exit(self):
        print('Exiting...')
        for i in range(self.n_server):
            self.servers[i].send_signal(signal.SIGINT)

        try:
            self.serializer.dump(self.population, 'run--' + str(datetime.now()) + '--' + str(self.generation-1))
        except Exception as e:
            print('serializer failed :', e)

    def main_loop(self):
        #self.init_pop()
        if self.start_from is not None:
            self.generation = int(self.start_from.split('--')[-1])
            self.population = self.serializer.load(self.start_from)

        self.start_servers()
        time.sleep(6)
        start = True
        scores = None
        selected = None
        try:
            while True:
                self.generation += 1
                self.send_mating()

                if start:
                    start = False
                else:
                    self.plotter.update(self.population, scores, selected, self.generation-1)
                    self.plotter.plot()

                offspring = self.receive_evolved()
                scores, selected = self.select(offspring)

                print(self.population)

        except (KeyboardInterrupt, EXIT):
            self.exit()
        print('EXITED.')

