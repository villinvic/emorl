""" 
Author : goji .
Date : 17/02/2021 .
File : plotting.py .

Description : None

Observations : None
"""

# == Imports ==
import p5
import numpy as np
import threading
import sys
import time
# =============

# TODO : pop score area, stats,
#  global stats ( graph maybe ?), generation number.
#  objective performance evolution. ( animate selection)

class Plotter(threading.Thread):
    def __init__(self, population):
        super(Plotter, self).__init__()
        self.population = population
        self.font = None

    def setup(self):
        p5.size(1000, 500)


    def draw(self):
        p5.clear()
        p5.background(1)

        p5.text("Hello Strings!", (10, 100))

        for i in range(len(self.population)):
            p5.rect((i*10, 500), 8, -self.population[i]*10)


    def run(self):
        try:
            p5.run(self.setup, self.draw)
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    def close(self):
        sys.exit(0)


pop = np.random.uniform(0, 10, (100,))
x = Plotter(pop)
x.start()
for _ in range(20):
    time.sleep(1)
    pop[:] = np.random.uniform(0, 10, (100,))
x.close()
x.join()
