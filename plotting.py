""" 
Author : goji .
Date : 17/02/2021 .
File : plotting.py .

Description : None

Observations : None
"""

# == Imports ==
import numpy as np
import threading
import sys
import matplotlib.pyplot as plt
# =============

# TODO : pop score area, stats,
#  global stats ( graph maybe ?), generation number.
#  objective performance evolution. ( animate selection)

colors = {
    "maroon": (128, 0, 0),
    "dark red": (139, 0, 0),
    "brown": (165, 42, 42),
    "firebrick": (178, 34, 34),
    "crimson": (220, 20, 60),
    "red": (255, 0, 0),
    "tomato": (255, 99, 71),
    "coral": (255, 127, 80),
    "indian red": (205, 92, 92),
    "light coral": (240, 128, 128),
    "dark salmon": (233, 150, 122),
    "salmon": (250, 128, 114),
    "light salmon": (255, 160, 122),
    "orange red": (255, 69, 0),
    "dark orange": (255, 140, 0),
    "orange": (255, 165, 0),
    "gold": (255, 215, 0),
    "dark golden rod": (184, 134, 11),
    "golden rod": (218, 165, 32),
    "pale golden rod": (238, 232, 170),
    "dark khaki": (189, 183, 107),
    "khaki": (240, 230, 140),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "yellow green": (154, 205, 50),
    "dark olive green": (85, 107, 47),
    "olive drab": (107, 142, 35),
    "lawn green": (124, 252, 0),
    "chart reuse": (127, 255, 0),
    "green yellow": (173, 255, 47),
    "dark green": (0, 100, 0),
    "green": (0, 128, 0),
    "forest green": (34, 139, 34),
    "lime": (0, 255, 0),
    "lime green": (50, 205, 50),
    "light green": (144, 238, 144),
    "pale green": (152, 251, 152),
    "dark sea green": (143, 188, 143),
    "medium spring green": (0, 250, 154),
    "spring green": (0, 255, 127),
    "sea green": (46, 139, 87),
    "medium aqua marine": (102, 205, 170),
    "medium sea green": (60, 179, 113),
    "light sea green": (32, 178, 170),
    "dark slate gray": (47, 79, 79),
    "teal": (0, 128, 128),
    "dark cyan": (0, 139, 139),
    "aqua": (0, 255, 255),
    "cyan": (0, 255, 255),
    "light cyan": (224, 255, 255),
    "dark turquoise": (0, 206, 209),
    "turquoise": (64, 224, 208),
    "medium turquoise": (72, 209, 204),
    "pale turquoise": (175, 238, 238),
    "aqua marine": (127, 255, 212),
    "powder blue": (176, 224, 230),
    "cadet blue": (95, 158, 160),
    "steel blue": (70, 130, 180),
    "corn flower blue": (100, 149, 237),
    "deep sky blue": (0, 191, 255),
    "dodger blue": (30, 144, 255),
    "light blue": (173, 216, 230),
    "sky blue": (135, 206, 235),
    "light sky blue": (135, 206, 250),
    "midnight blue": (25, 25, 112),
    "navy": (0, 0, 128),
    "dark blue": (0, 0, 139),
    "medium blue": (0, 0, 205),
    "blue": (0, 0, 255),
    "royal blue": (65, 105, 225),
    "blue violet": (138, 43, 226),
    "indigo": (75, 0, 130),
    "dark slate blue": (72, 61, 139),
    "slate blue": (106, 90, 205),
    "medium slate blue": (123, 104, 238),
    "medium purple": (147, 112, 219),
    "dark magenta": (139, 0, 139),
    "dark violet": (148, 0, 211),
    "dark orchid": (153, 50, 204),
    "medium orchid": (186, 85, 211),
    "purple": (128, 0, 128),
    "thistle": (216, 191, 216),
    "plum": (221, 160, 221),
    "violet": (238, 130, 238),
    "magenta ": (255, 0, 255),
    "orchid": (218, 112, 214),
    "medium violet red": (199, 21, 133),
    "pale violet red": (219, 112, 147),
    "deep pink": (255, 20, 147),
    "hot pink": (255, 105, 180),
    "light pink": (255, 182, 193),
    "pink": (255, 192, 203),
    "antique white": (250, 235, 215),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "blanched almond": (255, 235, 205),
    "wheat": (245, 222, 179),
    "corn silk": (255, 248, 220),
    "lemon chiffon": (255, 250, 205),
    "light golden rod yellow": (250, 250, 210),
    "light yellow": (255, 255, 224),
    "saddle brown": (139, 69, 19),
    "sienna": (160, 82, 45),
    "chocolate": (210, 105, 30),
    "peru": (205, 133, 63),
    "sandy brown": (244, 164, 96),
    "burly wood": (222, 184, 135),
    "tan": (210, 180, 140),
    "rosy brown": (188, 143, 143),
    "moccasin": (255, 228, 181),
    "navajo white": (255, 222, 173),
    "peach puff": (255, 218, 185),
    "misty rose": (255, 228, 225),
    "lavender blush": (255, 240, 245),
    "linen": (250, 240, 230),
    "old lace": (253, 245, 230),
    "papaya whip": (255, 239, 213),
    "sea shell": (255, 245, 238),
    "mint cream": (245, 255, 250),
    "slate gray": (112, 128, 144),
    "light slate gray": (119, 136, 153),
    "light steel blue": (176, 196, 222),
    "lavender": (230, 230, 250),
    "floral white": (255, 250, 240),
    "alice blue": (240, 248, 255),
    "ghost white": (248, 248, 255),
    "honeydew": (240, 255, 240),
    "ivory": (255, 255, 240),
    "azure": (240, 255, 255),
    "snow": (255, 250, 250),
    "black": (0, 0, 0),
    "dim grey": (105, 105, 105),
    "grey": (128, 128, 128),
    "dark grey": (169, 169, 169),
    "silver": (192, 192, 192),
    "light grey": (211, 211, 211),
    "gainsboro": (220, 220, 220),
    "white smoke": (245, 245, 245),
    "white": (255, 255, 255),
}


class PlotterV2:

    def __init__(self, dump_path='generations/', top=10):
        super(PlotterV2, self).__init__()
        self.path = dump_path
        self.exit = False
        self.top = top
        self.updated = False
        self.mean_ent_hist = []

    def update(self, pop, scores, selected, sparse_select, sparse_frontier, gen):
        self.pop = pop
        self.scores = scores
        self.selected = selected
        self.sparse_select = sparse_select
        self.sparse_frontier = sparse_frontier
        self.gen = gen
        self.updated = True

    def plot(self):
        x = np.arange(10)
        width = 0.2
        fig, ((bars, opti), (entropy, _)) = plt.subplots(2, 2, figsize=(10, 10))
        fig.tight_layout()
        data = np.empty((6 + 1, self.top), dtype=np.float32)
        ent = 0
        for i in range(self.top):
            data[0][i] = self.pop.individuals[i].behavior_stats['win_rate']
            data[1][i] = self.pop.individuals[i].behavior_stats['move_rate']
            data[2][i] = self.pop.individuals[i].behavior_stats['no_op_rate']
            data[3][i] = self.pop.individuals[i].reward_weight[0] * 0.1
            data[4][i] = self.pop.individuals[i].reward_weight[1] * 0.1
            data[5][i] = self.pop.individuals[i].reward_weight[2] * 0.1
            ent += self.pop.individuals[i].behavior_stats['entropy']
            data[6][i] = self.pop.individuals[i].gen
        self.mean_ent_hist.append(ent/float(self.top))
        rects1 = bars.bar(x - width, data[0], width, label='Win rate', color='y')
        rects2 = bars.bar(x,  data[1], width, label='Move rate', color='b')
        rects1 = bars.bar(x + width, data[2], width, label='No act rate', color='r')
        rects1 = bars.bar(x - width, data[3], width/3.0, color='k')
        rects2 = bars.bar(x, data[4], width/3.0, color='k')
        rects1 = bars.bar(x + width, data[5], width/3.0, color='k')

        bars.set_ylabel('Scores')
        bars.set_xlabel('Individual Generations')
        bars.set_title('Top %d scores, ranked by win rate' % self.top)
        bars.set_xticks(x)
        bars.set_xticklabels(data[-1].astype(int))
        bars.legend()
        print(self.sparse_select, self.sparse_frontier)
        for index in range(len(self.scores[0])):
            if self.sparse_select is not None and index in self.sparse_frontier:
                color = (0, 0, 1.0) if index not in self.sparse_select else (0.1, 0.7, 0.2)
            else:
                if index >= self.pop.size:
                    color = (1.0, 0.6, 0.6) if index in self.selected else (0.4, 0.4, 0.4)
                else:
                    color = 'r' if index in self.selected else 'k'

            opti.plot(*self.scores[:, index, 1], marker='o', color=color)
        opti.set_ylabel('Lazy')
        opti.set_xlabel('Busy')
        opti.set_title('Selected individuals')

        ticks = list(range(len(self.mean_ent_hist)))
        entropy.plot(ticks, self.mean_ent_hist)
        entropy.set_ylabel('Entropy')
        entropy.set_xlabel('Iteration')
        entropy.set_xticks(ticks)
        entropy.set_title('Top 10 mean entropy over iterations')

        fig.subplots_adjust(bottom=0.05, left=0.1, top=0.96, hspace=0.2, wspace=0.2)

        fig.savefig(self.path+'iteration_%d.png' % self.gen)
        plt.close(fig)

'''
R+
wr1 > wr2
aux1 ? aux2
=> f(wr1, aux1) > f(wr2, aux2), f(a,b) != f(a)

f(a,b) = a 
'''
