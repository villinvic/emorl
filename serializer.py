import pickle


class Serializer:
    def __init__(self, path='checkpoint/'):
        self.path = path

    def dump(self, population, ckpt_name):
        full_path = self.path + ckpt_name
        with open(full_path, 'wb') as f:
            pickle.dump(population, f)
        print('Successfully saved population.')

    def load(self, ckpt_name):
        full_path = self.path + ckpt_name
        with open(full_path, 'rb') as f:
            pop = pickle.load(f)

        print('Successfully loaded population.')

        return pop
