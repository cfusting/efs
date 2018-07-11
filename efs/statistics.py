class Statistics:

    def __init__(self):
        self.scores = []
        self.generations = []
        self.num_features = []
        self.index = 0

    def add(self, gen, score, num_features):
        self.generations.append(gen)
        self.scores.append(score)
        self.num_features.append(num_features)
