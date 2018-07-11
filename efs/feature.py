class Feature:

    def __init__(self, value, string, infix_string, size=0, fitness=1, original_variable=False):
        self.value = value
        self.fitness = fitness
        self.string = string
        self.infix_string = infix_string
        self.size = size
        self.original_variable = original_variable

    def __str__(self):
        return self.string
