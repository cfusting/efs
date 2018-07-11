import random
import math
from copy import deepcopy

import numpy as np

from scipy.stats import skew

from efs.feature import Feature


def name_operation(operation, name):
        operation.__name__ = name
        return operation


class RangeOperation(Feature):

    def __init__(self, variable_type_indices, names, X, operation=None, begin_range_name=None,
                 end_range_name=None, original_variable=True, string=None):
        Feature.__init__(self, None, 'RangeOperation', 'RangeOperation', original_variable=original_variable)
        self.X = X
        self.begin_range = None
        self.end_range = None
        self.operation = None
        self.names = None
        self.lower_bound = None
        self.upper_bound = None
        self.variable_type_indices = variable_type_indices
        self.operations = {
            'sum': name_operation(np.sum, 'sum'),
            'min': name_operation(np.min, 'min'),
            'max': name_operation(np.max, 'max'),
            'mean': name_operation(np.mean, 'mean'),
            'vari': name_operation(np.var, 'vari'),
            'skew': name_operation(skew, 'skew')
        }
        if string:
            parts = string.split('_')
            self.initialize_parameters(variable_type_indices, names, parts[1], parts[2], parts[3])
        else:
            self.initialize_parameters(variable_type_indices, names, operation, begin_range_name, end_range_name)
        self.value = self.create_input_vector()
        self.string = self.format()
        self.infix_string = self.format()

    def __deepcopy__(self, memo):
        new = self.__class__(self.variable_type_indices, self.names, self.X)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        new.X = self.X
        new.value = self.value
        return new

    def initialize_parameters(self, variable_type_indices, names, operation=None, begin_range_name=None,
                              end_range_name=None):
        """
        :param variable_type_indices: A sequence of variable type indices where each entry defines the
        index of a variable type in the design matrix. For example a design matrix with two variable types will have
        indices [j,n] where variable type A spans 0 to j and variable type B spans j + 1 to n.
        :param names:
        :param operation
        :param begin_range_name
        :param end_range_name
        :return:
        """
        self.names = names
        for r in variable_type_indices:
            if r[1] - r[0] < 2:
                raise ValueError('Invalid variable type indices: ' + str(r))
        rng = random.choice(variable_type_indices)
        self.lower_bound = rng[0]
        self.upper_bound = rng[1]
        if operation is not None and begin_range_name is not None and end_range_name is not None:
            if self.operations.get(operation) is None:
                raise ValueError('Invalid operation provided to Range Terminal: ' + operation)
            if begin_range_name not in self.names:
                raise ValueError('Invalid range name provided to Range Termnial: ' + str(begin_range_name))
            if end_range_name not in self.names:
                raise ValueError('Invalid range name provided to Range Terminal: ' + str(end_range_name))
            begin_range = self.names.index(begin_range_name)
            end_range = self.names.index(end_range_name) + 1
            valid = False
            for r in variable_type_indices:
                if r[0] <= begin_range < end_range <= r[1]:
                    valid = True
            if not valid:
                raise ValueError('Invalid range provided to Range Terminal: (' + str(begin_range) + ',' +
                                 str(end_range) + ')')
            self.operation = self.operations[operation]
            self.begin_range = begin_range
            self.end_range = end_range
        else:
            self.operation = random.choice(list(self.operations.values()))
            self.begin_range = np.random.randint(self.lower_bound, self.upper_bound - 1)
            self.end_range = np.random.randint(self.begin_range + 1, self.upper_bound)

    def mutate_parameters(self):
        mutation = random.choice(['low', 'high'])
        span = self.end_range - self.begin_range
        if span == 0:
            span = 1
        value = random.gauss(0, math.sqrt(span))
        amount = int(math.ceil(abs(value)))
        if value < 0:
            amount *= -1
        if mutation == 'low':
            location = amount + self.begin_range
            if location < self.lower_bound:
                self.begin_range = self.lower_bound
            elif location > self.end_range - 2:
                self.begin_range = self.end_range - 2
            elif location > self.upper_bound - 2:
                self.begin_range = self.upper_bound - 2
            else:
                self.begin_range = location
        elif mutation == 'high':
            location = amount + self.end_range
            if location > self.upper_bound:
                self.end_range = self.upper_bound
            elif location < self.begin_range + 2:
                self.end_range = self.begin_range + 2
            elif location < self.lower_bound + 2:
                self.end_range = self.lower_bound + 2
            else:
                self.end_range = location
        self.value = self.create_input_vector()
        self.infix_string = self.format()
        self.string = self.format()

    def create_input_vector(self):
        array = self.X[:, self.begin_range:self.end_range]
        if array.shape[1] == 0:
            return np.zeros((array.shape[0], 1))
        else:
            return self.operation(array, axis=1)

    def format(self):
        return "RangeOperation_{}_{}_{}".format(self.operation.__name__, self.names[self.begin_range],
                                                self.names[self.end_range - 1])
