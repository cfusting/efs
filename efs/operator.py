import random

import numpy as np


def square(x):
    return np.power(x, 2)


def cube(x):
    return np.power(x, 3)


def is_huge(x):
    return x > np.finfo(np.float64).max / 100000


def numpy_safe_exp(x):
    with np.errstate(invalid='ignore'):
        result = np.exp(x)
        if isinstance(result, np.ndarray):
            result[np.isnan(x)] = 1
            result[np.isinf(x)] = 1
            result[is_huge(x)] = 1
        elif np.isinf(result):
            result = 1
        elif np.isnan(x):
            result = 1
        elif is_huge(x):
            result = 1
        return result


def numpy_protected_div_dividend(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = left[np.isinf(x)]
            x[np.isnan(x)] = left[np.isnan(x)]
        elif np.isinf(x) or np.isnan(x):
            x = left
    return x


def numpy_protected_log_one(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(np.abs(x))
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1.0
            x[np.isnan(x)] = 1.0
        elif np.isinf(x):
            x = 1.0
        elif np.isnan(x):
            x = 1.0
    return x


def numpy_protected_sqrt(x):
    with np.errstate(invalid='ignore'):
        x = np.sqrt(x)
        if isinstance(x, np.ndarray):
            x[np.isnan(x)] = 0
        elif np.isnan(x):
            x = 0
        return x


class Operator:

    def __init__(self, operation, parity, string, infix, infix_name, weight=1):
        self.operation = operation
        self.parity = parity
        self.string = string
        self.infix = infix
        self.infix_name = infix_name
        self.weight = weight


class OperatorDistribution:

    def __init__(self):
        self.operators_map = None
        self.weights = None
        self.weights_are_current = False

    def add(self, operator):
        self.operators_map[operator.infix_name] = operator
        self.weights_are_current = False

    def get_random(self, k=1):
        if self.weights_are_current:
            weights = self.weights
        else:
            weights = list(map(lambda x: self.operators_map[x].weight, self.operators_map))
            self.weights_are_current = True
        return random.choices(self.operators_map, weights=weights, k=k)

    def get(self, key):
        return self.operators_map[key]

    def contains(self, key):
        return key in self.operators_map.keys()


default_operators = OperatorDistribution()
ops = [
    Operator(np.add, 2, '({0} + {1})', 'add({0},{1})', 'add'),
    Operator(np.subtract, 2, '({0} - {1})', 'sub({0},{1})', 'sub'),
    Operator(np.multiply, 2, '({0} * {1})', 'mul({0},{1})', 'mul'),
    Operator(numpy_protected_div_dividend, 2, '({0} / {1})', 'div({0},{1})', 'div'),
    Operator(numpy_safe_exp, 1, 'exp({0})', 'exp({0})', 'exp'),
    Operator(numpy_protected_log_one, 1, 'log({0})', 'log({0})', 'log'),
    Operator(square, 1, 'sqr({0})', 'sqr({0})', 'sqr'),
    Operator(numpy_protected_sqrt, 1, 'sqt({0})', 'sqt({0})', 'sqt'),
    Operator(cube, 1, 'cbe({0})', 'cbe({0})', 'cbe'),
    Operator(np.cbrt, 1, 'cbt({0})', 'cbt({0})', 'cbt'),
    Operator(None, None, None, None, 'mutate'),
    Operator(None, None, None, None, 'transition')
]
map(lambda x: default_operators.add(x), ops)
