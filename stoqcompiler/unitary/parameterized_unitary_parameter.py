import numpy as np


class ParameterizedUnitaryParameter:

    def __init__(self, parameter_name, min_value, max_value, is_angle):
        assert parameter_name
        assert max_value >= min_value

        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_angle = is_angle

    def get_parameter_name(self):
        return self.parameter_name

    def get_is_angle(self):
        return self.is_angle

    def is_valid(self, parameter_value):
        return (parameter_value >= self.min_value
                and parameter_value <= self.max_value)

    def random_value(self):
        return (np.random.random_sample()
                * (self.max_value - self.min_value)) + self.min_value
