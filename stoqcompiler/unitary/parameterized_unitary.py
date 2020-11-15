import numpy as np

from .parameterized_unitary_parameter import ParameterizedUnitaryParameter
from .unitary import Unitary

class ParameterizedUnitary:
    def __init__(self, dimension, parameterized_matrix, parameters, operation_name):
        assert dimension > 0
        assert callable(parameterized_matrix)
        assert isinstance(parameters, list) or isinstance(parameters, np.ndarray)
        assert np.all([isinstance(p, ParameterizedUnitaryParameter) for p in parameters])
        assert isinstance(operation_name, str)

        self.dimension = dimension
        self.parameterized_matrix = parameterized_matrix
        self.parameters = parameters
        self.operation_name = operation_name

    def get_dimension(self):
        return self.dimension

    def get_parameters(self):
        return self.parameters

    def get_operation_name(self):
        return self.operation_name

    def as_unitary(self, parameter_values):
        assert isinstance(parameter_values, list) or isinstance(parameter_values, np.ndarray)
        assert np.all([self.get_parameters()[i].is_valid(p) for i,p in enumerate(parameter_values)])

        parameter_dict = {p.get_parameter_name(): (parameter_values[i], p.get_is_angle()) for i,p in enumerate(self.get_parameters())}
        return Unitary(self.get_dimension(), self.parameterized_matrix(*parameter_values), self.get_operation_name(), parameter_dict)
