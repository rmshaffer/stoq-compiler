'''
Defines the ParameterizedUnitary class.
'''
import numpy as np
from typing import Callable, List

from .parameterized_unitary_parameter import ParameterizedUnitaryParameter
from .unitary import Unitary


class ParameterizedUnitary:
    '''
    Represents a unitary parameterized by one or more parameters.
    '''
    def __init__(
        self,
        dimension: int,
        parameterized_matrix: Callable[..., np.ndarray],
        parameters: List[ParameterizedUnitaryParameter],
        operation_name: str
    ):
        '''
        Creates a ParameterizedUnitary object.

        :param dimension: The dimension of the state space. For an n-qubit
        unitary, dimension should be set to 2**n.
        :param parameterized_matrix: A function that takes in the
        parameter values for this unitary and returns the matrix value when
        those parameter values are applied.
        :type parameterized_matrix: Callable[..., np.ndarray]
        :param parameters: The parameters with which this unitary is
        parameterized.
        :type parameters: List[ParameterizedUnitaryParameter]
        :param operation_name: The display name associated with this
        unitary operation.
        :type operation_name: str
        '''
        assert dimension > 0
        assert callable(parameterized_matrix)
        assert (isinstance(parameters, list)
                or isinstance(parameters, np.ndarray))
        assert np.all([
            isinstance(p, ParameterizedUnitaryParameter) for p in parameters])
        assert isinstance(operation_name, str)

        self.dimension = dimension
        self.parameterized_matrix = parameterized_matrix
        self.parameters = parameters
        self.operation_name = operation_name

    def get_dimension(self) -> int:
        '''
        Gets the dimension of the state space on which
        this unitary acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.dimension

    def get_parameters(self) -> List[ParameterizedUnitaryParameter]:
        '''
        Gets the list of parameters with which this unitary is
        parameterized.

        :return: The list of parameters.
        :rtype: List[ParameterizedUnitaryParameter]
        '''
        return self.parameters

    def get_operation_name(self) -> str:
        '''
        Gets the display name associated with this
        unitary operation.

        :return: The operation display name.
        :rtype: str
        '''
        return self.operation_name

    def as_unitary(
        self,
        parameter_values: List[float]
    ) -> Unitary:
        '''
        Gets the Unitary object resulting when the specified parameter
        values are applied to this parameterized unitary.

        :param parameter_values: The list of parameter values to apply.
        :type parameter_values: List[float]
        :return: The concrete Unitary object with the parameters applied.
        :rtype: Unitary
        '''
        assert (isinstance(parameter_values, list)
                or isinstance(parameter_values, np.ndarray))
        assert np.all([
            self.get_parameters()[i].is_valid(p)
            for i, p in enumerate(parameter_values)])

        parameter_dict = {
            p.get_parameter_name(): (parameter_values[i], p.get_is_angle())
            for i, p in enumerate(self.get_parameters())}
        return Unitary(
            self.get_dimension(),
            self.parameterized_matrix(*parameter_values),
            self.get_operation_name(),
            parameter_dict)
