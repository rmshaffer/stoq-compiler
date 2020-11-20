'''
Defines the ParameterizedUnitaryParameter class.
'''
import numpy as np


class ParameterizedUnitaryParameter:
    '''
    Represents an individual parameter for a parameterized unitary.

    :param parameter_name: The name of the parameter.
    :type parameter_name: str
    :param min_value: The minimum allowed value of the parameter.
    :type min_value: float
    :param max_value: The maximum allowed value of the parameter.
    :type max_value: float
    :param is_angle: Whether the parameter represents an angle.
    :type is_angle: bool
    '''

    def __init__(
        self,
        parameter_name: str,
        min_value: float,
        max_value: float,
        is_angle: bool
    ):
        '''
        Creates a ParameterizedUnitaryParameter object.
        '''
        assert parameter_name
        assert max_value >= min_value

        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_angle = is_angle

    def get_parameter_name(self) -> str:
        '''
        Gets the name of the parameter.

        :return: The parameter name.
        :rtype: str
        '''
        return self.parameter_name

    def get_is_angle(self) -> bool:
        '''
        Gets whether this parameter represents an angle.

        :return: Whether this parameter represents an angle.
        :rtype: bool
        '''
        return self.is_angle

    def is_valid(
        self,
        parameter_value: float
    ) -> bool:
        '''
        Determines whether the given parameter value is valid
        for this parameter.

        :param parameter_value: The proposed parameter value.
        :type parameter_value: float
        :return: Whether the proposed value is valid.
        :rtype: bool
        '''
        return (parameter_value >= self.min_value
                and parameter_value <= self.max_value)

    def random_value(self) -> float:
        '''
        Gets a random parameter value in the valid range of
        values for this parameter.

        :return: The random parameter value.
        :rtype: float
        '''
        return (np.random.random_sample()
                * (self.max_value - self.min_value)) + self.min_value
