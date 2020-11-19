'''
Defines the ParameterizedUnitaryDefinitions class.
'''
import numpy as np

from .unitary_definitions import UnitaryDefinitions
from .parameterized_unitary_parameter import ParameterizedUnitaryParameter
from .parameterized_unitary import ParameterizedUnitary


class ParameterizedUnitaryDefinitions:
    '''
    Provides methods to create several commonly-used
    parameterized unitaries.
    '''
    @staticmethod
    def rotation_xy() -> ParameterizedUnitary:
        '''
        Creates a parameterized rotation around an axis in the x-y plane with
        parameters theta (rotation angle) and phi (polar angle defining
        the axis of rotation).

        :return: The parameterized unitary object.
        :rtype: ParameterizedUnitary
        '''
        qubit_dimension = 2

        def rotation_matrix(
            theta: float,
            phi: float
        ) -> np.ndarray:
            return UnitaryDefinitions.rphi(theta, phi).get_matrix()

        min_value = 0
        max_value = 2 * np.pi
        parameters = [ParameterizedUnitaryParameter(
                      "theta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                      "phi", min_value, max_value, is_angle=True)]
        operation_name = "Rxy"
        return ParameterizedUnitary(
            qubit_dimension, rotation_matrix, parameters, operation_name)

    @staticmethod
    def rotation_xyz() -> ParameterizedUnitary:
        '''
        Creates a parameterized rotation around an arbitrary axis, where the
        rotation is defined by parameters alpha, beta, and gamma.

        :return: The parameterized unitary object.
        :rtype: ParameterizedUnitary
        '''
        qubit_dimension = 2

        def rotation_matrix(
            alpha: float,
            beta: float,
            gamma: float
        ) -> np.ndarray:
            return np.array(
                [[np.cos(beta / 2) * np.exp(-1j * (alpha + gamma) / 2),
                    -np.sin(beta / 2) * np.exp(-1j * (alpha - gamma) / 2)],
                 [np.sin(beta / 2) * np.exp(1j * (alpha - gamma) / 2),
                    np.cos(beta / 2) * np.exp(1j * (alpha + gamma) / 2)]])

        min_value = 0
        max_value = 2 * np.pi
        parameters = [ParameterizedUnitaryParameter(
                      "alpha", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                      "beta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter(
                      "gamma", min_value, max_value, is_angle=True)]
        operation_name = "Rxyz"
        return ParameterizedUnitary(
            qubit_dimension, rotation_matrix, parameters, operation_name)

    @staticmethod
    def xx() -> ParameterizedUnitary:
        '''
        Creates a parameterized two-qubit XX gate, where the rotation
        angle is defined by the parameter theta. A theta value of pi/4
        produces the traditional XX or Molmer-Sorensen gate.

        :return: The parameterized unitary object.
        :rtype: ParameterizedUnitary
        '''
        qubit_dimension = 2

        def xx_matrix(
            theta: float
        ) -> ParameterizedUnitary:
            return UnitaryDefinitions.xx(theta).get_matrix()

        min_value = 0
        max_value = 2 * np.pi
        parameters = [ParameterizedUnitaryParameter(
                      "theta", min_value, max_value, is_angle=True)]
        operation_name = "XX"
        return ParameterizedUnitary(
            qubit_dimension ** 2, xx_matrix, parameters, operation_name)

    @staticmethod
    def gms(
        num_qubits: int
    ) -> ParameterizedUnitary:
        '''
        Creates a parameterized n-qubit global Molmer-Sorensen gate,
        where the rotation angle is defined by the parameter theta.
        A theta value of pi/4 produces the traditional n-qubit global
        Molmer-Sorensen maximally-entangling gate.

        :param num_qubits: The number of qubits to use.
        :type num_qubits: int
        :return: The parameterized unitary object.
        :rtype: ParameterizedUnitary
        '''
        qubit_dimension = 2

        def gms_matrix(
            theta: float
        ) -> np.ndarray:
            return UnitaryDefinitions.gms(num_qubits, theta).get_matrix()

        min_value = 0
        max_value = 2 * np.pi
        parameters = [ParameterizedUnitaryParameter(
                      "theta", min_value, max_value, is_angle=True)]
        operation_name = f"GMS{num_qubits}"
        return ParameterizedUnitary(
            qubit_dimension ** num_qubits, gms_matrix,
            parameters, operation_name)

    @staticmethod
    def time_evolution(
        h_matrix: np.ndarray,
        t_min: float,
        t_max: float,
        h_suffix: str = ""
    ) -> ParameterizedUnitary:
        '''
        Creates a parameterized time-evolution unitary
        for the specified Hamiltonian matrix, where the
        parameter is the time for which to evolve the system.

        :param h_matrix: The Hamiltonian matrix to use for time evolution.
        :type h_matrix: np.ndarray
        :param t_min: The minimum allowed value for the time evolution
        parameter. This may be negative if time-reversal is allowed.
        :type t_min: float
        :param t_max: The maximum allowed value for the time evolution
        parameter.
        :type t_max: float
        :param h_suffix: A suffix used for display purposes to identify this
        Hamiltonian, defaults to "".
        :type h_suffix: str, optional
        :return: The parameterized unitary object.
        :rtype: ParameterizedUnitary
        '''
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())
        assert t_min <= t_max

        dimension = h_matrix.shape[0]

        def u_matrix(
            t: float
        ) -> np.ndarray:
            return UnitaryDefinitions.time_evolution(
                h_matrix, t, h_suffix).get_matrix()

        min_value = t_min
        max_value = t_max
        parameters = [ParameterizedUnitaryParameter(
                      "t", min_value, max_value, is_angle=False)]
        operation_name = "H" + h_suffix
        return ParameterizedUnitary(
            dimension, u_matrix, parameters, operation_name)
