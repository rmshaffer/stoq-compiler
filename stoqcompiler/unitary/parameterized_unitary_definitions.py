import numpy as np

from .unitary_definitions import UnitaryDefinitions
from .parameterized_unitary_parameter import ParameterizedUnitaryParameter
from .parameterized_unitary import ParameterizedUnitary

class ParameterizedUnitaryDefinitions:
    @staticmethod
    def rotation_xy():
        qubit_dimension = 2
        rotation_matrix = lambda theta, phi : UnitaryDefinitions.rphi(theta, phi).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter("phi", min_value, max_value, is_angle=True)]
        operation_name = "Rxy"
        return ParameterizedUnitary(qubit_dimension, rotation_matrix, parameters, operation_name)

    @staticmethod
    def rotation_xyz():
        qubit_dimension = 2
        rotation_matrix = lambda alpha, beta, gamma : np.array(
           [[np.cos(beta/2) * np.exp(-1j*(alpha+gamma)/2), -np.sin(beta/2) * np.exp(-1j*(alpha-gamma)/2)],
            [np.sin(beta/2) * np.exp(1j*(alpha-gamma)/2), np.cos(beta/2) * np.exp(1j*(alpha+gamma)/2)]])
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("alpha", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter("beta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter("gamma", min_value, max_value, is_angle=True)]
        operation_name = "Rxyz"
        return ParameterizedUnitary(qubit_dimension, rotation_matrix, parameters, operation_name)

    @staticmethod
    def xx():
        qubit_dimension = 2
        xx_matrix = lambda theta : UnitaryDefinitions.xx(theta).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True)]
        operation_name = "XX"
        return ParameterizedUnitary(qubit_dimension ** 2, xx_matrix, parameters, operation_name)

    @staticmethod
    def gms(num_qubits):
        qubit_dimension = 2
        gms_matrix = lambda theta : UnitaryDefinitions.gms(num_qubits, theta).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True)]
        operation_name = f"GMS{num_qubits}"
        return ParameterizedUnitary(qubit_dimension ** num_qubits, gms_matrix, parameters, operation_name)

    @staticmethod
    def time_evolution(h_matrix, t_min, t_max, h_suffix=""):
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())
        assert t_min <= t_max

        dimension = h_matrix.shape[0]
        u_matrix = lambda t : UnitaryDefinitions.time_evolution(h_matrix, t, h_suffix).get_matrix()
        min_value = t_min
        max_value = t_max
        parameters = [ParameterizedUnitaryParameter("t", min_value, max_value, is_angle=False)]
        operation_name = "H" + h_suffix
        return ParameterizedUnitary(dimension, u_matrix, parameters, operation_name)
