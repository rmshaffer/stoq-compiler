import numpy as np

from .unitary_definitions import UnitaryDefinitions
from .parameterized_unitary_parameter import ParameterizedUnitaryParameter
from .parameterized_unitary import ParameterizedUnitary


class ParameterizedUnitaryDefinitions:
    @staticmethod
    def rotation_xy() -> ParameterizedUnitary:
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
