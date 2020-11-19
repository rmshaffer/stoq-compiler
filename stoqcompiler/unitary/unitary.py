'''
Defines the Unitary class.
'''
import numpy as np
import scipy.stats
from typing import Dict, List, Optional, Tuple


class Unitary:
    def __init__(
        self,
        dimension: int,
        matrix: Optional[np.ndarray] = None,
        operation_name: str = None,
        parameter_dict: Optional[Dict[str, Tuple[float, bool]]] = None,
        is_inverse: bool = False,
        apply_to: List[int] = []
    ):
        assert dimension > 0

        if matrix is None:
            matrix = np.identity(dimension)
        assert isinstance(matrix, np.ndarray)
        matrix = matrix.astype(np.complex128)

        if operation_name is None:
            operation_name = "U"
        assert isinstance(operation_name, str)
        self.operation_name = operation_name

        if parameter_dict is None:
            parameter_dict = {}
        assert isinstance(parameter_dict, dict)
        self.parameter_dict = parameter_dict

        assert isinstance(is_inverse, bool)
        self.is_inverse = is_inverse

        assert isinstance(apply_to, list)
        self.apply_to = apply_to

        # clean up very small values
        matrix.real[np.abs(matrix.real) < 1e-12] = 0
        matrix.imag[np.abs(matrix.imag) < 1e-12] = 0

        # verify that the matrix is unitary
        assert matrix.shape == (dimension, dimension), matrix.shape
        assert np.allclose(matrix @ matrix.T.conj(), np.identity(dimension)), \
            matrix @ matrix.T.conj()

        # balance the global phase
        global_phase_factor = (1 / np.linalg.det(matrix)) ** (1 / dimension)
        matrix = global_phase_factor * matrix

        self.matrix = matrix

    @staticmethod
    def identity(dimension: int) -> 'Unitary':
        operation_name = "I"
        return Unitary(dimension, np.identity(dimension), operation_name)

    @staticmethod
    def random(dimension: int) -> 'Unitary':
        random_matrix = scipy.stats.unitary_group.rvs(dimension)
        return Unitary(dimension, random_matrix)

    def get_operation_name(self) -> str:
        return self.operation_name

    def get_jaqal(self) -> str:
        def as_decimal(value: float) -> str:
            return str(round(value, 7))

        parameters = ""
        if len(self.parameter_dict) > 0:
            parameters = " ".join(
                as_decimal(v[0]) for k, v in self.parameter_dict.items())

        qubits = ""
        apply_to_qubits = self.apply_to
        if len(apply_to_qubits) == 0:
            num_qubits = int(np.log2(self.get_dimension()))
            apply_to_qubits = list(range(num_qubits))

        qubits = " ".join("q[" + str(q) + "]" for q in apply_to_qubits)

        return self.operation_name + " " + qubits + " " + parameters

    def get_qasm(self) -> str:
        def as_pi_fraction(value: float) -> str:
            return "pi*" + str(round(value / np.pi, 7))

        def as_decimal(value: float) -> str:
            return str(round(value, 7))

        parameters = ""
        if len(self.parameter_dict) > 0:
            parameters = "(" + ",".join(
                (as_pi_fraction(v[0]) if v[1] else as_decimal(v[0]))
                for k, v in self.parameter_dict.items()) + ")"

        qubits = ""
        apply_to_qubits = self.apply_to
        if len(apply_to_qubits) == 0:
            num_qubits = int(np.log2(self.get_dimension()))
            apply_to_qubits = list(range(num_qubits))

        qubits = "\t" + ",".join("q[" + str(q) + "]" for q in apply_to_qubits)

        return self.operation_name + parameters + qubits + ";"

    def get_display_name(self) -> str:
        def as_pi_fraction(value: float) -> str:
            return str(round(value / np.pi, 3)) + "π"

        def as_decimal(value: float) -> str:
            return str(round(value, 4))

        display_name = self.operation_name
        if len(self.apply_to) > 0:
            display_name += str(self.apply_to)
        if len(self.parameter_dict) > 0:
            display_name += "(" + ", ".join(
                k + "=" + (as_pi_fraction(v[0]) if v[1] else as_decimal(v[0]))
                for k, v in self.parameter_dict.items()) + ")"

        if self.is_inverse:
            display_name += '†'

        return display_name

    def get_dimension(self) -> int:
        return self.matrix.shape[0]

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def get_parameter_value(
        self,
        key: str
    ) -> Optional[Tuple[float, bool]]:
        if key in self.parameter_dict:
            return self.parameter_dict[key]
        return None

    def inverse(self) -> 'Unitary':
        is_inverse = not self.is_inverse
        return Unitary(
            self.get_dimension(), self.get_matrix().T.conj(),
            self.get_operation_name(), self.parameter_dict,
            is_inverse=is_inverse, apply_to=self.apply_to)

    def tensor(
        self,
        u: 'Unitary'
    ) -> 'Unitary':
        assert isinstance(u, Unitary)

        new_dimension = u.get_dimension() * self.get_dimension()
        new_operation_name = (
            self.get_operation_name()
            + "--"
            + u.get_operation_name())
        return Unitary(
            new_dimension,
            np.kron(self.get_matrix(), u.get_matrix()),
            new_operation_name, self.parameter_dict, apply_to=self.apply_to)

    def close_to(
        self,
        u: 'Unitary',
        threshold: Optional[float] = None
    ) -> bool:
        distance = self.distance_from(u)
        if threshold:
            max_distance = 1.0 - threshold
            return distance <= max_distance

        return np.isclose(distance, 0.0)

    def distance_from(
        self,
        u: 'Unitary'
    ) -> float:
        if isinstance(u, Unitary):
            u = u.get_matrix()

        assert isinstance(u, np.ndarray)
        assert u.shape == (self.get_dimension(), self.get_dimension())

        self_dag_u = self.inverse().get_matrix() @ u
        trace = np.trace(self_dag_u)
        normalized_trace = np.linalg.norm(trace) / self.get_dimension()
        return 1.0 - normalized_trace

    def left_multiply(
        self,
        factor: 'Unitary'
    ) -> 'Unitary':
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(
            self.get_dimension(), factor.get_matrix() @ self.get_matrix())

    def right_multiply(
        self,
        factor: 'Unitary'
    ) -> 'Unitary':
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(
            self.get_dimension(), self.get_matrix() @ factor.get_matrix())
