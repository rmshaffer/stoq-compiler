'''
Defines the Unitary class.
'''
import numpy as np
import scipy.stats
from typing import Dict, List, Optional, Tuple


class Unitary:
    '''
    Represents a unitary operation.

    :param dimension: The dimension of the state space. For an n-qubit
        unitary, dimension should be set to 2**n.
    :type dimension: int
    :param matrix: The unitary matrix representing this operation,
        defaults to None. If not specified, an identity matrix is used.
    :type matrix: Optional[np.ndarray], optional
    :param operation_name: The display name associated with this
        unitary operation.
    :type operation_name: str
    :param parameter_dict: The display parameters associated with this
        unitary operation, as a dictionary mapping the parameter name
        to its parameter value and whether it is an angle, defaults
        to None.
    :type parameter_dict: Optional[Dict[str, Tuple[float, bool]]], optional
    :param is_inverse: Whether to display this unitary as an inverse,
        defaults to False.
    :type is_inverse: bool, optional
    :param apply_to: The qubits to which this unitary is applied,
        defaults to [].
    :type apply_to: List[int], optional
    '''
    def __init__(
        self,
        dimension: int,
        matrix: Optional[np.ndarray] = None,
        operation_name: str = None,
        parameter_dict: Optional[Dict[str, Tuple[float, bool]]] = None,
        is_inverse: bool = False,
        apply_to: List[int] = []
    ):
        '''
        Creates a Unitary object.
        '''
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
        '''
        The identity operator of the given dimension.

        :param dimension: The dimension of the state space. For an n-qubit
            unitary, dimension should be set to 2**n.
        :type dimension: int
        :return: The identity operator.
        :rtype: Unitary
        '''
        operation_name = "I"
        return Unitary(dimension, np.identity(dimension), operation_name)

    @staticmethod
    def random(dimension: int) -> 'Unitary':
        '''
        A randomly-generated operator acting on the given dimension.

        :param dimension: The dimension of the state space. For an n-qubit
            unitary, dimension should be set to 2**n.
        :type dimension: int
        :return: The randomly-generated unitary operator.
        :rtype: Unitary
        '''
        random_matrix = scipy.stats.unitary_group.rvs(dimension)
        return Unitary(dimension, random_matrix)

    def get_operation_name(self) -> str:
        '''
        Gets the name associated with this unitary operation.

        :return: The operation name.
        :rtype: str
        '''
        return self.operation_name

    def get_jaqal(self) -> str:
        '''
        Returns a JAQAL-like representation of this unitary operation.
        This is a minimal-effort function and is not guaranteed to
        be a valid JAQAL statement (and very likely will not be).

        :return: The JAQAL representation of this unitary operation.
        :rtype: str
        '''
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
        '''
        Returns a QASM-like representation of this unitary operation.
        This is a minimal-effort function and is not guaranteed to
        be a valid QASM statement (and very likely will not be).

        :return: The QASM representation of this unitary operation.
        :rtype: str
        '''
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
        '''
        Gets the display string, including parameters, associated with
        this unitary operation.

        :return: The operation display string.
        :rtype: str
        '''
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
        '''
        Gets the dimension of the state space on which
        this unitary acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.matrix.shape[0]

    def get_matrix(self) -> np.ndarray:
        '''
        Gets the unitary matrix associated with this operation.

        :return: The unitary matrix.
        :rtype: np.ndarray
        '''
        return self.matrix

    def get_parameter_value(
        self,
        key: str
    ) -> Optional[Tuple[float, bool]]:
        '''
        Gets the value of the specified parameter, along
        with whether the parameter represents an angle.

        :param key: The parameter name.
        :type key: str
        :return: A tuple containing the parameter value and
            whether the parameter represents an angle.
        :rtype: Optional[Tuple[float, bool]]
        '''
        if key in self.parameter_dict:
            return self.parameter_dict[key]
        return None

    def inverse(self) -> 'Unitary':
        '''
        Gets the inverse of this unitary operation.

        :return: The inverse unitary object.
        :rtype: Unitary
        '''
        is_inverse = not self.is_inverse
        return Unitary(
            self.get_dimension(), self.get_matrix().T.conj(),
            self.get_operation_name(), self.parameter_dict,
            is_inverse=is_inverse, apply_to=self.apply_to)

    def tensor(
        self,
        u: 'Unitary'
    ) -> 'Unitary':
        '''
        Gets the unitary representing the tensor product of this
        unitary with the passed-in unitary.

        :return: The unitary representing the tensor product.
        :rtype: Unitary
        '''
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
        '''
        Determines whether the provided unitary is close to
        the current unitary, optionally using the specified threshold.

        :param u: The unitary to compare to this one.
        :type u: Unitary
        :param threshold: The maximum distance between unitaries that
            should still be considered "close", defaults to None.
        :type threshold: Optional[float], optional
        :return: Whether the distance between the unitaries is within
            the given threshold, or within numpy's default tolerance if
            no threshold is provided.
        :rtype: bool
        '''
        distance = self.distance_from(u)
        if threshold is not None:
            max_distance = 1.0 - threshold
            return distance <= max_distance

        return np.isclose(distance, 0.0)

    def distance_from(
        self,
        u: 'Unitary'
    ) -> float:
        '''
        Calculates the distance between the given unitary and this one.

        :param u: The unitary to compare.
        :type u: Unitary
        :return: The Hilbert-Schmidt distance, normalized so that the
            result is between 0 and 1.
        :rtype: float
        '''
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
        '''
        Returns the product when multiplying the specified factor
        from the left.

        :return: The product.
        :rtype: Unitary.
        '''
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(
            self.get_dimension(), factor.get_matrix() @ self.get_matrix())

    def right_multiply(
        self,
        factor: 'Unitary'
    ) -> 'Unitary':
        '''
        Returns the product when multiplying the specified factor
        from the right.

        :return: The product.
        :rtype: Unitary.
        '''
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(
            self.get_dimension(), self.get_matrix() @ factor.get_matrix())
