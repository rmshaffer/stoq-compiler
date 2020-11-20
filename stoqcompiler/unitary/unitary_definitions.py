'''
Defines the UnitaryDefinitions class.
'''
import numpy as np
import scipy.linalg

from .unitary import Unitary
from .unitary_sequence_entry import UnitarySequenceEntry


class UnitaryDefinitions:
    '''
    Provides methods to create several commonly-used unitaries.
    '''
    @staticmethod
    def rx(theta: float) -> Unitary:
        '''
        Rotation around the x-axis.

        :param theta: Angle of rotation.
        :type theta: float
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rx"
        return Unitary(dimension, np.array(
            [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
             [-1j * np.sin(theta / 2), np.cos(theta / 2)]]
        ), operation_name, parameter_dict)

    @staticmethod
    def ry(theta: float) -> Unitary:
        '''
        Rotation around the y-axis.

        :param theta: Angle of rotation.
        :type theta: float
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Ry"
        return Unitary(dimension, np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)],
             [np.sin(theta / 2), np.cos(theta / 2)]]
        ), operation_name, parameter_dict)

    @staticmethod
    def rphi(
        theta: float,
        phi: float
    ) -> Unitary:
        '''
        Rotation around an axis in the x-y plane.

        :param theta: Angle of rotation.
        :type theta: float
        :param phi: Angle defining axis of rotation.
        :type phi: float
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 2
        parameter_dict = {"θ": (theta, True), "Φ": (phi, True)}
        operation_name = "R"
        return Unitary(dimension, np.array(
            [[np.cos(theta / 2),
                np.exp(-1j * (np.pi / 2 + phi)) * np.sin(theta / 2)],
             [np.exp(-1j * (np.pi / 2 - phi)) * np.sin(theta / 2),
                np.cos(theta / 2)]]
        ), operation_name, parameter_dict)

    @staticmethod
    def rz(theta: float) -> Unitary:
        '''
        Rotation around the z-axis.

        :param theta: Angle of rotation.
        :type theta: float
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rz"
        return Unitary(dimension, np.array(
            [[np.exp(-1j * theta / 2), 0],
             [0, np.exp(1j * theta / 2)]]
        ), operation_name, parameter_dict)

    @staticmethod
    def h() -> Unitary:
        '''
        The single-qubit Hadamard gate.

        :return: The unitary object.
        :rtype: Unitary
        '''
        h = UnitaryDefinitions.rz(np.pi).left_multiply(
            UnitaryDefinitions.ry(np.pi / 2))
        return Unitary(h.get_dimension(), h.get_matrix(), "H")

    @staticmethod
    def t() -> Unitary:
        '''
        The single-qubit T gate.

        :return: The unitary object.
        :rtype: Unitary
        '''
        t = UnitaryDefinitions.rz(np.pi / 4)
        return Unitary(t.get_dimension(), t.get_matrix(), "T")

    @staticmethod
    def sigmax() -> Unitary:
        '''
        Rotation around the x-axis by pi.

        :return: The unitary object.
        :rtype: Unitary
        '''
        rx = UnitaryDefinitions.rx(np.pi)
        return Unitary(rx.get_dimension(), rx.get_matrix(), "X")

    @staticmethod
    def sigmay() -> Unitary:
        '''
        Rotation around the y-axis by pi.

        :return: The unitary object.
        :rtype: Unitary
        '''
        ry = UnitaryDefinitions.ry(np.pi)
        return Unitary(ry.get_dimension(), ry.get_matrix(), "Y")

    @staticmethod
    def sigmaz() -> Unitary:
        '''
        Rotation around the z-axis by pi.

        :return: The unitary object.
        :rtype: Unitary
        '''
        rz = UnitaryDefinitions.rz(np.pi)
        return Unitary(rz.get_dimension(), rz.get_matrix(), "Z")

    @staticmethod
    def cnot() -> Unitary:
        '''
        The two-qubit CNOT gate.

        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 4
        operation_name = "CNOT"
        return Unitary(dimension, np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]), operation_name)

    @staticmethod
    def ccnot() -> Unitary:
        '''
        The three-qubit CCNOT gate (or Toffoli gate).

        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 8
        operation_name = "CCNOT"
        return Unitary(dimension, np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0]]), operation_name)

    @staticmethod
    def qecc_phase_flip() -> Unitary:
        '''
        The three-qubit operation implementing syndrome
        detection for a phase-flip QECC scheme.

        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 8
        operation_name = "QECC"
        return Unitary(dimension, (1 / np.sqrt(8)) * np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, -1, -1, 1, -1, 1, 1, -1],
             [1, 1, -1, -1, 1, 1, -1, -1],
             [1, -1, 1, -1, -1, 1, -1, 1],
             [1, 1, 1, 1, -1, -1, -1, -1],
             [1, -1, -1, 1, 1, -1, -1, 1],
             [1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, -1, -1, -1, -1, 1, 1]]), operation_name)

    @staticmethod
    def xx(
        theta: float = np.pi / 4
    ) -> Unitary:
        '''
        The two-qubit XX gate, where a rotation angle of pi/4
        produces the traditional XX or Molmer-Sorensen gate.

        :param theta: Angle of rotation, defaults to np.pi/4.
        :type theta: float, optional
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 4
        parameter_dict = {"θ": (theta, True)}
        operation_name = "XX"
        return Unitary(dimension, np.array(
            [[np.cos(theta), 0, 0, -1j * np.sin(theta)],
             [0, np.cos(theta), -1j * np.sin(theta), 0],
             [0, -1j * np.sin(theta), np.cos(theta), 0],
             [-1j * np.sin(theta), 0, 0, np.cos(theta)]]
        ), operation_name, parameter_dict)

    @staticmethod
    def gms(
        num_qubits: int,
        theta: float = np.pi / 4
    ) -> Unitary:
        '''
        The n-qubit global Molmer-Sorensen gate, where a
        rotation angle of pi/4 produces the maximally-entangling
        version of the gate.

        :param num_qubits: Number of qubits.
        :type num_qubits: int
        :param theta: Angle of rotation, defaults to np.pi/4.
        :type theta: float, optional
        :return: The unitary object.
        :rtype: Unitary
        '''
        dimension = 2**num_qubits
        parameter_dict = {"θ": (theta, True)}
        operation_name = f"GMS{num_qubits}"
        local_unitaries = [UnitarySequenceEntry(
            UnitaryDefinitions.xx(theta), [i, j]).get_full_unitary(dimension)
            for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        global_unitary = Unitary.identity(dimension)
        for u in local_unitaries:
            global_unitary = global_unitary.left_multiply(u)
        return Unitary(
            dimension, global_unitary.get_matrix(),
            operation_name, parameter_dict)

    @staticmethod
    def time_evolution(
        h_matrix: np.ndarray,
        t: float,
        h_suffix: str = ""
    ) -> Unitary:
        '''
        Creates a time-evolution unitary for the specified
        Hamiltonian matrix.

        :param h_matrix: The Hamiltonian matrix to use for time evolution.
        :type h_matrix: np.ndarray
        :param t: The time for which to perform the time evolution.
            This may be negative if time-reversal is allowed.
        :type t: float
        :param h_suffix: A suffix used for display purposes to identify this
            Hamiltonian, defaults to "".
        :type h_suffix: str, optional
        :return: [description]
        :rtype: Unitary
        '''
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())

        dimension = h_matrix.shape[0]
        parameter_dict = {"t": (t, False)}
        operation_name = "H" + str(h_suffix)
        return Unitary(
            dimension, scipy.linalg.expm(1j * h_matrix * t),
            operation_name, parameter_dict)
