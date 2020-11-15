import numpy as np
import scipy.linalg

from .unitary import Unitary
from .unitary_sequence_entry import UnitarySequenceEntry

class UnitaryDefinitions:
    @staticmethod
    def rx(theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rx"
        return Unitary(dimension, np.array(
            [[np.cos(theta/2), -1j * np.sin(theta/2)],
             [-1j * np.sin(theta/2), np.cos(theta/2)]]), operation_name, parameter_dict)

    @staticmethod
    def ry(theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Ry"
        return Unitary(dimension, np.array(
            [[np.cos(theta/2), -np.sin(theta/2)],
             [np.sin(theta/2), np.cos(theta/2)]]
            ), operation_name, parameter_dict)

    @staticmethod
    def rphi(theta, phi):
        dimension = 2
        parameter_dict = {"θ": (theta, True), "Φ": (phi, True)}
        operation_name = "R"
        return Unitary(dimension, np.array(
            [[np.cos(theta/2), np.exp(-1j*(np.pi/2+phi))*np.sin(theta/2)],
             [np.exp(-1j*(np.pi/2-phi))*np.sin(theta/2), np.cos(theta/2)]]
            ), operation_name, parameter_dict)

    @staticmethod
    def rz(theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rz"
        return Unitary(dimension, np.array(
            [[np.exp(-1j*theta/2), 0],
             [0, np.exp(1j*theta/2)]]
            ), operation_name, parameter_dict)

    @staticmethod
    def h():
        h = UnitaryDefinitions.rz(np.pi).left_multiply(UnitaryDefinitions.ry(np.pi/2))
        return Unitary(h.get_dimension(), h.get_matrix(), "H")

    @staticmethod
    def t():
        t = UnitaryDefinitions.rz(np.pi/4)
        return Unitary(t.get_dimension(), t.get_matrix(), "T")

    @staticmethod
    def sigmax():
        rx = UnitaryDefinitions.rx(np.pi)
        return Unitary(rx.get_dimension(), rx.get_matrix(), "X")

    @staticmethod
    def sigmay():
        ry = UnitaryDefinitions.ry(np.pi)
        return Unitary(ry.get_dimension(), ry.get_matrix(), "Y")

    @staticmethod
    def sigmaz():
        rz = UnitaryDefinitions.rz(np.pi)
        return Unitary(rz.get_dimension(), rz.get_matrix(), "Z")

    @staticmethod
    def cnot():
        dimension = 4
        operation_name = "CNOT"
        return Unitary(dimension, np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]), operation_name)

    @staticmethod
    def ccnot():
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
    def qecc_phase_flip():
        dimension = 8
        operation_name = "QECC"
        return Unitary(dimension, (1/np.sqrt(8)) * np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, -1, -1, 1, -1, 1, 1, -1],
             [1, 1, -1, -1, 1, 1, -1, -1],
             [1, -1, 1, -1, -1, 1, -1, 1],
             [1, 1, 1, 1, -1, -1, -1, -1],
             [1, -1, -1, 1, 1, -1, -1, 1],
             [1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, -1, -1, -1, -1, 1, 1]]), operation_name)

    @staticmethod
    def xx(theta=np.pi/4):
        dimension = 4
        parameter_dict = {"θ": (theta, True)}
        operation_name = "XX"
        return Unitary(dimension, np.array(
            [[np.cos(theta), 0, 0, -1j*np.sin(theta)],
             [0, np.cos(theta), -1j*np.sin(theta), 0],
             [0, -1j*np.sin(theta), np.cos(theta), 0],
             [-1j*np.sin(theta), 0, 0, np.cos(theta)]]
            ), operation_name, parameter_dict)

    @staticmethod
    def gms(num_qubits, theta=np.pi/4):
        dimension = 2**num_qubits
        parameter_dict = {"θ": (theta, True)}
        operation_name = f"GMS{num_qubits}"
        local_unitaries = [UnitarySequenceEntry(UnitaryDefinitions.xx(theta), [i,j]).get_full_unitary(dimension) for i in range(num_qubits) for j in range(i+1, num_qubits)]
        global_unitary = Unitary.identity(dimension)
        for u in local_unitaries:
            global_unitary = global_unitary.left_multiply(u)
        return Unitary(dimension, global_unitary.get_matrix(), operation_name, parameter_dict)

    @staticmethod
    def time_evolution(h_matrix, t, h_suffix=""):
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())

        dimension = h_matrix.shape[0]
        parameter_dict = {"t": (t, False)}
        operation_name = "H" + str(h_suffix)
        return Unitary(dimension, scipy.linalg.expm(1j*h_matrix*t), operation_name, parameter_dict)
