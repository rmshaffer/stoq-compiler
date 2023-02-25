import numpy as np

import pygsti

from qscout.v1.native_gates import U_R, U_Rz, U_MS, NATIVE_GATES
from jaqalpaq.emulator.pygsti import AbstractNoisyNativeEmulator


class JaqalNoisyEmulator(AbstractNoisyNativeEmulator):
    # This tells AbstractNoisyNativeEmulator what gate set we're modeling:
    jaqal_gates = NATIVE_GATES

    def __init__(self, *args, **kwargs):
        """Builds a JaqalNoisyEmulator instance for particular parameters

        :param depolarization float: (default 0.0) The depolarization during one pi/2
          R gate.
        :param r_rotation_error float: (default 0.0) The over-rotation angle during one
          pi/2 R gate.
        :param r_phase_error: (default 0.0) The error in the x-y angle for each R
          rotation gate, as a fraction of 2*pi radians.
        :param ms_rotation_error float: (default 0.0) The over-rotation angle during one
          pi/2 MS gate.
        :param ms_phase_error: (default 0.0) The error in the x-y angle for each MS
          gate, as a fraction of 2*pi radians.
        """
        # Equivalent to
        # self.depolarization = kwargs.pop('depolarization', 0.0)
        # ...
        self.set_defaults(
            kwargs,
            depolarization=0.0,
            r_rotation_error=0.0,
            r_phase_error=0.0,
            ms_rotation_error=0.0,
            ms_phase_error=0.0
        )

        # Pass through the balance of the parameters to AbstractNoisyNativeEmulator
        # In particular: passes the number of qubits to emulated (in args)
        super().__init__(*args, **kwargs)

    # For every gate, we need to specify a superoperator and a duration:

    # GJR
    def gateduration_R(self, q, axis_angle, rotation_angle):
        return np.abs(rotation_angle) / (np.pi / 2)

    def gate_R(self, q, axis_angle, rotation_angle):
        # We model the depolarization as a function of the gate duration:
        duration = self.gateduration_R(q, axis_angle, rotation_angle)
        depolarization_term = (1 - self.depolarization) ** duration

        # Combine these all, returning a superoperator in the Pauli basis
        return pygsti.unitary_to_pauligate(
            U_R(axis_angle + 2 * np.pi * self.r_phase_error, rotation_angle + self.r_rotation_error * np.sign(rotation_angle))
        ) @ np.diag([1, depolarization_term, depolarization_term, depolarization_term])

    # GJMS
    def gateduration_MS(self, q0, q1, axis_angle, rotation_angle):
        # Assume MS pi/2 gate 10 times longer than Sx, Sy, Sz
        return 10 * np.abs(rotation_angle) / (np.pi / 2)

    def gate_MS(self, q0, q1, axis_angle, rotation_angle):
        # We model the depolarization as a function of the gate duration:
        duration = self.gateduration_MS(q0, q1, axis_angle, rotation_angle)
        depolarization_term = (1 - self.depolarization) ** duration

        return pygsti.unitary_to_pauligate(
            U_MS(axis_angle + 2 * np.pi * self.ms_phase_error, rotation_angle + self.ms_rotation_error * np.sign(rotation_angle))
        ) @ np.kron(
            np.diag([1] + 3 * [depolarization_term]), np.diag([1] + 3 * [depolarization_term])
        )

    # Rz is performed entirely in software.
    # GJRz
    def gateduration_Rz(self, q, angle):
        return 0

    def gate_Rz(self, q, angle):
        return pygsti.unitary_to_pauligate(U_Rz(angle))

    # A process matrix for the idle behavior of a qubit.
    # Gidle
    def idle(self, q, duration):
        depolarization_term = (1 - self.depolarization) ** duration

        return np.diag([1, depolarization_term, depolarization_term, depolarization_term])

    # Instead of copy-pasting the above definitions, use _curry to create new methods
    # with some arguments.  None is a special argument that means: require an argument
    # in the created function and pass it through.
    C = AbstractNoisyNativeEmulator._curry

    gateduration_Rx, gate_Rx = C((None, 0.0, None), gateduration_R, gate_R)
    gateduration_Ry, gate_Ry = C((None, np.pi / 2, None), gateduration_R, gate_R)
    gateduration_Px, gate_Px = C((None, 0.0, np.pi), gateduration_R, gate_R)
    gateduration_Py, gate_Py = C((None, np.pi / 2, np.pi), gateduration_R, gate_R)
    gateduration_Pz, gate_Pz = C((None, np.pi), gateduration_Rz, gate_Rz)
    gateduration_Sx, gate_Sx = C((None, 0.0, np.pi / 2), gateduration_R, gate_R)
    gateduration_Sy, gate_Sy = C((None, np.pi / 2, np.pi / 2), gateduration_R, gate_R)
    gateduration_Sz, gate_Sz = C((None, np.pi / 2), gateduration_Rz, gate_Rz)
    gateduration_Sxd, gate_Sxd = C((None, 0.0, -np.pi / 2), gateduration_R, gate_R)
    gateduration_Syd, gate_Syd = C((None, np.pi / 2, -np.pi / 2), gateduration_R, gate_R)
    gateduration_Szd, gate_Szd = C((None, -np.pi / 2), gateduration_Rz, gate_Rz)
    gateduration_Sxx, gate_Sxx = C((None, None, 0.0, np.pi / 2), gateduration_MS, gate_MS)

    del C
