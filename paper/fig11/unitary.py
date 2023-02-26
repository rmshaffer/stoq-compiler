import copy
import numpy as np
import scipy.sparse
import scipy.stats

class Unitary:

    @classmethod
    def identity(this_class, dimension):
        operation_name = "I"
        return this_class(dimension, np.identity(dimension), operation_name)

    @classmethod
    def random(this_class, dimension):
        random_matrix = scipy.stats.unitary_group.rvs(dimension)
        return this_class(dimension, random_matrix)

    @classmethod
    def rx(this_class, theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rx"
        return this_class(dimension, np.array(
            [[np.cos(theta/2), -1j * np.sin(theta/2)],
             [-1j * np.sin(theta/2), np.cos(theta/2)]]), operation_name, parameter_dict)

    @classmethod
    def ry(this_class, theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Ry"
        return this_class(dimension, np.array(
            [[np.cos(theta/2), -np.sin(theta/2)],
             [np.sin(theta/2), np.cos(theta/2)]]
            ), operation_name, parameter_dict)

    @classmethod
    def rphi(this_class, theta, phi):
        dimension = 2
        parameter_dict = {"θ": (theta, True), "Φ": (phi, True)}
        operation_name = "R"
        return this_class(dimension, np.array(
            [[np.cos(theta/2), np.exp(-1j*(np.pi/2+phi))*np.sin(theta/2)],
             [np.exp(-1j*(np.pi/2-phi))*np.sin(theta/2), np.cos(theta/2)]]
            ), operation_name, parameter_dict)

    @classmethod
    def rz(this_class, theta):
        dimension = 2
        parameter_dict = {"θ": (theta, True)}
        operation_name = "Rz"
        return this_class(dimension, np.array(
            [[np.exp(-1j*theta/2), 0],
             [0, np.exp(1j*theta/2)]]
            ), operation_name, parameter_dict)

    @classmethod
    def h(this_class):
        h = this_class.rz(np.pi).left_multiply(this_class.ry(np.pi/2))
        return this_class(h.get_dimension(), h.get_matrix(), "H")

    @classmethod
    def t(this_class):
        t = this_class.rz(np.pi/4)
        return this_class(t.get_dimension(), t.get_matrix(), "T")

    @classmethod
    def sigmax(this_class):
        rx = this_class.rx(np.pi)
        return this_class(rx.get_dimension(), rx.get_matrix(), "X")

    @classmethod
    def sigmay(this_class):
        ry = this_class.ry(np.pi)
        return this_class(ry.get_dimension(), ry.get_matrix(), "Y")

    @classmethod
    def sigmaz(this_class):
        rz = this_class.rz(np.pi)
        return this_class(rz.get_dimension(), rz.get_matrix(), "Z")

    @classmethod
    def cnot(this_class):
        dimension = 4
        operation_name = "CNOT"
        return this_class(dimension, np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]), operation_name)

    @classmethod
    def ccnot(this_class):
        dimension = 8
        operation_name = "CCNOT"
        return this_class(dimension, np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0]]), operation_name)

    @classmethod
    def qecc_phase_flip(this_class):
        dimension = 8
        operation_name = "QECC"
        return this_class(dimension, (1/np.sqrt(8)) * np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, -1, -1, 1, -1, 1, 1, -1],
             [1, 1, -1, -1, 1, 1, -1, -1],
             [1, -1, 1, -1, -1, 1, -1, 1],
             [1, 1, 1, 1, -1, -1, -1, -1],
             [1, -1, -1, 1, 1, -1, -1, 1],
             [1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, -1, -1, -1, -1, 1, 1]]), operation_name)

    @classmethod
    def xx(this_class, theta=np.pi/4):
        dimension = 4
        parameter_dict = {"θ": (theta, True)}
        operation_name = "XX"
        return this_class(dimension, np.array(
            [[np.cos(theta), 0, 0, -1j*np.sin(theta)],
             [0, np.cos(theta), -1j*np.sin(theta), 0],
             [0, -1j*np.sin(theta), np.cos(theta), 0],
             [-1j*np.sin(theta), 0, 0, np.cos(theta)]]
            ), operation_name, parameter_dict)

    @classmethod
    def gms(this_class, num_qubits, theta=np.pi/4):
        dimension = 2**num_qubits
        parameter_dict = {"θ": (theta, True)}
        operation_name = f"GMS{num_qubits}"
        local_unitaries = [UnitarySequenceEntry(this_class.xx(theta), [i,j]).get_full_unitary(dimension) for i in range(num_qubits) for j in range(i+1, num_qubits)]
        global_unitary = this_class.identity(dimension)
        for u in local_unitaries:
            global_unitary = global_unitary.left_multiply(u)
        return this_class(dimension, global_unitary.get_matrix(), operation_name, parameter_dict)

    @classmethod
    def time_evolution(this_class, h_matrix, t, h_suffix=""):
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())

        dimension = h_matrix.shape[0]
        parameter_dict = {"t": (t, False)}
        operation_name = "H" + str(h_suffix)
        return this_class(dimension, scipy.linalg.expm(1j*h_matrix*t), operation_name, parameter_dict)

    def __init__(self, dimension, matrix=None, operation_name=None, parameter_dict=None, is_inverse=False, apply_to=[]):
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
        assert np.allclose(matrix @ matrix.T.conj(), np.identity(dimension)), matrix @ matrix.T.conj()

        # balance the global phase
        global_phase_factor = (1/np.linalg.det(matrix)) ** (1/dimension)
        matrix = global_phase_factor * matrix

        self.matrix = matrix

    def get_operation_name(self):
        return self.operation_name

    def get_jaqal(self):
        def as_decimal(value):
            return str(round(value, 7))

        parameters = ""
        if len(self.parameter_dict) > 0:
            parameters = " ".join(as_decimal(v[0]) for k,v in self.parameter_dict.items())

        qubits = ""
        apply_to_qubits = self.apply_to
        if len(apply_to_qubits) == 0:
            num_qubits = int(np.log2(self.get_dimension()))
            apply_to_qubits = list(range(num_qubits))

        qubits = " ".join("q[" + str(q) + "]" for q in apply_to_qubits)

        return self.operation_name + " " + qubits + " " + parameters

    def get_qasm(self):
        def as_pi_fraction(value):
            return "pi*" + str(round(value/np.pi, 7))

        def as_decimal(value):
            return str(round(value, 7))

        parameters = ""
        if len(self.parameter_dict) > 0:
            parameters = "(" + ",".join((as_pi_fraction(v[0]) if v[1] else as_decimal(v[0])) for k,v in self.parameter_dict.items()) + ")"

        qubits = ""
        apply_to_qubits = self.apply_to
        if len(apply_to_qubits) == 0:
            num_qubits = int(np.log2(self.get_dimension()))
            apply_to_qubits = list(range(num_qubits))

        qubits = "\t" + ",".join("q[" + str(q) + "]" for q in apply_to_qubits)

        return self.operation_name + parameters + qubits + ";"

    def get_display_name(self):
        def as_pi_fraction(value):
            return str(round(value/np.pi, 3)) + "π"

        def as_decimal(value):
            return str(round(value, 4))

        display_name = self.operation_name
        if len(self.apply_to) > 0:
            display_name += str(self.apply_to)
        if len(self.parameter_dict) > 0:
            display_name += "(" + ", ".join(k + "=" + (as_pi_fraction(v[0]) if v[1] else as_decimal(v[0])) for k,v in self.parameter_dict.items()) + ")"

        if self.is_inverse:
            display_name += '†'

        return display_name

    def get_dimension(self):
        return self.matrix.shape[0]

    def get_matrix(self):
        return self.matrix

    def get_parameter_value(self, key):
        if key in self.parameter_dict:
            return self.parameter_dict[key]
        return None

    def inverse(self):
        is_inverse = not self.is_inverse
        return Unitary(self.get_dimension(), self.get_matrix().T.conj(), self.get_operation_name(), self.parameter_dict, is_inverse=is_inverse, apply_to=self.apply_to)

    def tensor(self, u):
        assert isinstance(u, Unitary)

        new_dimension = u.get_dimension() * self.get_dimension()
        new_operation_name = self.get_operation_name() + "--" + u.get_operation_name()
        return Unitary(new_dimension, np.kron(self.get_matrix(), u.get_matrix()), new_operation_name, self.parameter_dict, apply_to=self.apply_to)

    def close_to(self, u, threshold=None):
        distance = self.distance_from(u)
        if threshold:
            max_distance = 1.0 - threshold
            return distance <= max_distance

        return np.isclose(distance, 0.0)

    def distance_from(self, u):
        if isinstance(u, Unitary):
            u = u.get_matrix()

        assert isinstance(u, np.ndarray)
        assert u.shape == (self.get_dimension(), self.get_dimension())

        self_dag_u = self.inverse().get_matrix() @ u
        trace = np.trace(self_dag_u)
        normalized_trace = np.linalg.norm(trace) / self.get_dimension()
        return 1.0 - normalized_trace

    def left_multiply(self, factor):
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(self.get_dimension(), factor.get_matrix() @ self.get_matrix())

    def right_multiply(self, factor):
        assert self.get_dimension() == factor.get_dimension()
        return Unitary(self.get_dimension(), self.get_matrix() @ factor.get_matrix())

class ParameterizedUnitary:

    @classmethod
    def rotation_xy(this_class):
        qubit_dimension = 2
        rotation_matrix = lambda theta, phi : Unitary.rphi(theta, phi).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True),
                      ParameterizedUnitaryParameter("phi", min_value, max_value, is_angle=True)]
        operation_name = "Rxy"
        return this_class(qubit_dimension, rotation_matrix, parameters, operation_name)

    @classmethod
    def rotation_xyz(this_class):
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
        return this_class(qubit_dimension, rotation_matrix, parameters, operation_name)

    @classmethod
    def xx(this_class):
        qubit_dimension = 2
        xx_matrix = lambda theta : Unitary.xx(theta).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True)]
        operation_name = "XX"
        return this_class(qubit_dimension ** 2, xx_matrix, parameters, operation_name)

    @classmethod
    def gms(this_class, num_qubits):
        qubit_dimension = 2
        gms_matrix = lambda theta : Unitary.gms(num_qubits, theta).get_matrix()
        min_value = 0
        max_value = 2*np.pi
        parameters = [ParameterizedUnitaryParameter("theta", min_value, max_value, is_angle=True)]
        operation_name = f"GMS{num_qubits}"
        return this_class(qubit_dimension ** num_qubits, gms_matrix, parameters, operation_name)

    @classmethod
    def time_evolution(this_class, h_matrix, t_min, t_max, h_suffix=""):
        assert isinstance(h_matrix, np.ndarray)
        assert np.allclose(h_matrix, h_matrix.T.conj())
        assert t_min <= t_max

        dimension = h_matrix.shape[0]
        u_matrix = lambda t : Unitary.time_evolution(h_matrix, t, h_suffix).get_matrix()
        min_value = t_min
        max_value = t_max
        parameters = [ParameterizedUnitaryParameter("t", min_value, max_value, is_angle=False)]
        operation_name = "H" + h_suffix
        return this_class(dimension, u_matrix, parameters, operation_name)

    def __init__(self, dimension, parameterized_matrix, parameters, operation_name):
        assert dimension > 0
        assert callable(parameterized_matrix)
        assert isinstance(parameters, list) or isinstance(parameters, np.ndarray)
        assert np.all([isinstance(p, ParameterizedUnitaryParameter) for p in parameters])
        assert isinstance(operation_name, str)

        self.dimension = dimension
        self.parameterized_matrix = parameterized_matrix
        self.parameters = parameters
        self.operation_name = operation_name

    def get_dimension(self):
        return self.dimension

    def get_parameters(self):
        return self.parameters

    def get_operation_name(self):
        return self.operation_name

    def as_unitary(self, parameter_values):
        assert isinstance(parameter_values, list) or isinstance(parameter_values, np.ndarray)
        assert np.all([self.get_parameters()[i].is_valid(p) for i,p in enumerate(parameter_values)])

        parameter_dict = {p.get_parameter_name(): (parameter_values[i], p.get_is_angle()) for i,p in enumerate(self.get_parameters())}
        return Unitary(self.get_dimension(), self.parameterized_matrix(*parameter_values), self.get_operation_name(), parameter_dict)

class ParameterizedUnitaryParameter:

    def __init__(self, parameter_name, min_value, max_value, is_angle):
        assert parameter_name
        assert max_value >= min_value

        self.parameter_name = parameter_name
        self.min_value = min_value
        self.max_value = max_value
        self.is_angle = is_angle

    def get_parameter_name(self):
        return self.parameter_name

    def get_is_angle(self):
        return self.is_angle

    def is_valid(self, parameter_value):
        return parameter_value >= self.min_value and parameter_value <= self.max_value

    def random_value(self):
        return np.random.random() * (self.max_value - self.min_value) + self.min_value

class UnitaryPrimitive:

    def __init__(self, unitary, allowed_apply_to=None):
        if allowed_apply_to is not None:
            allowed_apply_to = list(allowed_apply_to)
            assert np.all([len(apply_to) == len(set(apply_to)) for apply_to in allowed_apply_to])
            assert np.all([2**len(apply_to) == unitary.get_dimension() for apply_to in allowed_apply_to])
            assert np.all([np.min(apply_to) >= 0 for apply_to in allowed_apply_to])

        self.unitary = unitary
        self.allowed_apply_to = allowed_apply_to

    def get_unitary(self):
        return self.unitary

    def get_allowed_apply_to(self):
        return self.allowed_apply_to

class UnitarySequenceEntry:

    def __init__(self, unitary, apply_to):
        apply_to = list(apply_to)
        assert len(apply_to) == len(set(apply_to))
        assert 2**len(apply_to) == unitary.get_dimension()
        assert np.min(apply_to) >= 0

        self.unitary = unitary
        self.apply_to = apply_to
        self.full_unitary_by_dimension = {}

    def get_dimension(self):
        return self.unitary.get_dimension()

    def get_apply_to(self):
        return self.apply_to

    def get_permute_matrix(self, system_dimension):
        """
        Function returning permute matrix for given system dimension.
        Adapted from qutip.permute._permute()
        """
        assert system_dimension >= self.get_dimension()
        assert system_dimension >= 2**(np.max(self.get_apply_to())+1)

        def _select(sel, dims):
            """
            Private function finding selected components
            Copied from qutip.ptrace._select()
            """
            sel = np.asarray(sel)  # make sure sel is np.array
            dims = np.asarray(dims)  # make sure dims is np.array
            rlst = dims.take(sel)
            rprod = np.prod(rlst)
            ilist = np.ones((rprod, len(dims)), dtype=int)
            counter = np.arange(rprod)
            for k in range(len(sel)):
                ilist[:, sel[k]] = np.remainder(
                    np.fix(counter / np.prod(dims[sel[k + 1:]])), dims[sel[k]]) + 1
            return ilist

        def _perm_inds(dims, order):
            """
            Private function giving permuted indices for permute function.
            Copied from qutip.permute._perm_inds()
            """
            dims = np.asarray(dims)
            order = np.asarray(order)
            assert np.all(np.sort(order) == np.arange(len(dims))), 'Requested permutation does not match tensor structure.'
            sel = _select(order, dims)
            irev = np.fliplr(sel) - 1
            fact = np.append(np.array([1]), np.cumprod(np.flipud(dims)[:-1]))
            fact = fact.reshape(len(fact), 1)
            perm_inds = np.dot(irev, fact)
            return perm_inds

        num_system_qubits = int(np.log2(system_dimension))
        permute_order = [None] * num_system_qubits
        next_index = 0
        for apply in self.get_apply_to():
            permute_order[apply] = next_index
            next_index += 1
        for i in range(len(permute_order)):
            if permute_order[i] is None:
                permute_order[i] = next_index
                next_index += 1

        permute_indices = _perm_inds([2] * num_system_qubits, permute_order)
        data = np.ones(system_dimension, dtype=int)
        rows = np.arange(system_dimension, dtype=int)
        permute_matrix = scipy.sparse.coo_matrix((data, (rows, permute_indices.T[0])), shape=(system_dimension, system_dimension), dtype=int).tocsr()

        return permute_matrix

    def get_full_unitary(self, system_dimension):
        assert system_dimension >= self.get_dimension()

        if system_dimension not in self.full_unitary_by_dimension:
            expansion = Unitary.identity(int(system_dimension / self.get_dimension()))
            expanded_unitary = self.unitary.tensor(expansion)
            assert expanded_unitary.get_dimension() == system_dimension

            permute_matrix = self.get_permute_matrix(system_dimension)
            full_matrix = permute_matrix * expanded_unitary.get_matrix() * permute_matrix.T

            operation_name = self.unitary.get_operation_name()
            self.full_unitary_by_dimension[system_dimension] = Unitary(
                system_dimension,
                full_matrix,
                operation_name,
                self.unitary.parameter_dict,
                apply_to=self.get_apply_to())

        return self.full_unitary_by_dimension[system_dimension]

class UnitarySequence:

    @classmethod
    def combine(this_class, *sequences):
        assert len(sequences) > 0
        assert np.all([isinstance(sequence, this_class) for sequence in sequences])

        dimension = sequences[0].get_dimension()
        assert np.all([sequence.get_dimension() == dimension for sequence in sequences])

        new_sequence_entries = []
        new_product = Unitary.identity(dimension)
        for sequence in sequences:
            new_sequence_entries.extend(sequence.get_sequence_entries())
            new_product = new_product.left_multiply(sequence.product())

        new_sequence = this_class(dimension, new_sequence_entries)
        new_sequence.sequence_product = new_product
        return new_sequence

    def __init__(self, dimension, sequence_entries=[]):
        self.dimension = dimension

        assert isinstance(sequence_entries, list) or isinstance(sequence_entries, np.ndarray)
        [self.assert_is_entry_valid(entry) for entry in sequence_entries]

        self.sequence_entries = copy.deepcopy(sequence_entries)
        self.sequence_product = None

        self.previous_entries = None
        self.previous_sequence_product = None

    def assert_is_entry_valid(self, sequence_entry):
        assert isinstance(sequence_entry, UnitarySequenceEntry), type(sequence_entry)
        assert sequence_entry.get_dimension() <= self.dimension
        assert 2**(np.max(sequence_entry.get_apply_to())+1) <= self.dimension

    def get_dimension(self):
        return self.dimension

    def get_length(self):
        return len(self.sequence_entries)

    def get_sequence_entries(self):
        return copy.deepcopy(self.sequence_entries)

    def save_undo_state(self):
        self.previous_entries = copy.deepcopy(self.sequence_entries)
        if self.sequence_product:
            self.previous_sequence_product = Unitary(self.sequence_product.get_dimension(), self.sequence_product.get_matrix())

    def append_first(self, sequence_entry, save_undo=True):
        self.assert_is_entry_valid(sequence_entry)

        if save_undo:
            self.save_undo_state()
        self.sequence_entries.insert(0, sequence_entry)
        if self.sequence_product:
            u_appended = sequence_entry.get_full_unitary(self.dimension)
            self.sequence_product = self.sequence_product.right_multiply(u_appended)

    def append_last(self, sequence_entry, save_undo=True):
        self.assert_is_entry_valid(sequence_entry)

        if save_undo:
            self.save_undo_state()
        self.sequence_entries.append(sequence_entry)
        if self.sequence_product:
            u_appended = sequence_entry.get_full_unitary(self.dimension)
            self.sequence_product = self.sequence_product.left_multiply(u_appended)

    def remove_first(self, save_undo=True):
        if self.get_length() > 0:
            if save_undo:
                self.save_undo_state()
            entry_removed = self.sequence_entries.pop(0)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.right_multiply(u_removed.inverse())

    def remove_last(self, save_undo=True):
        if self.get_length() > 0:
            if save_undo:
                self.save_undo_state()
            entry_removed = self.sequence_entries.pop(-1)
            if self.sequence_product:
                u_removed = entry_removed.get_full_unitary(self.dimension)
                self.sequence_product = self.sequence_product.left_multiply(u_removed.inverse())

    def undo(self):
        assert not self.previous_entries is None, "can't undo"

        self.sequence_entries = self.previous_entries
        self.sequence_product = self.previous_sequence_product

        self.previous_entries = None
        self.previous_sequence_product = None

    def product(self):
        if self.sequence_product is None:
            sequence_product = Unitary(self.dimension)
            for entry in self.sequence_entries:
                assert isinstance(entry, UnitarySequenceEntry)
                u_entry = entry.get_full_unitary(self.dimension)
                sequence_product = sequence_product.left_multiply(u_entry)
            self.sequence_product = sequence_product

        return self.sequence_product

    def inverse(self):
        inverse_sequence = UnitarySequence(self.get_dimension(), self.get_sequence_entries())
        inverse_sequence.sequence_entries.reverse()
        for entry in inverse_sequence.sequence_entries:
            entry.unitary = entry.unitary.inverse()
            entry.full_unitary_by_dimension = {}
        inverse_sequence.sequence_product = self.product().inverse()
        return inverse_sequence

    def get_jaqal(self):
        return "// JAQAL generated from UnitarySequence.get_jaqal()\n" + \
            "\n".join([entry.get_full_unitary(self.get_dimension()).get_jaqal() for entry in self.sequence_entries])

    def get_qasm(self):
        return "# QASM generated from UnitarySequence.get_qasm()\n" + \
            "\n".join([entry.get_full_unitary(self.get_dimension()).get_qasm() for entry in self.sequence_entries])

    def get_display_output(self):
        return "# Display output generated from UnitarySequence.get_display_output()\n" + \
            "\n".join([entry.get_full_unitary(self.get_dimension()).get_display_name() for entry in self.sequence_entries])
