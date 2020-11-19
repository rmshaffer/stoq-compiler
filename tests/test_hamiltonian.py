'''
Tests for stoqcompiler.hamiltonian modules.
'''
import pytest
import numpy as np

from stoqcompiler.hamiltonian import Hamiltonian, HamiltonianTerm
from stoqcompiler.compiler import CompilerResult
from stoqcompiler.unitary import Unitary, UnitarySequence

qubit_dimension = 2


class TestHamiltonian:
    def test_no_terms(self) -> None:
        terms = None
        with pytest.raises(Exception):
            Hamiltonian(terms)

    def test_terms_mismatched_dimension(self) -> None:
        term1 = HamiltonianTerm(np.array([
            [3, 2 + 1j],
            [2 - 1j, 3]]))
        term2 = HamiltonianTerm(np.array([
            [3, 2 + 1j, 0, 0],
            [2 - 1j, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3]]))
        with pytest.raises(Exception):
            Hamiltonian([term1, term2])

    def test_simple_terms(self) -> None:
        term = HamiltonianTerm(np.array([[3, 2 + 1j], [2 - 1j, 3]]))
        identity = Unitary.identity(term.get_dimension())

        h_2terms = Hamiltonian([term, term])
        h_3terms = Hamiltonian([term, term, term])
        assert term.get_dimension() == h_2terms.get_dimension()
        assert term.get_dimension() == h_3terms.get_dimension()

        u = h_2terms.get_time_evolution_operator(0)
        assert isinstance(u, Unitary)
        assert u.close_to(identity)

        time = 1.234
        u_2terms = h_2terms.get_time_evolution_operator(3 * time)
        u_3terms = h_3terms.get_time_evolution_operator(2 * time)
        assert not u_2terms.close_to(identity)
        assert u_2terms.close_to(u_3terms)

    def test_ideal_sequence(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        x_term = HamiltonianTerm(2 * sigmax)
        y_term = HamiltonianTerm(-3 * sigmay)
        terms = [x_term, y_term]
        h = Hamiltonian(terms)

        time = 1.234
        u = h.get_time_evolution_operator(time)

        num_steps = 1000
        ideal_sequence = h.get_ideal_sequence(time, num_steps)
        assert isinstance(ideal_sequence, UnitarySequence)
        assert ideal_sequence.get_length() == num_steps

        assert u.close_to(ideal_sequence.product()), \
            u.distance_from(ideal_sequence.product())

    def test_trotterization(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        x_term = HamiltonianTerm(2 * sigmax)
        y_term = HamiltonianTerm(-3 * sigmay)
        terms = [x_term, y_term]
        h = Hamiltonian(terms)

        time = 1.234
        u = h.get_time_evolution_operator(time)

        num_trotter_steps = 20
        trotter_sequence = h.get_trotter_sequence(time, num_trotter_steps)
        assert isinstance(trotter_sequence, UnitarySequence)
        assert trotter_sequence.get_length() == num_trotter_steps * len(terms)

        randomized_trotter_sequence = h.get_trotter_sequence(
            time, num_trotter_steps, randomize=True)
        assert isinstance(randomized_trotter_sequence, UnitarySequence)
        assert (randomized_trotter_sequence.get_length()
                == num_trotter_steps * len(terms))

        assert u.close_to(trotter_sequence.product(), 0.95), \
            u.distance_from(trotter_sequence.product())
        assert u.close_to(randomized_trotter_sequence.product(), 0.95), \
            u.distance_from(randomized_trotter_sequence.product())
        assert not trotter_sequence.product().close_to(
            randomized_trotter_sequence.product())

    def test_qdrift(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        x_term = HamiltonianTerm(10 * sigmax)
        y_term = HamiltonianTerm(-1 * sigmay)
        terms = [x_term, y_term]
        h = Hamiltonian(terms)

        time = 0.543
        u = h.get_time_evolution_operator(time)

        num_repetitions = 1000
        qdrift_sequence = h.get_qdrift_sequence(time, num_repetitions)
        assert isinstance(qdrift_sequence, UnitarySequence)
        assert qdrift_sequence.get_length() == num_repetitions

        assert u.close_to(qdrift_sequence.product(), 0.95), \
            u.distance_from(qdrift_sequence.product())

    def test_stoq(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        x_term = HamiltonianTerm(2 * sigmax)
        y_term = HamiltonianTerm(-3 * sigmay)
        terms = [x_term, y_term]
        h = Hamiltonian(terms)

        time = 0.543
        u = h.get_time_evolution_operator(time)
        max_t_step = time / 10

        threshold = 0.9
        stoq_compiler_result = h.compile_stoq_sequence(
            time, max_t_step, threshold, allow_simultaneous_terms=False)
        assert isinstance(stoq_compiler_result, CompilerResult)
        assert u.close_to(
            stoq_compiler_result.compiled_sequence.product(), threshold), \
            u.distance_from(stoq_compiler_result.compiled_sequence.product())

        threshold = 0.9
        stoq_compiler_result = h.compile_stoq_sequence(
            time, max_t_step, threshold, allow_simultaneous_terms=True)
        assert isinstance(stoq_compiler_result, CompilerResult)
        assert u.close_to(
            stoq_compiler_result.compiled_sequence.product(), threshold), \
            u.distance_from(stoq_compiler_result.compiled_sequence.product())
        assert stoq_compiler_result.compiled_sequence.get_qasm()

    def test_rav(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        x_term = HamiltonianTerm(2 * sigmax)
        y_term = HamiltonianTerm(-3 * sigmay)
        terms = [x_term, y_term]
        h = Hamiltonian(terms)

        time = 0.543
        max_t_step = time / 10
        threshold = 0.9

        rav_result = h.compile_rav_sequence(
            time, max_t_step, threshold, allow_simultaneous_terms=True)
        assert isinstance(rav_result, CompilerResult)

        product = rav_result.compiled_sequence.product()
        assert product.close_to(
            Unitary.identity(h.get_dimension()), threshold), \
            product.distance_from(Unitary.identity(h.get_dimension()))
        assert rav_result.compiled_sequence.get_qasm()

    def test_two_qubits(self) -> None:
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1j], [1j, 0]])
        xx = np.kron(sigmax, sigmax)
        y1 = np.kron(sigmay, np.identity(qubit_dimension))
        y2 = np.kron(np.identity(qubit_dimension), sigmay)
        terms = [
            HamiltonianTerm(2 * xx),
            HamiltonianTerm(1.5 * y1),
            HamiltonianTerm(1.1 * y2)]
        h = Hamiltonian(terms)

        u = h.get_time_evolution_operator(0)
        assert isinstance(u, Unitary)
        assert u.get_dimension() == h.get_dimension()
        assert u.close_to(Unitary.identity(h.get_dimension()))

        time = 1.234
        u = h.get_time_evolution_operator(time)

        num_trotter_steps = 20
        randomized_trotter_sequence = h.get_trotter_sequence(
            time, num_trotter_steps, randomize=True)
        assert isinstance(randomized_trotter_sequence, UnitarySequence)

        num_repetitions = 1000
        qdrift_sequence = h.get_qdrift_sequence(time, num_repetitions)
        assert isinstance(qdrift_sequence, UnitarySequence)
        assert qdrift_sequence.get_length() == num_repetitions

        # should be close
        assert u.close_to(randomized_trotter_sequence.product(), 0.95), \
            u.distance_from(randomized_trotter_sequence.product())
        assert u.close_to(qdrift_sequence.product(), 0.95), \
            u.distance_from(qdrift_sequence.product())

        # but should not be exactly the same
        assert not u.close_to(randomized_trotter_sequence.product())
        assert not u.close_to(qdrift_sequence.product())
        assert not randomized_trotter_sequence.product().close_to(
            qdrift_sequence.product())


class TestHamiltonianTerm:

    def test_no_matrix(self) -> None:
        matrix = None
        with pytest.raises(Exception):
            HamiltonianTerm(matrix)

    def test_non_hermitian_matrix(self) -> None:
        matrix = np.array([[1, 0], [1, 1]])
        with pytest.raises(Exception):
            HamiltonianTerm(matrix)

    def test_simple_hermitian_matrix(self) -> None:
        matrix = np.array([[3, 2 + 1j], [2 - 1j, 3]])
        term = HamiltonianTerm(matrix)
        assert term.get_dimension() == matrix.shape[0]

        assert np.allclose(matrix, term.get_matrix())

        coefficient = term.get_coefficient()
        normalized_matrix = term.get_normalized_matrix()
        assert np.isreal(coefficient)
        assert coefficient >= 0.0
        assert np.allclose(matrix, coefficient * normalized_matrix)
        _, s, _ = np.linalg.svd(normalized_matrix)
        assert np.isclose(np.max(s), 1.0)
