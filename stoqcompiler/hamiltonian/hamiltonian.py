'''
Defines the Hamiltonian class.
'''
import copy
import numpy as np
import random
import itertools
from typing import Iterable, List, Tuple

from stoqcompiler.compiler import Compiler, CompilerResult
from stoqcompiler.unitary import (
    Unitary,
    UnitaryPrimitive,
    UnitarySequence,
    UnitarySequenceEntry,
    UnitaryDefinitions,
    ParameterizedUnitaryDefinitions)

from .hamiltonian_term import HamiltonianTerm


class Hamiltonian:
    '''
    Defines a Hamiltonian from a set of HamiltonianTerm objects.

    :param terms: The list of Hamiltonian terms that make up this
        Hamiltonian. The list must be non-empty and all terms must
        have the same dimension.
    :type terms: List[HamiltonianTerm]
    '''
    def __init__(
        self,
        terms: List[HamiltonianTerm]
    ):
        '''
        Creates a Hamiltonian object.
        '''
        assert isinstance(terms, list)
        assert len(terms) > 0
        assert np.all([
            isinstance(term, HamiltonianTerm) for term in terms])

        self.dimension = terms[0].get_dimension()
        assert np.all([
            term.get_dimension() == self.dimension for term in terms])

        self.terms = copy.deepcopy(terms)

    def get_dimension(self) -> int:
        '''
        Gets the dimension of the state space on which
        this Hamiltonian acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.dimension

    def get_qubit_count(self) -> int:
        '''
        Gets the number of qubits on which this Hamiltonian
        acts. This is calculated as the base-2 log of the
        dimension.

        :return: The number of qubits.
        :rtype: int
        '''
        return int(np.log2(self.get_dimension()))

    def get_time_evolution_operator(
        self,
        time: float
    ) -> Unitary:
        '''
        Creates a unitary operator representing the time evolution of
        a system under this Hamiltonian for the specified time.

        :param time: Evolution time of the system.
        :type time: float
        :return: The unitary time-evolution operator for the specified time.
        :rtype: Unitary
        '''
        h = np.sum([term.get_matrix() for term in self.terms], axis=0)
        return UnitaryDefinitions.time_evolution(h, time)

    def get_ideal_sequence(
        self,
        time: float,
        num_steps: int
    ) -> UnitarySequence:
        '''
        Returns a sequence of identical unitaries, where each unitary is the
        time-evolution operator under this Hamiltonian for time / num_steps,
        and the length of the sequence is num_steps.

        :param time: The total time to evolve the system.
        :type time: float
        :param num_steps: The number of steps in which to break up the
            time evolution of the system.
        :type num_steps: int
        :return: A sequence of num_steps identical unitaries implementing
            the time evolution of the system.
        :rtype: UnitarySequence
        '''
        sequence_entries = []
        time_per_step = time / num_steps
        u = self.get_time_evolution_operator(time_per_step)
        apply_to = list(range(self.get_qubit_count()))
        for _ in range(num_steps):
            entry = UnitarySequenceEntry(u, apply_to)
            sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def get_trotter_sequence(
        self,
        time: float,
        num_trotter_steps: int,
        randomize: bool = False
    ) -> UnitarySequence:
        '''
        Returns a sequence of unitaries using a Suzuki-Trotter decomposition
        of the time-evolution under this Hamiltonian. The sequence
        approximately implements the ideal time evolution of the system.

        :param time: The total time to evolve the system.
        :type time: float
        :param num_trotter_steps: The number of Trotter steps to use.
        :type num_trotter_steps: int
        :param randomize: Whether to randomize the order of Hamiltonian terms
            in each step of the Suzuki-Trotter decomposition, defaults
            to False.
        :type randomize: bool, optional
        :return: A sequence of unitaries implementing the Suzuki-Trotter
            decomposition of the time evolution of the system.
        :rtype: UnitarySequence
        '''
        sequence_entries = []
        time_per_step = time / num_trotter_steps
        apply_to = list(range(self.get_qubit_count()))
        term_indices = list(range(len(self.terms)))
        for _ in range(num_trotter_steps):
            if randomize:
                random.shuffle(term_indices)
            for term_index in term_indices:
                term = self.terms[term_index]
                u = UnitaryDefinitions.time_evolution(
                    term.get_matrix(), time_per_step, term_index)
                entry = UnitarySequenceEntry(u, apply_to)
                sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def get_qdrift_sequence(
        self,
        time: float,
        num_repetitions: int
    ) -> UnitarySequence:
        '''
        Returns a sequence of unitaries using a QDRIFT decomposition
        of the time-evolution under this Hamiltonian, as per
        Campbell, PRL 123, 070503 (2019). The sequence approximately
        implements the ideal time evolution of the system.

        :param time: The total time to evolve the system.
        :type time: float
        :param num_repetitions: The number of QDRIFT repetitions to use.
        :type num_repetitions: int
        :return: A sequence of unitaries implementing the QDRIFT
            decomposition of the time evolution of the system.
        :rtype: UnitarySequence
        '''
        sequence_entries = []
        coefficients = [term.get_coefficient() for term in self.terms]
        sum_coefficients = np.sum(coefficients)
        prob_coefficients = coefficients / sum_coefficients
        time_per_step = sum_coefficients * time / num_repetitions
        apply_to = list(range(self.get_qubit_count()))
        for _ in range(num_repetitions):
            term = np.random.choice(self.terms, p=prob_coefficients)
            display_suffix = str(self.terms.index(term)) + ' λ/N'
            u = UnitaryDefinitions.time_evolution(
                term.get_normalized_matrix(),
                time_per_step, display_suffix)
            entry = UnitarySequenceEntry(u, apply_to)
            sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def compile_stoq_sequence(
        self,
        time: float,
        max_t_step: float,
        threshold: float,
        allow_simultaneous_terms: bool = False
    ) -> CompilerResult:
        '''
        Returns a sequence of unitaries using a STOQ compilation
        of the time-evolution under this Hamiltonian. The sequence
        approximately implements the ideal time evolution of the system.

        :param time: The total time to evolve the system.
        :type time: float
        :param max_t_step: The maximum time to use for a single Hamiltonian
            term at each step of the sequence.
        :type max_t_step: float
        :param threshold: The overlap with the target unitary at which to
            stop compilation, defaults to None. A value of 1.0 implies an exact
            compilation.
        :type threshold: float
        :param allow_simultaneous_terms: Whether to allow multiple
            Hamiltonian terms to be executed simultaneously in the resulting
            sequence, defaults to False.
        :type allow_simultaneous_terms: bool, optional
        :return: A sequence of unitaries implementing a STOQ
            compilation of the time evolution of the system.
        :rtype: CompilerResult
        '''
        target_unitary = self.get_time_evolution_operator(time)
        return self._compile_stoq_sequence_for_target_unitary(
            target_unitary, max_t_step, threshold, allow_simultaneous_terms)

    def _compile_stoq_sequence_for_target_unitary(
        self,
        target_unitary: Unitary,
        max_t_step: float,
        threshold: float,
        allow_simultaneous_terms: bool
    ) -> CompilerResult:
        '''
        Internal implementation of STOQ time-evolution compilation.
        See Hamiltonian.compile_stoq_sequence() for full details.
        '''
        unitary_primitives = self._get_unitary_primitives(
            max_t_step, allow_simultaneous_terms)

        compiler = Compiler(self.get_dimension(), unitary_primitives)
        result = compiler.compile(
            target_unitary, threshold, max_step_count=10000)

        return result

    def _get_unitary_primitives(
        self,
        max_t_step: float,
        allow_simultaneous_terms: bool
    ) -> List[UnitaryPrimitive]:
        '''
        Gets the list of unitary primitives for STOQ compilation
        based on the individual Hamiltonian terms.
        '''
        unitary_primitives = []
        apply_to = list(range(self.get_qubit_count()))
        for indices, term_subset in self._get_term_subsets(
                allow_simultaneous_terms):
            term_subset_sum = np.sum([
                term.get_matrix() for term in term_subset], axis=0)
            h_suffix = "+".join([str(i) for i in indices])
            primitive = UnitaryPrimitive(
                ParameterizedUnitaryDefinitions.time_evolution(
                    term_subset_sum, -max_t_step, max_t_step, h_suffix
                ), [apply_to])
            unitary_primitives.append(primitive)

        return unitary_primitives

    def _get_term_subsets(
        self,
        allow_simultaneous_terms: bool
    ) -> Iterable[Tuple[List[int], List[HamiltonianTerm]]]:
        '''
        Gets the possible subsets of Hamiltonian terms to be
        used at each step of the STOQ compilation, depending
        on whether simultaneous terms are allowed.
        '''
        if allow_simultaneous_terms:
            for subset_size in range(1, len(self.terms) + 1):
                for indices in itertools.combinations(
                        list(range(len(self.terms))), subset_size):
                    yield indices, [self.terms[i] for i in indices]
        else:
            for i, term in enumerate(self.terms):
                yield [i], [term]

    def compile_rav_sequence(
        self,
        time: float,
        max_t_step: float,
        threshold: float,
        allow_simultaneous_terms: bool = False
    ) -> CompilerResult:
        '''
        Returns a randomized analog verification (RAV) sequence as
        per Shaffer et al., arXiv:2003.04500 (2020). The sequence of
        unitaries is built from terms of this Hamiltonian by first
        generating a random sequence and then using STOQ to compile
        the inverse such that the full sequence approximately
        implements the identity operation.

        :param time: The total time to evolve the system in the
            initial randomly-generated sequence.
        :type time: float
        :param max_t_step: The maximum time to use for a single Hamiltonian
            term at each step of the sequence.
        :type max_t_step: float
        :param threshold: The overlap with the target unitary at which to
            stop the STOQ compilation, defaults to None. A value of 1.0
            implies an exact compilation.
        :type threshold: float
        :param allow_simultaneous_terms: Whether to allow multiple
            Hamiltonian terms to be executed simultaneously in the resulting
            sequence, defaults to False.
        :type allow_simultaneous_terms: bool, optional
        :return: A sequence of unitaries implementing RAV.
        :rtype: CompilerResult
        '''
        # Generate a random sequence, mostly forward in time
        forward_probability = 0.8
        unitary_primitives = self._get_unitary_primitives(
            max_t_step, allow_simultaneous_terms)
        apply_to = list(range(self.get_qubit_count()))
        random_sequence = UnitarySequence(self.get_dimension())
        total_time = 0.0
        while total_time < time:
            t_step = (max_t_step * np.random.random_sample()) * (
                1 if np.random.random_sample() < forward_probability else -1)
            u_step = np.random.choice(
                unitary_primitives).get_unitary().as_unitary([t_step])
            random_sequence.append_last(UnitarySequenceEntry(u_step, apply_to))
            total_time += np.abs(t_step)

        # Calculate the product of this sequence and invert it
        target_unitary = random_sequence.product().inverse()

        # Call _compile_stoq_sequence_from_unitary to compile a new sequence
        # implementing the inverse
        result = self._compile_stoq_sequence_for_target_unitary(
            target_unitary, max_t_step, threshold, allow_simultaneous_terms)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(
            random_sequence, result.compiled_sequence)
        return result
