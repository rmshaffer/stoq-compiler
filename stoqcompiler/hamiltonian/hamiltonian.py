import copy
import numpy as np
import random
import itertools

from stoqcompiler.compiler import Compiler
from stoqcompiler.unitary import (
    Unitary,
    UnitaryPrimitive,
    UnitarySequence,
    UnitarySequenceEntry,
    UnitaryDefinitions,
    ParameterizedUnitary,
    ParameterizedUnitaryDefinitions)

from .hamiltonian_term import HamiltonianTerm

class Hamiltonian:

    def __init__(self, terms):
        assert isinstance(terms, list)
        assert len(terms) > 0
        assert np.all([isinstance(term, HamiltonianTerm) for term in terms])

        self.dimension = terms[0].get_dimension()
        assert np.all([term.get_dimension() == self.dimension for term in terms])

        self.terms = copy.deepcopy(terms)

    def get_dimension(self):
        return self.dimension

    def get_qubit_count(self):
        return int(np.log2(self.get_dimension()))

    def get_time_evolution_operator(self, time):
        h = np.sum([term.get_matrix() for term in self.terms], axis=0)
        return UnitaryDefinitions.time_evolution(h, time)

    def get_ideal_sequence(self, time, num_steps):
        sequence_entries = []
        time_per_step = time / num_steps
        u = self.get_time_evolution_operator(time_per_step)
        apply_to = list(range(self.get_qubit_count()))
        for _ in range(num_steps):
            entry = UnitarySequenceEntry(u, apply_to)
            sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def get_trotter_sequence(self, time, num_trotter_steps, randomize=False):
        sequence_entries = []
        time_per_step = time / num_trotter_steps
        apply_to = list(range(self.get_qubit_count()))
        term_indices = list(range(len(self.terms)))
        for _ in range(num_trotter_steps):
            if randomize:
                random.shuffle(term_indices)
            for term_index in term_indices:
                term = self.terms[term_index]
                u = UnitaryDefinitions.time_evolution(term.get_matrix(), time_per_step, term_index)
                entry = UnitarySequenceEntry(u, apply_to)
                sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def get_qdrift_sequence(self, time, num_repetitions):
        # QDRIFT as per Campbell, PRL 123, 070503 (2019)
        sequence_entries = []
        coefficients = [term.get_coefficient() for term in self.terms]
        sum_coefficients = np.sum(coefficients)
        prob_coefficients = coefficients / sum_coefficients
        time_per_step = sum_coefficients * time / num_repetitions
        apply_to = list(range(self.get_qubit_count()))
        for _ in range(num_repetitions):
            term = np.random.choice(self.terms, p=prob_coefficients)
            display_suffix = str(self.terms.index(term)) + ' λ/N'
            u = UnitaryDefinitions.time_evolution(term.get_normalized_matrix(), time_per_step, display_suffix)
            entry = UnitarySequenceEntry(u, apply_to)
            sequence_entries.append(entry)

        return UnitarySequence(self.get_dimension(), sequence_entries)

    def compile_qmcmc_sequence(self, time, max_t_step, threshold, allow_simultaneous_terms=False):
        # QMCMC as per Shaffer et al., arXiv:2003.04500 (2020)
        target_unitary = self.get_time_evolution_operator(time)
        return self._compile_qmcmc_sequence_for_target_unitary(target_unitary, max_t_step, threshold, allow_simultaneous_terms)

    def _compile_qmcmc_sequence_for_target_unitary(self, target_unitary, max_t_step, threshold, allow_simultaneous_terms):
        unitary_primitives = self._get_unitary_primitives(max_t_step, allow_simultaneous_terms)

        compiler = Compiler(self.get_dimension())
        compiler.set_unitary_primitives(unitary_primitives)
        result = compiler.compile(target_unitary, threshold, max_step_count=10000)

        return result

    def _get_unitary_primitives(self, max_t_step, allow_simultaneous_terms):
        unitary_primitives = []
        apply_to = list(range(self.get_qubit_count()))
        for indices, term_subset in self._get_term_subsets(allow_simultaneous_terms):
            term_subset_sum = np.sum([term.get_matrix() for term in term_subset], axis=0)
            h_suffix = "+".join([str(i) for i in indices])
            primitive = UnitaryPrimitive(ParameterizedUnitaryDefinitions.time_evolution(term_subset_sum, -max_t_step, max_t_step, h_suffix), [apply_to])
            unitary_primitives.append(primitive)

        return unitary_primitives

    def _get_term_subsets(self, allow_simultaneous_terms):
        if allow_simultaneous_terms:
            for subset_size in range(1, len(self.terms) + 1):
                for indices in itertools.combinations(list(range(len(self.terms))), subset_size):
                    yield indices, [self.terms[i] for i in indices]
        else:
            for i, term in enumerate(self.terms):
                yield [i], [term]

    def compile_rav_sequence(self, time, max_t_step, threshold, allow_simultaneous_terms=False):
        # Randomized analog verification (RAV) as per Shaffer et al., arXiv:2003.04500 (2020)

        # Generate a random sequence, mostly forward in time
        forward_probability = 0.8
        unitary_primitives = self._get_unitary_primitives(max_t_step, allow_simultaneous_terms)
        apply_to = list(range(self.get_qubit_count()))
        random_sequence = UnitarySequence(self.get_dimension())
        total_time = 0.0
        while total_time < time:
            t_step = (max_t_step * np.random.random_sample()) * (1 if np.random.random_sample() < forward_probability else -1)
            u_step = np.random.choice(unitary_primitives).get_unitary().as_unitary([t_step])
            random_sequence.append_last(UnitarySequenceEntry(u_step, apply_to))
            total_time += np.abs(t_step)

        # Calculate the product of this sequence and invert it
        target_unitary = random_sequence.product().inverse()

        # Call _compile_qmcmc_sequence_from_unitary to compile a new sequence implementing the inverse
        result = self._compile_qmcmc_sequence_for_target_unitary(target_unitary, max_t_step, threshold, allow_simultaneous_terms)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(random_sequence, result.compiled_sequence)
        return result
