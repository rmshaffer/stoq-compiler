'''
Defines the Compiler class implementing STOQ compilation.
'''
import copy
import numpy as np
import time
from typing import Dict, List, Optional

from stoqcompiler.unitary import (
    Unitary,
    UnitaryPrimitive,
    UnitarySequence,
    UnitarySequenceEntry,
    ParameterizedUnitary)

from .compiler_action import CompilerAction
from .compiler_result import CompilerResult


class Compiler:
    '''
    Implements the STOQ compilation technique.
    '''

    def __init__(
        self,
        dimension: int,
        append_probability: float = 0.5,
        annealing_rate: float = 0.1
    ):
        '''
        Creates a Compiler object.

        :param dimension: The dimension of the state space. For an n-qubit
            system, dimension should be set to 2**n.
        :type dimension: int
        :param append_probability: Probability of appending a new gate at each
            step in the compilation, defaults to 0.5.
        :type append_probability: float, optional
        :param annealing_rate: Rate at which MCMC annealing occurs during the
            compilation process, defaults to 0.1.
        :type annealing_rate: float, optional
        '''
        assert dimension > 0
        self.dimension = dimension
        self.unitary_primitives = []
        self.unitary_primitive_probabilities = None
        self.max_beta = (2**dimension) ** 2
        self.beta = 0.0
        self.annealing_rate = annealing_rate
        self.append_probability = append_probability

    def set_unitary_primitives(
        self,
        unitary_primitives: List[UnitaryPrimitive],
        unitary_primitive_probabilities: Optional[List[float]] = None
    ) -> None:
        '''
        Sets the unitary primitives and associated probabilities for
        use in the STOQ compilation process.

        :param unitary_primitives: The unitary primitives to be used for
            the compilation.
        :type unitary_primitives: List[UnitaryPrimitive]
        :param unitary_primitive_probabilities: The probability for STOQ to
            choose each of the primitives specified in unitary_primitives when
            proposing new gates at each step of the compilation process,
            defaults to None. If not specified, each unitary primitive is
            chosen with uniform probability.
        :type unitary_primitive_probabilities: Optional[List[float]], optional
        '''
        assert (isinstance(unitary_primitives, list)
                or isinstance(unitary_primitives, np.ndarray))
        assert np.all([
            isinstance(primitive, UnitaryPrimitive)
            for primitive in unitary_primitives])
        assert np.all([
            primitive.get_unitary().get_dimension() <= self.dimension
            for primitive in unitary_primitives])

        self.unitary_primitives = copy.deepcopy(unitary_primitives)
        self.unitary_primitive_probabilities = copy.deepcopy(
            unitary_primitive_probabilities)

    def compile(
        self,
        target_unitary: Unitary,
        threshold: Optional[float] = None,
        max_step_count: int = np.iinfo(np.int32).max
    ) -> CompilerResult:
        '''
        Compiles a unitary sequence approximately implementing the specified
        unitary using the STOQ compilation technique.

        :param target_unitary: The target unitary to be compiled.
        :type target_unitary: Unitary
        :param threshold: The overlap with the target unitary at which to
            stop compilation, defaults to None. A value of 1.0 implies an exact
            compilation. If None, a threshold of 1.0 is used.
        :type threshold: Optional[float], optional
        :param max_step_count: Maximum number of steps to perform while
            attempting to perform the approximate compilation, defaults to
            np.iinfo(np.int32).max. Compilation will terminate after this
            number of steps regardless of whether the threshold has
            been reached.
        :type max_step_count: int, optional
        :return: The CompilerResult object containing the output of
            the compilation.
        :rtype: CompilerResult
        '''
        assert isinstance(target_unitary, Unitary)
        assert self.unitary_primitives

        initial_time = time.perf_counter()
        compiled_sequence, cost_by_step = self._compile(
            target_unitary, threshold, max_step_count)
        total_elapsed_time = time.perf_counter() - initial_time

        result = CompilerResult(
            compiled_sequence, cost_by_step, total_elapsed_time)
        return result

    def compile_layered(
        self,
        target_unitary: Unitary,
        unitary_primitive_counts: Dict[UnitaryPrimitive, int],
        threshold: Optional[float] = None,
        max_step_count: int = np.iinfo(np.int32).max
    ) -> CompilerResult:
        '''
        Compiles a unitary sequence approximately implementing the specified
        unitary using the layered STOQ compilation technique.

        :param target_unitary: The target unitary to be compiled.
        :type target_unitary: Unitary
        :param unitary_primitive_counts: Specifies the fixed set of unitary
            primitives to be contained in each layer of the compilation.
            Each key is the unitary primitive to be included, and each value
            is the count of that unitary primitive per layer.
        :type unitary_primitive_counts: Dict[UnitaryPrimitive, int]
        :param threshold: The overlap with the target unitary at which to
            stop compilation, defaults to None. A value of 1.0 implies an exact
            compilation. If None, a threshold of 1.0 is used.
        :type threshold: Optional[float], optional
        :param max_step_count: Maximum number of steps to perform while
            attempting to perform the approximate compilation, defaults to
            np.iinfo(np.int32).max. Compilation will terminate after
            this number of steps regardless of whether the threshold has
            been reached.
        :type max_step_count: int, optional
            the compilation.
        :rtype: CompilerResult
        '''
        assert isinstance(target_unitary, Unitary)
        assert isinstance(unitary_primitive_counts, dict)

        initial_time = time.perf_counter()
        compiled_sequence, cost_by_step = self._compile_layered(
            target_unitary, unitary_primitive_counts, threshold,
            max_step_count)
        total_elapsed_time = time.perf_counter() - initial_time

        result = CompilerResult(
            compiled_sequence, cost_by_step, total_elapsed_time)
        return result

    def _compile(
        self,
        target_unitary: Unitary,
        threshold: float,
        max_step_count: int
    ) -> CompilerResult:
        '''
        Internal implementation of STOQ compilation.
        See documentation for Compiler.compile() for full details.
        '''
        compiled_sequence = UnitarySequence(self.dimension)
        cost_by_step = []

        while not (
                compiled_sequence.product().close_to(target_unitary, threshold)
                and len(cost_by_step) < max_step_count):
            self.beta = min(self.beta + self.annealing_rate, self.max_beta)
            product_before_change = compiled_sequence.product()
            self._make_random_change(compiled_sequence)
            current_cost = target_unitary.distance_from(product_before_change)
            proposed_cost = target_unitary.distance_from(
                compiled_sequence.product())
            accept = self._accept_proposed_change(
                target_unitary, current_cost, proposed_cost)
            if accept:
                cost_by_step.append(proposed_cost)
            else:
                compiled_sequence.undo()
                cost_by_step.append(current_cost)

        return compiled_sequence, cost_by_step

    @staticmethod
    def create_random_sequence_entry(
        dimension: int,
        unitary_primitives: List[UnitaryPrimitive],
        probabilities: Optional[List[float]] = None
    ) -> UnitarySequenceEntry:
        '''
        Creates and returns a randomly-generated unitary sequence entry.

        :param dimension: The dimension of the state space. For an n-qubit
            system, dimension should be set to 2**n.
        :type dimension: int
        :param unitary_primitives: The unitary primitives from which to
            choose when randomly generating a sequence entry.
        :type unitary_primitives: List[UnitaryPrimitive]
        :param probabilities: The probability for STOQ to
            choose each of the primitives specified in unitary_primitives when
            proposing new gates at each step of the compilation process,
            defaults to None. If not specified, each unitary primitive is
            chosen with uniform probability.
        :type probabilities: Optional[List[float]], optional
        :return: A sequence entry specifying a randomly-chosen unitary
            with randomly-chosen parameters and applied to a randomly-chosen
            set of qubits.
        :rtype: UnitarySequenceEntry
        '''
        # choose from unitary_primitives where the allowed_apply_to list
        # is not empty
        unitary_primitives = [
            u for u in unitary_primitives
            if (u.get_allowed_apply_to() is None
                or len(u.get_allowed_apply_to()) > 0)]

        # randomly choose one of the unitary primitives and assign random
        # parameter values
        new_unitary_primitive = np.random.choice(
            unitary_primitives, p=probabilities)
        new_unitary = new_unitary_primitive.get_unitary()
        if isinstance(new_unitary, ParameterizedUnitary):
            random_parameter_values = [
                p.random_value() for p in new_unitary.get_parameters()]
            new_unitary = new_unitary.as_unitary(random_parameter_values)
        assert isinstance(new_unitary, Unitary)

        # randomly choose the qubit(s) to which to apply the unitary
        num_system_qubits = int(np.log2(dimension))
        if new_unitary_primitive.get_allowed_apply_to() is not None:
            allowed_apply_to = new_unitary_primitive.get_allowed_apply_to()
            apply_to = allowed_apply_to[
                np.random.choice(len(allowed_apply_to))]
        else:
            num_unitary_qubits = int(np.log2(new_unitary.get_dimension()))
            apply_to = np.random.choice(list(range(num_system_qubits)),
                                        num_unitary_qubits, replace=False)

        return UnitarySequenceEntry(new_unitary, apply_to)

    def _compile_layered(
        self,
        target_unitary: Unitary,
        unitary_primitive_counts: Dict[UnitaryPrimitive, int],
        threshold: float,
        max_step_count: int
    ) -> CompilerResult:
        '''
        Internal implementation of layered STOQ compilation.
        See documentation for Compiler.compile_layered() for full details.
        '''
        compiled_sequence = UnitarySequence(self.dimension)
        cost_by_step = []

        while not (
                compiled_sequence.product().close_to(target_unitary, threshold)
                and len(cost_by_step) < max_step_count):
            self.beta = min(self.beta + self.annealing_rate, self.max_beta)
            product_before_change = compiled_sequence.product()
            self._make_random_change_layered(
                compiled_sequence, unitary_primitive_counts)
            current_cost = target_unitary.distance_from(product_before_change)
            proposed_cost = target_unitary.distance_from(
                compiled_sequence.product())
            accept = self._accept_proposed_change(
                target_unitary, current_cost, proposed_cost)
            if accept:
                cost_by_step.append(proposed_cost)
            else:
                compiled_sequence.undo()
                cost_by_step.append(current_cost)

        return compiled_sequence, cost_by_step

    @staticmethod
    def create_random_layer(
        dimension: int,
        unitary_primitive_counts: Dict[UnitaryPrimitive, int]
    ) -> List[UnitarySequenceEntry]:
        '''
        Creates a layer of randomly-generated unitary sequence
        entries with the specified unitary primitive counts.

        :param dimension: The dimension of the state space. For an n-qubit
            system, dimension should be set to 2**n.
        :type dimension: int
        :param unitary_primitive_counts: Specifies the fixed set of unitary
            primitives to be contained in each layer of the compilation.
            Each key is the unitary primitive to be included, and each value
            is the count of that unitary primitive per layer.
        :type unitary_primitive_counts: Dict[UnitaryPrimitive, int]
        :param threshold: The overlap with the target unitary at which to
            stop compilation, defaults to None. A value of 1.0 implies an exact
            compilation. If None, a threshold of 1.0 is used.
        :type threshold: Optional[float], optional
        :return: A randomly-ordered list of sequence entries consisting
            of the specified counts of each unitary primitive, with each
            given randomly-chosen parameters and applied to a randomly-chosen
            set of qubits.
        :rtype: List[UnitarySequenceEntry]
        '''
        layer = []
        for primitive, count in unitary_primitive_counts.items():
            layer.extend([
                Compiler.create_random_sequence_entry(dimension, [primitive])
                for _ in range(count)])
        np.random.shuffle(layer)
        return layer

    def _make_random_change(
        self,
        compiled_sequence: UnitarySequence
    ) -> None:
        '''
        Chooses and implements a random change to the specified
        unitary sequence.
        '''
        count_append = np.count_nonzero([
            CompilerAction.is_append(action)
            for action in list(CompilerAction)])
        count_non_append = len(CompilerAction) - count_append
        p_append = self.append_probability / count_append
        p_non_append = (1 - self.append_probability) / count_non_append
        action = np.random.choice(list(CompilerAction), 1, p=[
            p_append if CompilerAction.is_append(action) else p_non_append
            for action in list(CompilerAction)])

        new_sequence_entry = None
        if CompilerAction.is_append(action):
            new_sequence_entry = Compiler.create_random_sequence_entry(
                self.dimension, self.unitary_primitives,
                self.unitary_primitive_probabilities)

        if action == CompilerAction.AppendFirst:
            compiled_sequence.append_first(new_sequence_entry)
        elif action == CompilerAction.AppendLast:
            compiled_sequence.append_last(new_sequence_entry)
        elif action == CompilerAction.RemoveFirst:
            compiled_sequence.remove_first()
        elif action == CompilerAction.RemoveLast:
            compiled_sequence.remove_last()

    def _make_random_change_layered(
        self,
        compiled_sequence: UnitarySequence,
        unitary_primitive_counts: Dict[UnitaryPrimitive, int]
    ) -> None:
        '''
        Chooses and implements a random change to the specified
        layered unitary sequence.
        '''
        count_append = np.count_nonzero([
            CompilerAction.is_append(action)
            for action in list(CompilerAction)])
        count_non_append = len(CompilerAction) - count_append
        p_append = self.append_probability / count_append
        p_non_append = (1 - self.append_probability) / count_non_append
        action = np.random.choice(list(CompilerAction), 1, p=[
            p_append if CompilerAction.is_append(action) else p_non_append
            for action in list(CompilerAction)])

        layer_length = sum(unitary_primitive_counts.values())
        new_layer = None
        if CompilerAction.is_append(action):
            new_layer = Compiler.create_random_layer(
                self.dimension, unitary_primitive_counts)

        # only save the undo state for the first modification to the sequence
        # this is so that a future undo() call will reverse the entire action
        if action == CompilerAction.AppendFirst:
            for i, sequence_entry in enumerate(new_layer):
                compiled_sequence.append_first(
                    sequence_entry, save_undo=(i == 0))
        elif action == CompilerAction.AppendLast:
            for i, sequence_entry in enumerate(new_layer):
                compiled_sequence.append_last(
                    sequence_entry, save_undo=(i == 0))
        elif action == CompilerAction.RemoveFirst:
            for i in range(layer_length):
                compiled_sequence.remove_first(save_undo=(i == 0))
        elif action == CompilerAction.RemoveLast:
            for i in range(layer_length):
                compiled_sequence.remove_last(save_undo=(i == 0))

    def _accept_proposed_change(
        self,
        target_unitary: Unitary,
        current_cost: float,
        proposed_cost: float
    ) -> bool:
        '''
        Determines whether to accept the proposed change, given the current
        cost and proposed cost and current annealing parameter value beta.
        '''
        cost_difference = proposed_cost - current_cost
        acceptance_probability = min(1.0, np.exp(-self.beta * cost_difference))
        return bool(np.random.uniform() < acceptance_probability)
