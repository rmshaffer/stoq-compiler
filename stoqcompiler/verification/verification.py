'''
Defines the Verification class.
'''
import numpy as np
from typing import Dict, List, Optional

from stoqcompiler.unitary import UnitaryPrimitive, UnitarySequence
from stoqcompiler.compiler import Compiler, CompilerResult


class Verification:
    '''
    Implements verification techniques using the STOQ compiler.
    '''
    @staticmethod
    def generate_rav_sequence(
        dimension: int,
        unitary_primitives: List[UnitaryPrimitive],
        sequence_length: int,
        threshold: float,
        stoq_append_probability: float = 0.5,
        unitary_primitive_probabilities: Optional[List[float]] = None
    ) -> CompilerResult:
        '''
        Implements randomized analog verification (RAV) as per
        Shaffer et al., arXiv:2003.04500 (2020).

        :param dimension: The dimension of the state space. For an n-qubit
        system, dimension should be set to 2**n.
        :type dimension: int
        :param unitary_primitives: The unitary primitives to be used for
        the compilation.
        :type unitary_primitives: List[UnitaryPrimitive]
        :param sequence_length: The length of the initial randomly-generated
        sequence.
        :type sequence_length: int
        :param threshold: The overlap with the target unitary at which to
        stop compilation, defaults to None. A value of 1.0 implies an exact
        compilation. If None, a threshold of 1.0 is used.
        :type threshold: float
        :param stoq_append_probability: Probability of appending a new gate
        at each step in the compilation, defaults to 0.5.
        :type stoq_append_probability: float, optional
        :param unitary_primitive_probabilities: The probability for STOQ to
        choose each of the primitives specified in unitary_primitives when
        proposing new gates at each step of the compilation process, defaults
        to None. If not specified, each unitary primitive is chosen with
        uniform probability.
        :type unitary_primitive_probabilities: Optional[List[float]], optional
        :return: The result of the compilation, including the RAV sequence.
        :rtype: CompilerResult
        '''
        assert (isinstance(unitary_primitives, list)
                or isinstance(unitary_primitives, np.ndarray))
        assert np.all([
            isinstance(primitive, UnitaryPrimitive)
            for primitive in unitary_primitives])
        assert np.all([
            primitive.get_unitary().get_dimension() <= dimension
            for primitive in unitary_primitives])
        assert sequence_length >= 0
        assert threshold >= 0.0 and threshold <= 1.0

        # Generate a random sequence of the desired length
        random_sequence = UnitarySequence(dimension)
        for _ in range(sequence_length):
            new_sequence_entry = Compiler.create_random_sequence_entry(
                dimension, unitary_primitives, unitary_primitive_probabilities)
            random_sequence.append_last(new_sequence_entry)

        # Calculate the product of this sequence and invert it
        target_unitary = random_sequence.product().inverse()

        # Use Compiler to compile a new sequence implementing the inverse
        compiler = Compiler(
            dimension, stoq_append_probability)
        compiler.set_unitary_primitives(
            unitary_primitives, unitary_primitive_probabilities)
        result = compiler.compile(
            target_unitary, threshold, max_step_count=10000)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(
            random_sequence, result.compiled_sequence)
        return result

    @staticmethod
    def generate_layered_rav_sequence(
        dimension: int,
        unitary_primitive_counts: Dict[UnitaryPrimitive, int],
        layer_count: int,
        threshold: float,
        stoq_append_probability: float = 0.5,
        max_step_count: int = 10000
    ) -> CompilerResult:
        '''
        Implements layered randomized analog verification (RAV).

        :param dimension: [description]
        :type dimension: int
        :param unitary_primitive_counts: Specifies the fixed set of unitary
        primitives to be contained in each layer of the compilation. Each key
        is the unitary primitive to be included, and each value is the count
        of that unitary primitive per layer.
        :type unitary_primitive_counts: Dict[UnitaryPrimitive, int]
        :param layer_count: The number of layers to create in the initial
        randomly-generated sequence.
        :type layer_count: int
        :param threshold: The overlap with the target unitary at which to
        stop compilation, defaults to None. A value of 1.0 implies an exact
        compilation. If None, a threshold of 1.0 is used.
        :type threshold: float
        :param stoq_append_probability: Probability of appending a new gate
        at each step in the compilation, defaults to 0.5.
        :type stoq_append_probability: float, optional
        :param max_step_count: Maximum number of steps to perform while
        attempting to perform the approximate compilation, defaults to 10000.
        Compilation of the inversion sequence will terminate after this number
        of steps regardless of whether the threshold has been reached.
        :type max_step_count: int, optional
        :return: The result of the compilation, including the layered
        RAV sequence.
        :rtype: CompilerResult
        '''
        assert isinstance(unitary_primitive_counts, dict)
        assert np.all([
            isinstance(primitive, UnitaryPrimitive)
            for primitive in unitary_primitive_counts.keys()])
        assert np.all([
            primitive.get_unitary().get_dimension() <= dimension
            for primitive in unitary_primitive_counts.keys()])
        assert np.all([
            isinstance(count, int)
            for count in unitary_primitive_counts.values()])
        assert layer_count >= 0
        assert threshold >= 0.0 and threshold <= 1.0

        # Generate a random sequence of the desired number of layers
        # Total sequence length will therefore be
        # sum(unitary_primitive_counts.values()) * layer_count
        random_sequence = UnitarySequence(dimension)
        for _ in range(layer_count):
            layer = Compiler.create_random_layer(
                dimension, unitary_primitive_counts)
            for sequence_entry in layer:
                random_sequence.append_last(sequence_entry)

        # Calculate the product of this sequence and invert it
        target_unitary = random_sequence.product().inverse()

        # Use Compiler to compile a new sequence implementing the inverse
        compiler = Compiler(dimension, stoq_append_probability)
        result = compiler.compile_layered(
            target_unitary, unitary_primitive_counts,
            threshold, max_step_count)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(
            random_sequence, result.compiled_sequence)
        return result
