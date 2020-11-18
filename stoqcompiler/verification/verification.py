import numpy as np

from stoqcompiler.unitary import UnitaryPrimitive, UnitarySequence
from stoqcompiler.compiler import Compiler


class Verification:

    @staticmethod
    def generate_rav_sequence(
            dimension, unitary_primitives, sequence_length,
            threshold, mcmc_append_probability=0.5,
            unitary_primitive_probabilities=None):
        # Randomized analog verification (RAV) as per Shaffer et al.,
        # arXiv:2003.04500 (2020)
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
            dimension, mcmc_append_probability,
            unitary_primitive_probabilities)
        compiler.set_unitary_primitives(unitary_primitives)
        result = compiler.compile(
            target_unitary, threshold, max_step_count=10000)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(
            random_sequence, result.compiled_sequence)
        return result

    @staticmethod
    def generate_layered_rav_sequence(
            dimension, unitary_primitive_counts, layer_count,
            threshold, mcmc_append_probability=0.5, max_step_count=10000):
        # Layered randomized analog verification (RAV) sequences
        # unitary_primitive_counts is a dictionary mapping each unitary
        # primitive to its count per layer
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
        compiler = Compiler(dimension, mcmc_append_probability)
        result = compiler.compile_layered(
            target_unitary, unitary_primitive_counts,
            threshold, max_step_count)

        # Return the CompilerResult with the combined sequence
        result.compiled_sequence = UnitarySequence.combine(
            random_sequence, result.compiled_sequence)
        return result
