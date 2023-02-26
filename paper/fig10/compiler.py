import copy
import numpy as np
from unitary import *
import time

class CompilerResult:

    def __init__(self, compiled_sequence, cost_by_step, total_elapsed_time):
        self.compiled_sequence = compiled_sequence
        self.cost_by_step = cost_by_step
        self.total_elapsed_time = total_elapsed_time

class Compiler:

    def __init__(self, dimension):
        assert dimension > 0
        self.dimension = dimension
        self.unitary_primitives = []

    def set_unitary_primitives(self, unitary_primitives):
        assert isinstance(unitary_primitives, list) or isinstance(unitary_primitives, np.ndarray)
        assert np.all([isinstance(primitive, UnitaryPrimitive) for primitive in unitary_primitives])
        assert np.all([primitive.get_unitary().get_dimension() <= self.dimension for primitive in unitary_primitives])

        self.unitary_primitives = copy.deepcopy(unitary_primitives)

    def compile(self, target_unitary, threshold=None, max_step_count=np.iinfo(np.int32).max):
        assert isinstance(target_unitary, Unitary)
        assert self.unitary_primitives

        initial_time = time.perf_counter()
        compiled_sequence, cost_by_step = self._compile(target_unitary, threshold, max_step_count)
        total_elapsed_time = time.perf_counter() - initial_time

        result = CompilerResult(compiled_sequence, cost_by_step, total_elapsed_time)
        return result

    def compile_layered(self, target_unitary, unitary_primitive_counts, threshold=None, max_step_count=np.iinfo(np.int32).max):
        assert isinstance(target_unitary, Unitary)
        assert isinstance(unitary_primitive_counts, dict)

        initial_time = time.perf_counter()
        compiled_sequence, cost_by_step = self._compile_layered(target_unitary, unitary_primitive_counts, threshold, max_step_count)
        total_elapsed_time = time.perf_counter() - initial_time

        result = CompilerResult(compiled_sequence, cost_by_step, total_elapsed_time)
        return result

    def _compile(self, target_unitary, threshold, max_step_count):
        raise NotImplementedError
