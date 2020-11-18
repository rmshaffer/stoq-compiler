import numpy as np


class UnitaryPrimitive:

    def __init__(self, unitary, allowed_apply_to=None):
        if allowed_apply_to is not None:
            allowed_apply_to = list(allowed_apply_to)
            assert np.all([
                len(apply_to) == len(set(apply_to))
                for apply_to in allowed_apply_to])
            assert np.all([
                2**len(apply_to) == unitary.get_dimension()
                for apply_to in allowed_apply_to])
            assert np.all([
                np.min(apply_to) >= 0
                for apply_to in allowed_apply_to])

        self.unitary = unitary
        self.allowed_apply_to = allowed_apply_to

    def get_unitary(self):
        return self.unitary

    def get_allowed_apply_to(self):
        return self.allowed_apply_to
