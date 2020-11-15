import numpy as np

class HamiltonianTerm:

    def __init__(self, matrix):
        assert isinstance(matrix, np.ndarray)
        matrix = matrix.astype(np.complex128)

        # verify that the matrix is square
        self.dimension = matrix.shape[0]
        assert matrix.shape == (self.dimension, self.dimension)

        # verify that the matrix is Hermitian
        assert np.allclose(matrix, matrix.T.conj())

        # Split the matrix into a coefficient and a normalized matrix, such that:
        #  - largest singular value of normalized matrix is 1
        #  - coefficient is positive and real
        _, s, _ = np.linalg.svd(matrix)
        self.coefficient = np.max(s)
        self.normalized_matrix = matrix / self.coefficient
        assert np.isreal(self.coefficient)
        assert self.coefficient >= 0.0

        # validate that we did the normalization correctly
        assert np.allclose(matrix, self.get_matrix())

    def get_dimension(self):
        return self.dimension

    def get_matrix(self):
        return self.coefficient * self.normalized_matrix

    def get_coefficient(self):
        return self.coefficient

    def get_normalized_matrix(self):
        return self.normalized_matrix
        