'''
Defines the HamiltonianTerm class.
'''
import numpy as np


class HamiltonianTerm:
    '''
    Defines a single term of a Hamiltonian.
    '''
    def __init__(
        self,
        matrix: np.ndarray
    ):
        '''
        Creates a HamiltonianTerm object from the specified matrix.

        :param matrix: The square Hermitian matrix representing the
            action of the Hamiltonian term.
        :type matrix: np.ndarray
        '''
        assert isinstance(matrix, np.ndarray)
        matrix = matrix.astype(np.complex128)

        # verify that the matrix is square
        self.dimension = matrix.shape[0]
        assert matrix.shape == (self.dimension, self.dimension)

        # verify that the matrix is Hermitian
        assert np.allclose(matrix, matrix.T.conj())

        # Split the matrix into a coefficient and a normalized matrix,
        # such that:
        #  - largest singular value of normalized matrix is 1
        #  - coefficient is positive and real
        _, s, _ = np.linalg.svd(matrix)
        self.coefficient = np.max(s)
        self.normalized_matrix = matrix / self.coefficient
        assert np.isreal(self.coefficient)
        assert self.coefficient >= 0.0

        # validate that we did the normalization correctly
        assert np.allclose(matrix, self.get_matrix())

    def get_dimension(self) -> int:
        '''
        Gets the dimension of the state space on which
        this Hamiltonian term acts.

        :return: The state space dimension.
        :rtype: int
        '''
        return self.dimension

    def get_matrix(self) -> np.ndarray:
        '''
        Gets the matrix representing the action of the Hamiltonian term.

        :return: The square Hermitian matrix representing the action of
            the Hamiltonian term.
        :rtype: np.ndarray
        '''
        return self.coefficient * self.normalized_matrix

    def get_coefficient(self) -> float:
        '''
        Gets the normalization coefficient calculated for this matrix such
        that the coefficient is real and positive.

        :return: The normalization coefficient.
        :rtype: float
        '''
        return self.coefficient

    def get_normalized_matrix(self) -> np.ndarray:
        '''
        Gets the normalized matrix for this Hamiltonian term such that
        the largest singular value is 1.

        :return: The normalized matrix.
        :rtype: np.ndarray
        '''
        return self.normalized_matrix
