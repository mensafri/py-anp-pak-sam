import numpy as np
from scipy.linalg import eig


def calculate_priority(matrix):
    eigvals, eigvecs = eig(matrix)
    max_eigval = np.max(eigvals)
    index = np.argmax(eigvals)
    priority_vector = np.real(eigvecs[:, index])
    priority_vector = priority_vector / np.sum(priority_vector)
    return priority_vector
