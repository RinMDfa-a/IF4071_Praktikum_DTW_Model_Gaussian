import random
from typing import Literal

import numpy as np


class DynamicTimeWarping:
    """Implementation from scratch of averaged dynamic time warping"""

    def __init__(self, samples: list[np.ndarray], mode=Literal['DTW','DBA']) -> None:
        """Initializes the DTW"""
        self.samples = samples
        self.mode = mode

    def fit(self, max_iter: int | None = None) -> None:
        """Creates the template"""
        if self.mode == "DBA" and (max_iter == None):
            raise Exception.args("Type DBA needs iteration and threshold to be explicitly set")

        self.max_iter = max_iter
        self._align_samples()
        self.template = self.ref_sample

    def _medoid(self) -> np.ndarray:
        """Returns the seed for the first center by getting the idx of the smallest sum between the
        samples"""

        medoid_idx = -1
        best_ss = 1e20

        for idx_candidate, sample in enumerate(self.samples):
            ss = self._compare_dtw_scores(sample, self.samples)
            if medoid_idx == -1 or ss < best_ss:
                best_ss = ss
                medoid_idx = idx_candidate
        return medoid_idx

    def _compare_dtw_scores(self, curr_sample: np.ndarray, samples:np.ndarray) -> float:
        """Returns the total score of DTWs calculations"""
        score = 0
        for sample in samples:
            cost_matrix = self._fill_cost_matrix(sample, curr_sample)
            delta_matrix = self._fill_delta_matrix(cost_matrix)
            score += delta_matrix[sample.shape[0] - 1, curr_sample.shape[0] - 1]
        return score

    def _align_samples(self) -> None:
        """Align the samples to average the template by asigning a random sample as the main reference"""

        if self.mode == 'DBA':
            # Choose a sample using medoid by getting the idx
            self.ref_sample = self.samples[self._medoid()]

            # DBA (Barycenter Averaging)
            for _ in range (self.max_iter):
                updated_center = np.zeros(self.ref_sample.shape) # Updated_center shape to match the shape of the reference
                n_elements = np.zeros(len(self.ref_sample), dtype=int) # n_elements shape to match the shape of reference.shape[0] which are the frames

                for sample in self.samples:
                    cost_matrix = self._fill_cost_matrix(self.ref_sample, sample)
                    delta_matrix = self._fill_delta_matrix(cost_matrix)
                    _, path_alignment = self._dtw_alignment(delta_matrix) 

                    for N, M in path_alignment:
                        updated_center[N] += sample[M]
                        n_elements[N] += 1

                n_elements[n_elements == 0] = 1
                self.ref_sample = updated_center / n_elements[:, np.newaxis] # Enables broadcasting (100,39) (100,) -> (100, 39) (100, 1)
        else:
            self.ref_sample = random.choice(self.samples)

            summation_placeholder = np.zeros(self.ref_sample.shape) # Updated_center shape to match the shape of the reference
            n_elements = np.zeros(len(self.ref_sample), dtype=int) # n_elements shape to match the shape of reference.shape[0] which is the frames

            for sample in self.samples:
                cost_matrix = self._fill_cost_matrix(self.ref_sample, sample)
                delta_matrix = self._fill_delta_matrix(cost_matrix)
                _, path_alignment = self._dtw_alignment(delta_matrix)

                for N, M in path_alignment:
                    summation_placeholder[N] += sample[M]
                    n_elements[N] += 1 

            n_elements[n_elements == 0] = 1
            self.ref_sample = summation_placeholder / n_elements[:, np.newaxis]

    def _dtw_alignment(self,delta_matrix: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
        """Backtrack using the delta matrix"""
        N = delta_matrix.shape[0] - 1
        M = delta_matrix.shape[1] - 1

        alignment_score = delta_matrix[N, M]
        alignment_indices = []

        while (N > 0 or M > 0):

            alignment_indices.append((N, M)) # Extract indices
            if (N > 0 and M > 0):
                moves = {
                    (N - 1, M - 1): delta_matrix[N - 1, M - 1], # diagonal
                    (N - 1, M): delta_matrix[N - 1, M], # vertical
                    (N, M - 1): delta_matrix[N, M - 1] # horizontal
                }

                N, M = min(moves, key=moves.get)

            # Handle edge case
            elif N == 0: # first row
                M -= 1

            else: # first column
                N -= 1

        # Include last indice since the loop stops directly at (0,0)
        alignment_indices.append((0, 0))

        return alignment_score, alignment_indices

    def _fill_delta_matrix(self, cost_matrix:np.ndarray) -> np.ndarray:
        """Fills the delta matrix for preparing backtracking in DTW alignment"""

        delta_matrix = np.zeros(shape=(cost_matrix.shape))
        N = cost_matrix.shape[0] 
        M = cost_matrix.shape[1]

        # The rules strictly states that a cell can progress to the next cell by going
        # horizontally, diagonally, and vertically. So that gives 2 implications for the 
        # first row and the first column.
        # 1. The first row can only be accessed horizontally
        # 2. The first column can only be accesed vertically

        delta_matrix[0, 0] = cost_matrix[0, 0]

        # Fill the rows
        for i in range(1, N):
            delta_matrix[i, 0] = delta_matrix[i - 1, 0] + cost_matrix[i, 0]

        # Fill the columns
        for i in range(1, M):
            delta_matrix[0, i] = delta_matrix[0, i - 1] + cost_matrix[0, i]

        # Fill remainings
        for i in range(1, N):
            for j in range(1, M):
                delta_matrix[i, j] = cost_matrix[i, j] + min(
                   delta_matrix[i - 1, j], # Vertical
                   delta_matrix[i, j - 1], # Horizontal
                   delta_matrix[i - 1, j - 1] # Diagonal
                )

        return delta_matrix

    def _fill_cost_matrix(self, reference: np.ndarray, sample: np.ndarray) -> np.ndarray:
        """Fill the cost matrix with the size of N x M"""
        N = reference.shape[0]
        M = sample.shape[0]

        cost_matrix = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                cost_matrix[i, j] = self._calculate_distance(reference[i], sample[j])

        return cost_matrix

    def _calculate_distance(self, seqA: np.ndarray, seqB: np.ndarray) -> float:
        """Calculate the distance between 2 mfcc sequence using euclidean"""
        return np.linalg.norm(seqA - seqB)

    def model(self, input: np.ndarray) -> float:
        """Initiate modelling comparison for the input mfcc against the existing templates"""

        cost_matrix = self._fill_cost_matrix(self.ref_sample, input)
        delta_matrix = self._fill_delta_matrix(cost_matrix)
        score, _ = self._dtw_alignment(delta_matrix)

        return score

if __name__ == "__main__":
    data = np.load("../../audio/preprocessed_extracted_samples.npz", allow_pickle=True)
    X = data['x']
    y = data['y']

    X_a = X[y == 'a']
    X_e = X[y == 'e']
    X_i = X[y == 'i']
    X_o = X[y == 'o']
    X_u = X[y == 'u']

    print(len(X))

    dtw_a_template = DynamicTimeWarping(X, mode="DBA")
    dtw_a_template.fit(max_iter=10)

    print(dtw_a_template.template)