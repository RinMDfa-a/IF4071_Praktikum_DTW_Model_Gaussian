import numpy as np

from typing import Literal


class DynamicTimeWarping:
    """Implementation from scratch of averaged dynamic time warping"""

    def __init__(self, samples: list[np.ndarray], mode=Literal['default','DBA']) -> None:
        """Initializes the DTW"""
        self.samples = samples
        self.mode = mode

    def fit(self, max_iter: int) -> None:
        """Creates the template"""
        self.template = None

    def _align_samples(self, max_iter: int, threshold: float):
        """Align the samples to average the template by asigning a random sample as the main reference"""

        # Choose a random sample as a reference
        rng = np.random.default_rng()
        ref_idx = rng.choice(len(self.samples))
        self.ref_sample = self.samples[ref_idx]

        # Extract the samples except for the reference sample
        samples = [sample for idx, sample in enumerate(self.samples) if idx != ref_idx]

        if self.mode == 'DBA':
            # DBA (Barycenter Averaging)
            for i in range (max_iter):
                accum_sum = np.zeros(len(self.ref_sample))
                accum_cnt = np.zeros(len(self.ref_sample))

                for sample in samples:
                    cost_matrix = self._fill_cost_matrix(self.ref_sample, sample)
                    delta_matrix = self._fill_delta_matrix(cost_matrix)
                    _, path_alignment = self._dtw_alignment(delta_matrix) 

                    for N, M in path_alignment:
                        accum_sum[N] += sample[M]
                        accum_cnt[N] += 1

                    # Handling case where there are no alignment resulting in 0 counts for that specific frame on the reference sampling
                    accum_cnt = accum_cnt[accum_cnt == 0] = 1.0

                self.ref_sample = accum_sum / accum_cnt
        else:
            pass

    def _dtw_alignment(self,delta_matrix: np.ndarray, sample: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
        """Backtrack using the delta matrix"""
        N = delta_matrix.shape[0] - 1
        M = delta_matrix.shape[1] - 1

        alignment_score = delta_matrix[N, M]
        alignment_indices = []

        while (N > 0 or M > 0):

            alignment_indices.append((N, M)) # Extract indices

            # Handle edge case
            if (N > 0 and M > 0):
                moves = {
                    (N - 1, M - 1): delta_matrix[N - 1, M - 1], # diagonal
                    (N - 1, M): delta_matrix[N - 1, M], # vertical
                    (N, M - 1): delta_matrix[N, M - 1] # horizontal
                }

                N, M = min(moves, key=moves.get)

            elif N == 0: # first row
                M -= 1

            else: # first column
                N -= 1

        # Include last indice since the loop stops directly at (0,0)
        alignment_indices.append((0, 0))

        return alignment_score, alignment_indices

    def _fill_delta_matrix(self, cost_matrix:np.ndarray):
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
        