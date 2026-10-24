import os
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap


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
        """Returns the index for the first center by getting the idx of the smallest cost between the
        samples"""

        medoid_idx = -1
        best_ss = 1e20

        for idx_candidate, sample in enumerate(self.samples):
            ss = self._compare_dtw_scores(sample, self.samples)

            # Heuristik panjang: cost / weight
            weight = self._length_weight(sample, self.samples)
            ss = ss / weight if weight > 0 else np.inf

            if medoid_idx == -1 or ss < best_ss:
                best_ss = ss
                medoid_idx = idx_candidate
        return medoid_idx

    def _compare_dtw_scores(self, curr_sample: np.ndarray, samples:np.ndarray) -> float:
        """Returns the total score of DTWs calculations"""
        score = 0
        for sample in samples:
            # cost_matrix = self._fill_cost_matrix(sample, curr_sample)
            # delta_matrix = self._fill_delta_matrix(cost_matrix)
            # score += delta_matrix[sample.shape[0] - 1, curr_sample.shape[0] - 1]

            cost_matrix = self._fill_cost_matrix(curr_sample, sample)
            delta_matrix = self._fill_delta_matrix(cost_matrix)
            score += delta_matrix[curr_sample.shape[0] - 1, sample.shape[0] - 1]
        return score

    def _length_weight(self, curr_sample: np.ndarray, samples: list[np.ndarray]) -> float:
        """Weight [0, 1]: min weight dari semua sample

        r = panjang sample / panjang curr_sample
        r < 0.5      : 0      -> template maks dipadatkan 2x, sample tidak bisa di-align
        0.5 <= r <= 1: 2r - 1 -> makin dekat batas 2x, makin kecil
        r > 1        : 1 / r  -> tetap bisa, tapi banyak frame sample ditumpuk ke 1 frame template
        """
        weights = []
        for sample in samples:
            r = len(sample) / len(curr_sample)
            if r < 0.5:
                weights.append(0.0)
            elif r <= 1:
                weights.append(2 * r - 1)
            else:
                weights.append(1 / r)

        return min(weights)

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

                # Frame template yang skip -> pakai nilai lama
                updated_center[n_elements == 0] = self.ref_sample[n_elements == 0]

                n_elements[n_elements == 0] = 1
                self.ref_sample = updated_center / n_elements[:, np.newaxis] # Enables broadcasting (100,39) (100,) -> (100, 39) (100, 1)
        else:
            self.ref_sample = self.samples[self._medoid()]

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

        # (input < template / 2) -> tidak ada path
        if np.isinf(alignment_score):
            return alignment_score, alignment_indices

        while (N > 0 or M > 0):

            alignment_indices.append((N, M)) # Extract indices
            if (N > 0 and M > 0):
                moves = {
                    (N - 1, M - 1): delta_matrix[N - 1, M - 1], # diagonal
                    # (N - 1, M): delta_matrix[N - 1, M], # vertical
                    (N, M - 1): delta_matrix[N, M - 1] # horizontal
                }

                # Skip 1 frame template
                if N >= 2:
                    moves[(N - 2, M - 1)] = delta_matrix[N - 2, M - 1] # skip

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

        # P[i, j] = min(P[i, j-1], P[i-1, j-1], P[i-2, j-1]) + C[i, j]
        # i: frame template, j: frame input. 

        delta_matrix[0, 0] = cost_matrix[0, 0]

        # Fill the rows
        for i in range(1, N):
            # delta_matrix[i, 0] = delta_matrix[i - 1, 0] + cost_matrix[i, 0]
            delta_matrix[i, 0] = np.inf

        # Fill the columns
        for i in range(1, M):
            delta_matrix[0, i] = delta_matrix[0, i - 1] + cost_matrix[0, i]

        # Fill remainings
        for i in range(1, N):
            for j in range(1, M):
                delta_matrix[i, j] = cost_matrix[i, j] + min(
                   # delta_matrix[i - 1, j], # Vertical
                   delta_matrix[i, j - 1], # Horizontal
                   delta_matrix[i - 1, j - 1], # Diagonal
                   delta_matrix[i - 2, j - 1] if i >= 2 else np.inf # Skip
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

    def display_dtw_path(self, path: list[tuple[int, int]]) -> None:
        """Outputs a graph that displays the path of the DTW between the template and the input on the delta matrix"""
        # ax = heatmap(data=self.delta_matrix, linewidths=.5)

        # inf -> nan: node tak terjangkau tampil kosong
        ax = heatmap(data=np.where(np.isinf(self.delta_matrix), np.nan, self.delta_matrix), linewidths=.5)
        ax.plot([m + 0.5 for _, m in path], # M is the input so it should be the X
                [n + 0.5 for n, _ in path], # N is the ref so it should be the Y
                color="white",
                marker="^",
                label="Alignment Path")
        plt.legend()
        plt.show()

    def display_matrix(self, matrix: np.ndarray) -> None:
        """Outputs a display matrix that displays a value in each of the grid"""
        # heatmap(data=matrix, linewidths=.5)

        # inf -> nan: node tak terjangkau tampil kosong, skala warna tidak rusak
        heatmap(data=np.where(np.isinf(matrix), np.nan, matrix), linewidths=.5)
        plt.show()

    def model(self, input: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
        """Initiate modelling comparison for the input mfcc against the existing templates"""

        self.cost_matrix = self._fill_cost_matrix(self.ref_sample, input)
        self.delta_matrix = self._fill_delta_matrix(self.cost_matrix)
        score, path_alingment = self._dtw_alignment(self.delta_matrix)

        return score, path_alingment


class DTWRecognizer:
    """Recognizer DTW: 1 averaged template per vowel -> input vs semua template -> cost terkecil"""

    def __init__(self, mode: Literal['DTW', 'DBA'] = 'DBA', max_iter: int | None = 5) -> None:
        self.mode = mode
        self.max_iter = max_iter
        self.templates = {}

    def fit(self, X: list[np.ndarray], y: list[str]) -> None:
        """Template per vowel: X (list (N, 39)), y (label) -> self.templates"""
        for label in sorted(set(y)):
            samples = [x for x, lbl in zip(X, y) if lbl == label]
            dtw = DynamicTimeWarping(samples, mode=self.mode)
            dtw.fit(max_iter=self.max_iter)
            self.templates[label] = dtw

    def predict(self, x: np.ndarray) -> tuple[str, dict[str, float]]:
        """(N, 39) -> (label, total cost P[-1, -1] per vowel)

        Panjang path = panjang input untuk semua template"""
        scores = {}
        for label, dtw in self.templates.items():
            score, _ = dtw.model(x)
            scores[label] = score

        # (input < template / 2) -> tidak ada kandidat
        if all(np.isinf(score) for score in scores.values()):
            return "unknown", scores

        return min(scores, key=scores.get), scores


if __name__ == "__main__":
    # Path relatif
    npz_path = os.path.join(os.path.dirname(__file__), "../../data/preprocessed/template_preprocessed.npz")
    data = np.load(npz_path, allow_pickle=True)
    X = data['x']
    y = data['y']

    # Label: lowercase (dari nama folder data/template/<vowel>)
    X_a = X[y == 'a']
    X_e = X[y == 'e']
    X_i = X[y == 'i']
    X_o = X[y == 'o']
    X_u = X[y == 'u']

    print(len(X_a))

    dtw_a_template_dba = DynamicTimeWarping(X_a, mode="DBA")
    dtw_a_template_dba.fit(max_iter=5)

    dtw_a_template_dtw = DynamicTimeWarping(X_a, mode="DTW")
    dtw_a_template_dtw.fit()

    print("DTW DBA")
    print(dtw_a_template_dba.model(X_a[0])[0])
    print(dtw_a_template_dba.model(X_e[0])[0])
    print(dtw_a_template_dba.model(X_i[0])[0])
    print(dtw_a_template_dba.model(X_o[0])[0])
    print(dtw_a_template_dba.model(X_u[0])[0])

    print("\n")
    print("DTW")
    print(dtw_a_template_dtw.model(X_a[0])[0])
    print(dtw_a_template_dtw.model(X_e[0])[0])
    print(dtw_a_template_dtw.model(X_i[0])[0])
    print(dtw_a_template_dtw.model(X_o[0])[0])
    print(dtw_a_template_dtw.model(X_u[0])[0])

    dtw_a_template_dba.display_matrix(dtw_a_template_dba.delta_matrix)
    dtw_a_template_dba.display_dtw_path(dtw_a_template_dba.model(X_a[0])[1])

    dtw_a_template_dtw.display_matrix(dtw_a_template_dtw.delta_matrix)
    dtw_a_template_dtw.display_dtw_path(dtw_a_template_dtw.model(X_a[0])[1])