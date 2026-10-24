import numpy as np


class GaussianModel:
    """Model Gaussian 1 vowel: mean + full covariance dari semua frame (slide 47, 51, 52)"""

    def __init__(self, eps: float = 0.0) -> None:
        """eps: regularisasi C + eps * I, opsional kalau C singular (default 0 -> sesuai slide)"""
        self.eps = eps

    def fit(self, samples: list[np.ndarray]) -> None:
        """list (N_i, 39) -> mean m (39,), covariance C (39, 39)"""
        # Frame pooling: list (N_i, 39) -> (N, 39)
        frames = np.vstack(samples)
        N, D = frames.shape

        # Mean: m = 1/N * sum x(i)
        self.mean = np.sum(frames, axis=0) / N

        # Covariance: C = 1/N * sum (x(i) - m)(x(i) - m)^T
        self.cov = np.zeros((D, D))
        for x in frames:
            diff = x - self.mean
            self.cov += np.outer(diff, diff)
        self.cov = self.cov / N + self.eps * np.eye(D)

        # C^-1 dan |C| -> dipakai di setiap frame_cost
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def frame_cost(self, x: np.ndarray) -> float:
        """-log P(x | m, C) = 1/2 log((2 pi)^D |C|) + 1/2 (x - m)^T C^-1 (x - m)"""
        D = len(x)
        diff = x - self.mean

        variance_term = 0.5 * np.log((2 * np.pi) ** D * self.cov_det)
        mahalanobis_term = 0.5 * (diff.T @ self.cov_inv @ diff)

        return variance_term + mahalanobis_term

    def nll(self, x: np.ndarray) -> float:
        """Total -log likelihood: (N, 39) -> sum frame_cost (log: perkalian -> penjumlahan)"""
        return float(sum(self.frame_cost(frame) for frame in x))


class GaussianRecognizer:
    """Recognizer Gaussian: 1 model per vowel -> input vs semua model -> NLL terkecil"""

    def __init__(self, eps: float = 0.0) -> None:
        self.eps = eps
        self.models = {}

    def fit(self, X: list[np.ndarray], y: list[str]) -> None:
        """Model per vowel: X (list (N, 39)), y (label) -> self.models"""
        for label in sorted(set(y)):
            samples = [x for x, lbl in zip(X, y) if lbl == label]
            model = GaussianModel(eps=self.eps)
            model.fit(samples)
            self.models[label] = model

    def predict(self, x: np.ndarray) -> tuple[str, dict[str, float]]:
        """(N, 39) -> (label, NLL per vowel)"""
        scores = {label: model.nll(x) for label, model in self.models.items()}
        return min(scores, key=scores.get), scores
