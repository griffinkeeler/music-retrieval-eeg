"""Baseline model definitions used for comparison with the EEG-to-audio model."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeEEGToAudio:
    """Map flattened EEG windows to vector audio embeddings with ridge regression.

    EEG features and audio targets are standardized using training-set statistics
    only. Sequence-valued audio features must be pooled to ``[N, D]`` by the
    caller so that the baseline has one target vector per EEG window.

    Args:
        alpha: Non-negative ridge regularization strength.
        solver: A solver accepted by :class:`sklearn.linear_model.Ridge`.
    """

    def __init__(self, alpha=1.0, solver="auto"):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        self.alpha = float(alpha)
        self.solver = str(solver)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # Scaling removes the intercept. Disabling it avoids another centering
        # pass over the large, flattened EEG design matrix.
        self.model = Ridge(
            alpha=self.alpha,
            solver=self.solver,
            fit_intercept=False,
            copy_X=False,
        )
        self.eeg_window_shape_ = None

    def _flatten_eeg(self, eeg, *, fitting):
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim < 2:
            raise ValueError(
                "EEG input must have shape [N, ...] with at least one feature axis."
            )
        if len(eeg) == 0:
            raise ValueError("EEG input must contain at least one window.")

        window_shape = tuple(eeg.shape[1:])
        if fitting:
            self.eeg_window_shape_ = window_shape
        elif self.eeg_window_shape_ is None:
            raise RuntimeError("The ridge baseline must be fit before prediction.")
        elif window_shape != self.eeg_window_shape_:
            raise ValueError(
                "EEG window shape does not match training data: "
                f"expected {self.eeg_window_shape_}, got {window_shape}."
            )

        return eeg.reshape(len(eeg), -1)

    def fit(self, eeg_train, audio_train):
        """Fit the baseline and return ``self``."""
        X = self._flatten_eeg(eeg_train, fitting=True)
        Y = np.asarray(audio_train, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(
                "Audio targets must have shape [N, D]. Pool temporal audio "
                "embeddings before fitting the ridge baseline."
            )
        if len(Y) != len(X):
            raise ValueError(
                "EEG and audio training arrays must contain the same number of windows."
            )

        Xs = self.x_scaler.fit_transform(X)
        Ys = self.y_scaler.fit_transform(Y)
        self.model.fit(Xs, Ys)
        return self

    def predict(self, eeg):
        """Predict vector audio embeddings for EEG windows.

        Args:
            eeg: EEG windows with the same non-batch shape used for fitting.

        Returns:
            Predicted audio embeddings with shape [N, D].
        """
        X = self._flatten_eeg(eeg, fitting=False)
        Xs = self.x_scaler.transform(X)
        pred = self.model.predict(Xs)
        return self.y_scaler.inverse_transform(pred)
