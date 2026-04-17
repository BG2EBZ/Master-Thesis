import numpy as np


class VectorizedRollingWindow:
    """Vectorized rolling mean over multiple entities without Python-level loops."""

    def __init__(self, window_steps: int, n_entities: int):
        self.window_steps = max(1, int(window_steps))
        self.n_entities = int(n_entities)
        self.reset()

    def reset(self):
        self.history = np.full((self.window_steps, self.n_entities), np.nan, dtype=np.float32)
        self.valid_history = np.zeros((self.window_steps, self.n_entities), dtype=bool)
        self.rolling_sum = np.zeros(self.n_entities, dtype=np.float32)
        self.rolling_count = np.zeros(self.n_entities, dtype=np.int32)
        self.cursor = 0

    def update(self, current_values: np.ndarray) -> np.ndarray:
        """Push a new vector and return the current rolling mean."""
        current_values = np.asarray(current_values, dtype=np.float32)
        if current_values.shape != (self.n_entities,):
            raise ValueError(
                "current_values shape must match n_entities, "
                f"got {current_values.shape} for {self.n_entities} entities"
            )

        old_valid = self.valid_history[self.cursor]
        old_values = self.history[self.cursor]
        if np.any(old_valid):
            self.rolling_sum[old_valid] -= old_values[old_valid]
            self.rolling_count[old_valid] -= 1

        new_valid = np.isfinite(current_values)
        self.history[self.cursor] = current_values
        self.valid_history[self.cursor] = new_valid
        if np.any(new_valid):
            self.rolling_sum[new_valid] += current_values[new_valid]
            self.rolling_count[new_valid] += 1

        self.cursor = (self.cursor + 1) % self.window_steps
        return self.get_mean()

    def get_mean(self) -> np.ndarray:
        """Return the current rolling mean, leaving invalid positions as NaN."""
        rolling_mean = np.full(self.n_entities, np.nan, dtype=np.float32)
        valid_mean = self.rolling_count > 0
        if np.any(valid_mean):
            rolling_mean[valid_mean] = (
                self.rolling_sum[valid_mean] / self.rolling_count[valid_mean]
            ).astype(np.float32)
        return rolling_mean
