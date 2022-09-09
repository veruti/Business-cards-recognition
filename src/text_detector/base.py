import numpy as np


class BaseDetector:
    def predict(self, image: np.array):
        raise f"Not implemented"
