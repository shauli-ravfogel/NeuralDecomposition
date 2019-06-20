import numpy as np


class SyntacticExtractor(object):
    def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SiameseSyntacticExtractor(SyntacticExtractor):
    def __init__(self): return NotImplementedError

    def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CCASyntacticExtractor(SyntacticExtractor):
    def __init__(self): return NotImplementedError

    def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError