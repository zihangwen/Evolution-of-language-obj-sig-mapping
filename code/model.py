import numpy as np
from utilities import *
from dataclasses import dataclass
from typing import Type, List, Optional


@dataclass
class Config:
    obj : int # number of objects
    sound : int # number of sounds
    n_languages : Optional[int] = None # number of languages
    sample_times : int = 100
    self_communication : bool = False # whether self communication is allowed
    temperature : Optional[float] = 1.0


class LanguageModel:
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        self.obj, self.sound = obj, sound
        self.language_id = language_id
        self.P, self.Q = None, None
        self.initialize_language()
        self.fitness = 0
    
    def initialize_language(self) -> None:
        pass
    
    def update_language(self, sample_matrix : np.array = None,
                        temperature : Optional[float] = None) -> None:
        pass
    
    def sample_language(self, number_samples : int) -> np.array:
        sample_matrix = np.zeros([self.obj, self.sound])
        for i in range(self.obj):
            sample_matrix[i, :] = np.random.multinomial(number_samples, self.P[i,:])
        return sample_matrix


class LanguageModelSoftmax(LanguageModel):
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        super().__init__(language_id, obj, sound)
    
    def initialize_language(self) -> None:
        self.P = np.random.randn(self.obj, self.sound)
        self.Q = np.random.randn(self.sound, self.obj)
        self.update_language()
    
    def update_language(self, sample_matrix : np.array = None,
                        temperature : Optional[float] = 1.0) -> None:
        if sample_matrix is not None:
            self.P = sample_matrix
            self.Q = sample_matrix.T
        assert temperature is not None
        self.P = Softmax(self.P, temperature)
        self.Q = Softmax(self.Q, temperature)


class LanguageModelNorm(LanguageModel):
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        super().__init__(language_id, obj, sound)

    def initialize_language(self) -> None:
        self.P = np.random.uniform(0, 1, (self.obj, self.sound))
        self.Q = np.random.uniform(0, 1, (self.sound, self.obj))
        self.update_language()

    def update_language(self, sample_matrix : np.array = None,
                        temperature : Optional[float] = None) -> None:
        if sample_matrix is not None:
            self.P = sample_matrix
            self.Q = sample_matrix.T
        self.P = Normalize(self.P)
        self.Q = Normalize(self.Q)


class LanguageModelNormEPS(LanguageModel):
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        super().__init__(language_id, obj, sound)

    def initialize_language(self) -> None:
        self.P = np.random.uniform(0, 1, (self.obj, self.sound))
        self.Q = np.random.uniform(0, 1, (self.sound, self.obj))
        self.update_language()

    def update_language(self, sample_matrix : np.array = None,
                        temperature : Optional[float] = None) -> None:
        if sample_matrix is not None:
            self.P = sample_matrix
            self.Q = sample_matrix.T
        self.P = NormalizeEPS(self.P)
        self.Q = NormalizeEPS(self.Q)

