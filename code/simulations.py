import numpy as np
from utilities import *
from dataclasses import dataclass
from typing import Type, List, Optional
from model import LanguageModel, Config, directional_comm, similar_language_check


class Simulation():
    def __init__(self, config : Config) -> None:
        self.config = config
        self.obj, self.sound = config.obj, config.sound
        self.n_languages = config.n_languages
        self.sample_times = config.sample_times
        self.self_communication = config.self_communication
        self.temperature = config.temperature
        
    def initialize(self, language_model : Type[LanguageModel] = LanguageModel) -> None:
        self.logger = []
        self.languages = [language_model(language_id, self.obj, self.sound) for language_id in range(self.n_languages)]
        for language in self.languages:
            language.initialize_language()
        # language_id1 -> language_id2 : payoff_matrix[language_id1, language_id2]    
        self.payoff_matrix = np.zeros((self.n_languages, self.n_languages))
        self.self_payoff_vector = np.zeros(self.n_languages)
        self.update_payoff_all()
        self.assign_fitness()
        self.update_logger(-1)

    def run(self, iteration = 100) -> None:
        for i_t in range(iteration):
            # print(i_t)
            # b-d process
            birth_id, death_id = self.birth_death()
            # update language
            self.update_language(birth_id, death_id)
            # update payoff
            self.update_payoff_one(death_id)
            # self.update_payoff_all()
            self.assign_fitness()
            self.update_logger(i_t)

    def update_language(self, birth_id : int, death_id : int) -> None:
        language_birth = self.get_language(birth_id)
        language_death = self.get_language(death_id)
        sample_matrix = language_birth.sample_language(self.sample_times)
        language_death.update_language(sample_matrix, self.temperature)

    def birth_death(self) -> None:
        fitness_vector = self.get_fitness()
        fitness_vector = fitness_vector / fitness_vector.sum()
        birth_id = np.random.multinomial(1, fitness_vector).argmax()
        # no self-loop
        while True:
            death_id = np.random.multinomial(1, [1/self.n_languages]*self.n_languages).argmax()
            if death_id != birth_id:
                break
        # birth_id = np.random.choice(self.n_languages, p = fitness_vector)
        # death_id = np.random.choice(self.n_languages)
        return birth_id, death_id
        
    def update_payoff_all(self) -> None:
        payoff_matrix = np.zeros((self.n_languages, self.n_languages))
        for language_id1 in range(self.n_languages):
            language1 = self.get_language(language_id1)
            for language_id2 in range(self.n_languages):
                language2 = self.get_language(language_id2)
                payoff_matrix[language_id1, language_id2] = directional_comm(language1, language2)
        
        self.self_payoff_vector = payoff_matrix.diagonal().copy()

        # no self communication
        if not self.self_communication:
            np.fill_diagonal(payoff_matrix, 0)
        
        self.payoff_matrix = payoff_matrix

    def update_payoff_one(self, language_id : int) -> None:
        language = self.get_language(language_id)
        for language_id2 in range(self.n_languages):
            language2 = self.get_language(language_id2)
            self.payoff_matrix[language_id, language_id2] = directional_comm(language, language2)
            self.payoff_matrix[language_id2, language_id] = directional_comm(language2, language)
        
        self.self_payoff_vector[language_id] = self.payoff_matrix[language_id, language_id].copy()

        # no self communication
        if not self.self_communication:
            self.payoff_matrix[language_id, language_id] = 0
    
    def assign_fitness(self) -> None:
        fitness_matrix = 0.5 * (np.einsum('ij->i', self.payoff_matrix) + np.einsum('ij->j', self.payoff_matrix)) / (self.n_languages - 1)
        for language_id in range(self.n_languages):
            self.get_language(language_id).fitness = fitness_matrix[language_id]

    def get_language(self, language_id : int) -> LanguageModel:
        return self.languages[language_id]

    def get_languages(self) -> List[LanguageModel]:
        return self.languages
    
    def get_num_languages(self) -> int:
        language_tags = np.arange(self.n_languages)
        for language_id in range(1, self.n_languages):
            for language_id2 in range(language_id):
                if similar_language_check(self.get_language(language_id), self.get_language(language_id2)):
                    language_tags[language_id] = language_tags[language_id2]
                    break
        return len(np.unique(language_tags))
    
    def get_fitness(self) -> np.array:
        return np.array([language.fitness for language in self.languages])

    def update_logger(self, i_t : int = -1) -> None:
        # num_languages = np.nan
        if (i_t % 100) == (100 - 1):
            num_languages = self.get_num_languages()
            fitness_vector = self.get_fitness()
            self.logger.append([i_t, fitness_vector.max(), fitness_vector.mean(), num_languages])

    def get_logger(self) -> List:
        return self.logger
    

class SimulationGraph(Simulation):
    def __init__(self, config : Config, graph_file : str) -> None:
        self.graph = load_G(graph_file)
        self.n_neighbors = {k: len(v) for k, v in self.graph.items()}
        config.n_languages = len(self.graph)
        super().__init__(config)
            
    def birth_death(self) -> None:
        fitness_vector = self.get_fitness()
        fitness_vector = fitness_vector / fitness_vector.sum()
        birth_id = np.random.multinomial(1, fitness_vector).argmax()
        # birth_id = np.random.choice(self.n_languages, p = fitness_vector)
        # death_id = self.graph[birth_id][np.random.multinomial(1, [1/self.n_neighbors[birth_id]]*self.n_neighbors[birth_id]).argmax()]
        death_id = np.random.choice(self.graph[birth_id])

        return birth_id, death_id

    def update_payoff_all(self) -> None:
        payoff_matrix = np.zeros((self.n_languages, self.n_languages))
        self_payoff_vector = np.zeros(self.n_languages)
        for language_id1 in range(self.n_languages):
            language1 = self.get_language(language_id1)
            self_payoff_vector[language_id1] = directional_comm(language1, language1)
            for language_id2 in self.graph[language_id1]:
                language2 = self.get_language(language_id2)
                payoff_matrix[language_id1, language_id2] = directional_comm(language1, language2)
        
        self.self_payoff_vector = self_payoff_vector
        self.payoff_matrix = payoff_matrix
        
    def update_payoff_one(self, language_id : int) -> None:
        language = self.get_language(language_id)
        self.self_payoff_vector[language_id] = directional_comm(language, language)
        for language_id2 in self.graph[language_id]:
            language2 = self.get_language(language_id2)
            self.payoff_matrix[language_id, language_id2] = directional_comm(language, language2)
            self.payoff_matrix[language_id2, language_id] = directional_comm(language2, language)
    
    def assign_fitness(self) -> None:
        sum_payoff = 0.5 * (np.einsum('ij->i', self.payoff_matrix) + np.einsum('ij->j', self.payoff_matrix))
        for language_id in range(self.n_languages):
            self.get_language(language_id).fitness = sum_payoff[language_id] / self.n_neighbors[language_id]


class SimulationGraphInvade(SimulationGraph):
    def __init__(self, config : Config, graph_file : str, language_model : Type[LanguageModel] = LanguageModel) -> None:
        super().__init__(config, graph_file)
        self.groups = list(range(self.n_languages))

    def initialize(
            self,
            language_model : Type[LanguageModel] = LanguageModel,
            lang_init : Optional[LanguageModel] = None,
            lang_invade : Optional[LanguageModel] = None
    ) -> None:
        self.logger = []

        self.languages = [language_model(language_id, self.obj, self.sound) for language_id in range(self.n_languages)]
        if lang_init is not None:
            for language in self.languages:
                language.initialize_language(lang_init)
        else:
            for language in self.languages:
                language.initialize_language()
        
        if lang_invade is not None:
            invade_pos = np.random.randint(self.n_languages)
            self.get_language(invade_pos).update_language(lang_invade)
        
        self.groups = [language.group_id for language in self.languages]

        # language_id1 -> language_id2 : payoff_matrix[language_id1, language_id2]    
        self.payoff_matrix = np.zeros((self.n_languages, self.n_languages))
        self.update_payoff_all()
        self.assign_fitness()
        # self.update_logger(-1)
    
    def update_language(self, birth_id : int, death_id : int) -> None:
        language_birth = self.get_language(birth_id)
        language_death = self.get_language(death_id)
        if language_birth.group_id == language_death.group_id:
            return
        language_death.update_language(language_birth)
        self.groups[death_id] = language_death.group_id
        self.update_payoff_one(death_id)
        # self.update_payoff_all()
        self.assign_fitness()
    
    def run(self, iteration = None) -> None:
        i_t = 0
        iteration = iteration if iteration is not None else np.inf
        while self.unique_groups.size > 1 and i_t < iteration:
            i_t += 1
            # b-d process
            birth_id, death_id = self.birth_death()
            if self.get_language(birth_id).group_id == self.get_language(death_id).group_id:
                continue
            # update language
            self.update_language(birth_id, death_id)
            # update payoff
            # self.update_logger(i_t)
        
        if self.unique_groups.size > 1:
            result = "coexist"
        elif self.unique_groups.item() == 1:
            result = "fix"
        else:
            result = "lost"

        return result, i_t
    
    @property
    def unique_groups(self) -> int:
        return np.unique(self.groups)
        
# class Simulation:
#     def __init__(self, config : Config, graph_file : str) -> None:
#         self.config = config
#         self.obj, self.sound = config.obj, config.sound
#         self.n_languages = config.n_languages
#         self.sample_times = config.sample_times
#         self.self_communication = config.self_communication
#         self.graph = load_G(graph_file)
#         self.n_neighbors = {k: len(v) for k, v in self.graph.items()}
#         self.logger = []

#         self.languages = [LanguageModel(language_id, self.obj, self.sound) for language_id in range(self.n_languages)]
#         # language_id1 -> language_id2 : payoff_matrix[language_id1, language_id2]
#         self.payoff_matrix = np.zeros((self.n_languages, self.n_languages))
#         self.update_payoff_all()
#         self.assign_fitness()
#         self.update_logger(-1)

#     def run(self, iteration = 100) -> None:
#         for i_t in range(iteration):
#             # b-d process
#             birth_id, death_id = self.birth_death()
#             # update language
#             language_birth = self.get_language(birth_id)
#             language_death = self.get_language(death_id)
#             sample_matrix = language_birth.sample_language(self.sample_times)
#             language_death.update_language(sample_matrix)
#             # update payoff
#             self.update_payoff_one(death_id)
#             self.assign_fitness()
#             self.update_logger(i_t)
            
#     def birth_death(self) -> None:
#         fitness_vector = self.get_fitness()
#         fitness_vector = fitness_vector / np.sum(fitness_vector)
#         birth_id = np.random.multinomial(1, fitness_vector).argmax()
#         # birth_id = np.random.choice(self.n_languages, p = fitness_vector)
#         # death_id = self.graph[birth_id][np.random.multinomial(1, [1/self.n_neighbors[birth_id]]*self.n_neighbors[birth_id]).argmax()]
#         death_id = np.random.choice(self.graph[birth_id])

#         return birth_id, death_id

#     def update_payoff_all(self) -> None:
#         payoff_matrix = np.zeros((self.n_languages, self.n_languages))
#         for language_id1 in range(self.n_languages):
#             for language_id2 in self.graph[language_id1]:
#                 language1 = self.get_language(language_id1)
#                 language2 = self.get_language(language_id2)
#                 payoff_matrix[language_id1, language_id2] = directional_comm(language1, language2)
        
#         self.payoff_matrix = payoff_matrix

#     def update_payoff_one(self, language_id : int) -> None:
#         language = self.get_language(language_id)
#         for language_id2 in self.graph[language_id]:
#             language2 = self.get_language(language_id2)
#             self.payoff_matrix[language_id, language_id2] = directional_comm(language, language2)
#             self.payoff_matrix[language_id2, language_id] = directional_comm(language2, language)
    
#     def assign_fitness(self) -> None:
#         sum_payoff = 0.5 * (np.einsum('ij->i', self.payoff_matrix) + np.einsum('ij->j', self.payoff_matrix))
#         for language_id in range(self.n_languages):
#             self.get_language(language_id).fitness = sum_payoff[language_id] / self.n_neighbors[language_id]

#     def get_language(self, language_id : int) -> LanguageModel:
#         return self.languages[language_id]

#     def get_languages(self) -> list[LanguageModel]:
#         return self.languages
    
#     def get_fitness(self) -> np.array:
#         return np.array([language.fitness for language in self.languages])

#     def get_num_languages(self) -> int:
#         language_tags = np.arange(self.n_languages)
#         for language_id in range(1, self.n_languages):
#             for language_id2 in range(language_id):
#                 if similar_language_check(self.get_language(language_id), self.get_language(language_id2)):
#                     language_tags[language_id] = language_tags[language_id2]
#                     break
#         return len(np.unique(language_tags))

#     def update_logger(self, i_t : int = -1) -> None:
#         num_languages = self.get_num_languages()
#         fitness_vector = self.get_fitness()
#         self.logger.append([i_t, np.max(fitness_vector), np.mean(fitness_vector), num_languages])
#         # print("iteration: %d, max fitness: %.3f, mean fitness: %.3f" % (*self.logger[-1],))

#     def get_logger(self) -> list:
#         return self.logger
