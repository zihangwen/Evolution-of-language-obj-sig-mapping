import numpy as np
from utilities import *
from dataclasses import dataclass
from typing import Type, List
import networkx as nx ##### added #####
import os ##### added #####
import matplotlib.pyplot as plt ##### added #####
from scipy.optimize import linear_sum_assignment ##### added #####
from sklearn.metrics import confusion_matrix ##### added #####


@dataclass
class Config:
    obj : int # number of objects
    sound : int # number of sounds
    n_languages : int # number of languages
    sample_times : int = 100
    self_communication : bool = False # whether self communication is allowed


class LanguageModel:
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        self.obj, self.sound = obj, sound
        self.language_id = language_id
        self.P, self.Q = None, None
        self.initialize_language()
        self.fitness = 0
    
    def initialize_language(self) -> None:
        pass
    
    def update_language(self, sample_matrix : np.array = None) -> None:
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
    
    def update_language(self, sample_matrix : np.array = None) -> None:
        if sample_matrix is not None:
            self.P = sample_matrix
            self.Q = sample_matrix.T
        self.P = Softmax(self.P)
        self.Q = Softmax(self.Q)


class LanguageModelNorm(LanguageModel):
    def __init__(self, language_id : int, obj : int, sound : int) -> None:
        super().__init__(language_id, obj, sound)

    def initialize_language(self) -> None:
        self.P = np.random.uniform(0, 1, (self.obj, self.sound))
        self.Q = np.random.uniform(0, 1, (self.sound, self.obj))
        self.update_language()

    def update_language(self, sample_matrix : np.array = None) -> None:
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

    def update_language(self, sample_matrix : np.array = None) -> None:
        if sample_matrix is not None:
            self.P = sample_matrix
            self.Q = sample_matrix.T
        self.P = NormalizeEPS(self.P)
        self.Q = NormalizeEPS(self.Q)


def directional_comm(language1 : LanguageModel, language2 : LanguageModel) -> float:
    # return np.einsum('...ij,...ji->...', P, Q)
    return np.einsum('ij,ji', language1.P, language2.Q)


# def PairPayoff(language1 : LanguageModel, language2 : LanguageModel) -> float:
#     payoff1 = np.einsum('...ij,...ji->...', language1.P, language2.Q)
#     payoff2 = np.einsum('...ij,...ji->...', language2.P, language1.Q)
#     return 0.5 * (payoff1 + payoff2)


def similar_language_check(language1: LanguageModel, language2: LanguageModel) -> bool:
    if np.allclose(language1.P, language2.P, rtol=0, atol=EPS) and np.allclose(language1.Q, language2.Q, rtol=0, atol=EPS):
        return True
    else:
        return False


class Simulation():
    def __init__(self, config : Config, language_model : Type[LanguageModel] = LanguageModel) -> None:
        self.config = config
        self.obj, self.sound = config.obj, config.sound
        self.n_languages = config.n_languages
        self.sample_times = config.sample_times
        self.self_communication = config.self_communication
        self.logger = []

        self.languages = [language_model(language_id, self.obj, self.sound) for language_id in range(self.n_languages)]
        self.language_tags = None
        # language_id1 -> language_id2 : payoff_matrix[language_id1, language_id2]

        self.init_simulation()
    
    def init_simulation(self) -> None:
        self.payoff_matrix = np.zeros((self.n_languages, self.n_languages))
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
        language_death.update_language(sample_matrix)

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
            for language_id2 in range(self.n_languages):
                language1 = self.get_language(language_id1)
                language2 = self.get_language(language_id2)
                payoff_matrix[language_id1, language_id2] = directional_comm(language1, language2)
        
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
        
        if self.language_tags is not None:
            y_pred_aligned = align_cluster_labels(self.language_tags, rename_numbers(language_tags))
            self.language_tags = y_pred_aligned
        else:
            self.language_tags = rename_numbers(language_tags)
        return len(np.unique(language_tags)), self.language_tags
    
    def get_fitness(self) -> np.array:
        return np.array([language.fitness for language in self.languages])

    def update_logger(self, i_t : int = -1) -> None:
        # num_languages = np.nan
        # if (i_t % 10) == (10 - 1):
            num_languages, language_tags = self.get_num_languages()
            fitness_vector = self.get_fitness()
            self.logger.append([i_t, fitness_vector.max(), fitness_vector.mean(), num_languages])
            
            ##### added #####
            if num_languages > 7:
                return
            missing_values = set(range(7)) - set(language_tags)
            outlier_values = set(language_tags) - set(range(7))
            for o_value in outlier_values:
                language_tags[np.where(language_tags == o_value)] = missing_values.pop()

            language_tags_dict = {i: tag for i, tag in enumerate(language_tags)}
            fitness_vector_dict = {i: np.round(fitness, 2) for i, fitness in enumerate(fitness_vector)}
            plot_two_deme_in_sim(i_t, language_tags_dict, fitness_vector_dict)
            ##### ----- #####

    def get_logger(self) -> List:
        return self.logger
    

class SimulationGraph(Simulation):
    def __init__(self, config : Config, graph_file : str, language_model : Type[LanguageModel] = LanguageModel) -> None:
        self.graph = load_G(graph_file)
        self.n_neighbors = {k: len(v) for k, v in self.graph.items()}
        config.n_languages = len(self.graph)
        super().__init__(config, language_model)
            
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
        for language_id1 in range(self.n_languages):
            for language_id2 in self.graph[language_id1]:
                language1 = self.get_language(language_id1)
                language2 = self.get_language(language_id2)
                payoff_matrix[language_id1, language_id2] = directional_comm(language1, language2)
        
        self.payoff_matrix = payoff_matrix
        
    def update_payoff_one(self, language_id : int) -> None:
        language = self.get_language(language_id)
        for language_id2 in self.graph[language_id]:
            language2 = self.get_language(language_id2)
            self.payoff_matrix[language_id, language_id2] = directional_comm(language, language2)
            self.payoff_matrix[language_id2, language_id] = directional_comm(language2, language)
    
    def assign_fitness(self) -> None:
        sum_payoff = 0.5 * (np.einsum('ij->i', self.payoff_matrix) + np.einsum('ij->j', self.payoff_matrix))
        for language_id in range(self.n_languages):
            self.get_language(language_id).fitness = sum_payoff[language_id] / self.n_neighbors[language_id]


##### added #####
def plot_two_deme_in_sim(i_t, language_tags_dict, fitness_vector_dict):
    base_path = "/home/zihangw/EvoComm/"
    output_path = os.path.join(base_path, "graphs")
    G_combined = nx.read_edgelist(os.path.join(output_path,"two_deme_G_combined_14.txt"), nodetype = int)
    G = nx.read_edgelist(os.path.join(output_path,"two_deme_G_7.txt"), nodetype = int)
    G_copy = nx.read_edgelist(os.path.join(output_path,"two_deme_G_copy_7.txt"), nodetype = int)
    deme_size = 7
    G_combined_edges = set([frozenset(edge) for edge in G_combined.edges])
    G_edges = set([frozenset(edge) for edge in G.edges])
    G_copy_edges = set([frozenset(edge) for edge in G_copy.edges])
    inter_edges = G_combined_edges - G_edges - G_copy_edges

    # save_dir = os.path.join(base_path, "results_test", "softmax", "two_deme_14", "language_id_%d.png" % (i_t))
    # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # plot_two_deme(G_combined, deme_size, inter_edges, save_dir, labels = language_tags_dict)

    save_dir = os.path.join(base_path, "results_test", "softmax", "two_deme_G_combined_14", "fitness_%d.png" % (i_t))
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plot_two_deme(G_combined, deme_size, inter_edges, save_dir, language_tags_dict, labels = fitness_vector_dict)

def plot_two_deme(G_combined, deme_size, inter_edges, save_dir, color_dicts, labels = None): ##### added
    colors_list = ["#156082", "#E97132", "#702670", "#C02739", "#1B7F9F", "#D85C41", "#92374D"]
    # Generate circular layouts for each deme
    pos_left = nx.circular_layout(range(deme_size))  # Circular layout for first deme
    pos_right = nx.circular_layout(range(deme_size,2*deme_size))  # Circular layout for second deme
    
    node_colors = {}
    for node, color_id in color_dicts.items():
        node_colors[node] = colors_list[color_id]    

    node_border_colors = {}
    # Shift left deme to x ≈ -1 and right deme to x ≈ 1
    for node in pos_left:
        pos_left[node][0] -= 1.5  # Shift left deme to x = -1
        node_border_colors[node] = 'black'
    for node in pos_right:
        # pos_right[node][0] = -pos_right[node][0]
        pos_right[node][0] += 1.5  # Shift right deme to x = 1
        node_border_colors[node] = 'black'

    # Merge positions
    pos = {**pos_left, **pos_right}

    # border cell check
    border_node = find_nodes_with_different_colored_neighbors(G_combined, color_dicts)

    for node in border_node:
        node_border_colors[node] = "#75bd2d"

    # Adjust inter-deme edge node positions slightly toward the center
    # for (n1, n2) in inter_edges:
    #     node_border_colors[n1] = "#8ED973"
    #     node_border_colors[n2] = "#8ED973"
    #     pos[n1][0] += 1  # Shift left deme nodes slightly right
    #     pos[n2][0] -= 1  # Shift right deme nodes slightly left

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(
        G_combined, pos,
        node_color=[node_colors[node] for node in G_combined.nodes],
        edgecolors=[node_border_colors[node] for node in G_combined.nodes],
        node_size=700, linewidths=2
        )
    nx.draw_networkx_edges(G_combined, pos, edge_color="black")
    nx.draw_networkx_labels(G_combined, pos, labels, font_size=10, font_color="white")
    # nx.draw(G_combined, pos, with_labels=True, node_color=list(node_colors.values()), edge_color="gray", node_size=700, font_size=10)
    plt.title("Two-Deme Graph with Inter-Deme Connections")
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()

def find_nodes_with_different_colored_neighbors(graph, colors_dict):
    nodes_with_diff_colors = []

    for node in graph.nodes:
        node_color = colors_dict.get(node)

        # Check the colors of each neighbor
        for neighbor in graph.neighbors(node):
            neighbor_color = colors_dict.get(neighbor)
            if neighbor_color != node_color:
                nodes_with_diff_colors.append(node)
                break  # Once we find a different color neighbor, no need to check further

    return nodes_with_diff_colors

##### ----- #####