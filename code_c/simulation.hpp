#pragma once

#include <string>
#include <algorithm>
#include <numeric>

#include "Utilities.hpp"
#include "language.hpp"
#include "graph.hpp"

using namespace std;

struct Config {
    int obj;
    int sound;
    int n_languages = 0;
    int sample_times = 10;
    bool self_communication = true;
    double temperature = 1.0;
    int _log_every = 100; // NEW: logging cadence
};

class Logger {
public:
    std::vector<int> iteration;
    std::vector<double> max_fitness;
    std::vector<double> mean_fitness;
    std::vector<int> num_languages;
    std::vector<double> max_self_payoff;
    std::vector<double> mean_self_payoff;
};

class SimulationGraph {
public:
    Config config;
    Logger logger;
    Graph graph;

    vector<LanguageModel> languages;
    // vector<int> language_tags;

    vector<vector<double>> payoff_matrix;
    vector<double> self_payoff_vector;

    vector<double> fitness_vector;

    SimulationGraph(const Config& cfg, string graph_file);
    void initialize();
    void run(int iterations);
    pair<int, int> birth_death();
    void update_language(int birth_id, int death_id);
    void update_payoff_one(int lang_id);
    void update_payoff_all();
    void assign_fitness();
    int get_num_languages();
    void update_logger(int iter);
};

SimulationGraph::SimulationGraph(const Config& cfg, string graph_file) : config(cfg), graph(graph_file) {
    config.n_languages = graph.popsize;
    // cout << "Graph loaded with " << config.n_languages << " nodes." << endl;
    // language_tags.resize(config.n_languages, 0);
    payoff_matrix.resize(config.n_languages, vector<double>(config.n_languages, 0.0));
    self_payoff_vector.resize(config.n_languages, 0.0);
    fitness_vector.resize(config.n_languages, 0.0);
}

void SimulationGraph::initialize() {  
    // Initialize languages
    for (int i = 0; i < config.n_languages; ++i) {
        languages.emplace_back(i, config.obj, config.sound);
        // language_tags[i] = i;
    }

    payoff_matrix.assign(config.n_languages, vector<double>(config.n_languages, 0.0));
    self_payoff_vector.assign(config.n_languages, 0.0);
    fitness_vector.assign(config.n_languages, 0.0);

    update_payoff_all();
    assign_fitness();
    update_logger(-1);
}

void SimulationGraph::run(int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Simulation logic here
        // b-d process
        auto [birth_pos, death_pos] = birth_death();
        // update language
        update_language(birth_pos, death_pos);
        // update payoff
        update_payoff_one(death_pos);
        // self.update_payoff_all()
        assign_fitness();
        update_logger(iter);

        if (logger.num_languages.back() == 1) {
            // All languages have converged
            break;
        }
    }
}

pair<int, int> SimulationGraph::birth_death() {
    // Implement birth-death process based on graph structure
    std::vector<double> fitness_accu = accumulate_sum(fitness_vector);
    int birth_pos = random_choice_single(fitness_accu);
    int death_idx = uniform_int_sample(0, graph.degrees[birth_pos] - 1);
    int death_pos = graph.edgelist[birth_pos][death_idx];
    return {birth_pos, death_pos};
}

void SimulationGraph::update_language(int birth_id, int death_id) {
    SimpleMatrix sample_matrix = languages[birth_id].Samplelanguage(config.sample_times);
    languages[death_id].UpdateLanguage(sample_matrix, languages[birth_id].language_id);
}

void SimulationGraph::update_payoff_one(int lang_id) {
    for (int j = 0; j < graph.degrees[lang_id]; ++j) {
        int neighbor_id = graph.edgelist[lang_id][j];
        double payoff = MatrixDotProduct(languages[lang_id].P, languages[neighbor_id].QT);
        payoff_matrix[lang_id][neighbor_id] = payoff;
        double payoff2 = MatrixDotProduct(languages[neighbor_id].P, languages[lang_id].QT);
        payoff_matrix[neighbor_id][lang_id] = payoff2; // symmetric
    }
    self_payoff_vector[lang_id] = MatrixDotProduct(languages[lang_id].P, languages[lang_id].QT);
}

void SimulationGraph::update_payoff_all() {
    for (int i = 0; i < config.n_languages; ++i) {
        for (int j = 0; j < graph.degrees[i]; ++j) {
            int neighbor_id = graph.edgelist[i][j];
            double payoff = MatrixDotProduct(languages[i].P, languages[neighbor_id].QT);
            payoff_matrix[i][neighbor_id] = payoff;
        }
        self_payoff_vector[i] = MatrixDotProduct(languages[i].P, languages[i].QT);
    }
}

void SimulationGraph::assign_fitness() {
    for (int i = 0; i < config.n_languages; ++i) {
        double total_payoff = 0.0;
        for (int j = 0; j < graph.degrees[i]; ++j) {
            int neighbor_id = graph.edgelist[i][j];
            total_payoff += payoff_matrix[i][neighbor_id];
        }
        fitness_vector[i] = total_payoff / double(graph.degrees[i]);
        languages[i].fitness = fitness_vector[i];
    }
}

int SimulationGraph::get_num_languages() {
    // use LanguageEqual to count unique languages
    int count = 0;
    vector<bool> visited(config.n_languages, false);
    for (int i = 0; i < config.n_languages; ++i) {
        if (visited[i]) continue;
        ++count;
        for (int j = i + 1; j < config.n_languages; ++j) {
            if (!visited[j] && LanguageEqual(languages[i], languages[j])) {
                visited[j] = true;
            }
        }
    }
    return count;
}

void SimulationGraph::update_logger(int iter) {
    if (iter != -1 && (iter % config._log_every) != (config._log_every - 1)) return;
    double max_fit = *std::max_element(fitness_vector.begin(), fitness_vector.end());
    double mean_fit = std::accumulate(fitness_vector.begin(), fitness_vector.end(), 0.0) / double(fitness_vector.size());
    double max_self = *std::max_element(self_payoff_vector.begin(), self_payoff_vector.end());
    double mean_self = std::accumulate(self_payoff_vector.begin(), self_payoff_vector.end(), 0.0) / double(self_payoff_vector.size());

    logger.iteration.push_back(iter);
    logger.max_fitness.push_back(max_fit);
    logger.mean_fitness.push_back(mean_fit);
    logger.num_languages.push_back(get_num_languages());
    logger.max_self_payoff.push_back(max_self);
    logger.mean_self_payoff.push_back(mean_self);
}