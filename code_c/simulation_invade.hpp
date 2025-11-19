#pragma once

#include <string>
#include <algorithm>
#include <numeric>

#include "Utilities.hpp"
#include "graph.hpp"

using namespace std;

struct ConfigInvade {
    int n_languages = 0;
    int _log_every = 100; // NEW: logging cadence
};

// class LoggerInvade {
// public:
//     std::vector<int> iteration;
//     std::vector<double> max_fitness;
//     std::vector<double> mean_fitness;
//     std::vector<int> num_group_0;
//     std::vector<double> num_group_1;
// };

class SimulationInvade {
public:
    ConfigInvade config;
    // LoggerInvade logger;
    Graph graph;

    vector<int> groups; // 0 or 1
    vector<double> payoff_values; // init -> init, init -> invader, invader -> init, invader -> invader

    vector<vector<double>> payoff_matrix;
    vector<double> fitness_vector;

    SimulationInvade(const ConfigInvade& cfg, string graph_file);
    void initialize(vector<double> init_payoff_values);
    pair<string, int> run(int iterations);
    pair<int, int> birth_death();
    void update_language(int birth_id, int death_id);
    void update_payoff_one(int lang_id);
    void update_payoff_all();
    void assign_fitness();
    int get_num_languages();
    // void update_logger(int iter);
};

SimulationInvade::SimulationInvade(const ConfigInvade& cfg, string graph_file) : config(cfg), graph(graph_file) {
    config.n_languages = graph.popsize;
    groups.resize(config.n_languages, 0);
    payoff_matrix.resize(config.n_languages, vector<double>(config.n_languages, 0.0));
    fitness_vector.resize(config.n_languages, 0.0);
}

void SimulationInvade::initialize(vector<double> init_payoff_values) {
    // Initialize groups with all zero
    groups.assign(config.n_languages, 0);
    // random select an individual to be invader (group 1)
    int invader_id = uniform_int_sample(0, config.n_languages - 1);
    groups[invader_id] = 1;
    // cout << "Invader initialized at node " << invader_id << endl;

    // Initialize payoff values
    payoff_values = vector<double>(init_payoff_values.size());
    for (int i = 0; i < init_payoff_values.size(); ++i) {
        payoff_values[i] = init_payoff_values[i];
    }

    payoff_matrix.assign(config.n_languages, vector<double>(config.n_languages, 0.0));
    fitness_vector.assign(config.n_languages, 0.0);

    // Initial update
    update_payoff_all();
    assign_fitness();
    // update_logger(-1);

    // plot graph info
    // cout << "Graph loaded with " << config.n_languages << " nodes." << endl;
    // for (int i = 0; i < config.n_languages; ++i) {
    //     cout << "Node " << i << " (group " << groups[i] << ") neighbors: ";
    //     for (int j = 0; j < graph.degrees[i]; ++j) {
    //         cout << graph.edgelist[i][j] << " ";
    //     }
    //     cout << endl;
    //     cout << "  Payoffs: ";
    //     for (int j = 0; j < graph.degrees[i]; ++j) {
    //         int neighbor_id = graph.edgelist[i][j];
    //         cout << payoff_matrix[i][neighbor_id] << " ";
    //     }
    //     cout << endl;
    //     cout << "  Fitness: " << fitness_vector[i] << endl;
    // }
}

pair<string, int> SimulationInvade::run(int iterations = -1) {
    int iter = 0;
    if (iterations == -1) {
        iterations = std::numeric_limits<int>::max();
    }
    while (get_num_languages() > 1 && iter < iterations) {
        iter++;

        pair<int, int> bd_pair = birth_death();
        int birth_id = bd_pair.first;
        int death_id = bd_pair.second;

        // cout << "Iteration " << iter + 1 << ": Birth at " << birth_id << " (group " << groups[birth_id] << "), Death at " << death_id << " (group " << groups[death_id] << ")" << endl;
        if (groups[birth_id] != groups[death_id]) {
            update_language(birth_id, death_id);
            update_payoff_one(death_id);
            assign_fitness();
        }
        
        // update_logger(iter);
    }
    if (get_num_languages() > 1) {
        return make_pair("coexist", iter);
    } else {
        return make_pair(groups[0] == 1 ? "fix" : "lost", iter);
    }
}

pair<int, int> SimulationInvade::birth_death() {
    std::vector<double> fitness_accu = accumulate_sum(fitness_vector);
    int birth_pos = random_choice_single(fitness_accu);
    int death_idx = uniform_int_sample(0, graph.degrees[birth_pos] - 1);
    int death_pos = graph.edgelist[birth_pos][death_idx];
    return {birth_pos, death_pos};
}

void SimulationInvade::update_language(int birth_id, int death_id) {
    groups[death_id] = groups[birth_id];
}

void SimulationInvade::update_payoff_one(int lang_id) {
    for (int j = 0; j < graph.degrees[lang_id]; ++j) {
        int neighbor_id = graph.edgelist[lang_id][j];
        double payoff = payoff_values[groups[lang_id] * 2 + groups[neighbor_id]];
        payoff_matrix[lang_id][neighbor_id] = payoff;
        double payoff2 = payoff_values[groups[neighbor_id] * 2 + groups[lang_id]];
        payoff_matrix[neighbor_id][lang_id] = payoff2; // symmetric
    }
}

void SimulationInvade::update_payoff_all() {
    for (int i = 0; i < config.n_languages; ++i) {
        for (int j = 0; j < graph.degrees[i]; ++j) {
            int neighbor_id = graph.edgelist[i][j];
            double payoff = payoff_values[groups[i] * 2 + groups[neighbor_id]];
            payoff_matrix[i][neighbor_id] = payoff;
        }
    }
}

void SimulationInvade::assign_fitness() {
    for (int i = 0; i < config.n_languages; ++i) {
        double total_payoff = 0.0;
        for (int j = 0; j < graph.degrees[i]; ++j) {
            int neighbor_id = graph.edgelist[i][j];
            total_payoff += payoff_matrix[i][neighbor_id] + payoff_matrix[neighbor_id][i];
        }
        fitness_vector[i] = total_payoff / double(graph.degrees[i]) / 2;
    }
    // cout << "Fitness vector: ";
    // for (int i = 0; i < config.n_languages; ++i) {
    //     cout << fitness_vector[i] << " ";
    // }
    // cout << endl;
}

int SimulationInvade::get_num_languages() {
    set<int> unique_groups(groups.begin(), groups.end());
    return unique_groups.size();
}