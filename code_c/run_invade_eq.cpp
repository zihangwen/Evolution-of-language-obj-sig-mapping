#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

#include "simulation_invade.hpp"
#include "language.hpp"

using namespace std;

int main(int argc, char **argv) {
    int num_objects = atoi(argv[1]);
    int num_sounds = atoi(argv[2]);
    string graph_path = argv[3];
    int num_runs = atoi(argv[4]);
    string out_path_base = argv[5];
    int n_trial = atoi(argv[6]);

    int max_iterations = -1;
    int _log_every = 100; // not used currently
    ConfigInvade cfg = {
        0, // n_languages will be set in SimulationInvade constructor
        _log_every // _log_every
    };

    SimpleMatrix init_P(num_objects, num_sounds, false, false);
    // [[1., 0., 0., 0., 0.],
    // [0., 1., 0., 0., 0.],
    // [0., 0., 1., 0., 0.],
    // [0., 1., 0., 0., 0.],
    // [0., 0., 0., 1., 0.]]
    // objects 1 and 3 share sound 1; objects 0, 2, 4 are unique
    // self-payoff: 1.0 + 0.5 + 1.0 + 0.5 + 1.0 = 4.0
    init_P.data[0][0] = 1.0;
    init_P.data[1][1] = 1.0;
    init_P.data[2][2] = 1.0;
    init_P.data[3][1] = 1.0;
    init_P.data[4][3] = 1.0;
    SimpleMatrix init_Q = init_P.Transpose();
    init_Q.Normalize();
    SimpleMatrix init_QT = init_Q.Transpose();

    SimpleMatrix invade_P(num_objects, num_sounds, false, false);
    // [[1., 0., 0., 0., 0.],
    // [0., 1., 0., 0., 0.],
    // [0., 0., 0., 1., 0.],
    // [0., 0., 1., 0., 0.],
    // [0., 0., 1., 0., 0.]]
    // objects 3 and 4 share sound 2; objects 0, 1, 2 are unique
    // self-payoff: 1.0 + 1.0 + 1.0 + 0.5 + 0.5 = 4.0
    // objects 0,1 share sounds with init -> cross-payoffs: init->invader=2.0, invader->init=1.5
    invade_P.data[0][0] = 1.0;
    invade_P.data[1][1] = 1.0;
    invade_P.data[2][3] = 1.0;
    invade_P.data[3][2] = 1.0;
    invade_P.data[4][2] = 1.0;
    SimpleMatrix invade_Q = invade_P.Transpose();
    invade_Q.Normalize();
    SimpleMatrix invade_QT = invade_Q.Transpose();

    vector<double> init_payoff_values(4, 0.0);
    // init -> init, init -> invader, invader -> init, invader -> invader
    init_payoff_values[0] = MatrixDotProduct(init_P, init_QT);
    init_payoff_values[1] = MatrixDotProduct(init_P, invade_QT);
    init_payoff_values[2] = MatrixDotProduct(invade_P, init_QT);
    init_payoff_values[3] = MatrixDotProduct(invade_P, invade_QT);

    cout << "Payoff values: " << endl;
    cout << "init -> init: " << init_payoff_values[0] << endl;
    cout << "init -> invader: " << init_payoff_values[1] << endl;
    cout << "invader -> init: " << init_payoff_values[2] << endl;
    cout << "invader -> invader: " << init_payoff_values[3] << endl;

    string graph_base = graph_path.substr(0, graph_path.find_last_of("/\\"));
    string graph_name = graph_path.substr(graph_path.find_last_of("/\\") + 1, graph_path.find_last_of(".") - graph_path.find_last_of("/\\") - 1);
    string out_path = out_path_base + "/" + graph_base + "/" + graph_name;

    std::filesystem::create_directories(out_path);

    float fixation_time = 0;
    int co_existence_count = 0;
    int fixation_count = 0;
    for (int i_trial = 0; i_trial < num_runs; ++i_trial) {
        SimulationInvade sim(cfg, graph_path);

        sim.initialize(init_payoff_values);

        pair<string, int> result = sim.run(max_iterations);
        if (result.first == "fix") {
            fixation_time = fixation_time * (double(fixation_count) / double(fixation_count + 1)) + double(result.second) / double(fixation_count + 1);
            fixation_count += 1;
        } else if (result.first == "coexist") {
            co_existence_count += 1;
        }
    }

    string out_file = out_path + "/" + to_string(n_trial) + ".txt";
    ofstream file;
    file.open(out_file, ios::app);
    file << "# graph_name:\t" << "num_trials\t" << "fixation_count\t" << "fixation_time\t" << "co_existence_count" << endl;
    file << graph_name << "\t"
         << num_runs << "\t"
         << fixation_count << "\t"
         << fixation_time << "\t"
         << co_existence_count << endl;
    file.close();
}

// g++ -static -std=c++17 run_invade_eq.cpp -o run_invade_eq