#include <iostream>
#include <iomanip>
#include <fstream>
#include "simulation.hpp"
#include <filesystem>

using namespace std;

// void print_logger(const Logger& logger) {
//     cout << "\n=== Logger Contents ===" << endl;
//     cout << setw(10) << "Iteration" 
//          << setw(15) << "MaxFitness" 
//          << setw(15) << "MeanFitness" 
//          << setw(15) << "NumLanguages"
//          << setw(18) << "MaxSelfPayoff"
//          << setw(18) << "MeanSelfPayoff" << endl;
//     cout << string(91, '-') << endl;
    
//     for (size_t i = 0; i < logger.iteration.size(); ++i) {
//         cout << setw(10) << logger.iteration[i]
//              << setw(15) << fixed << setprecision(4) << logger.max_fitness[i]
//              << setw(15) << fixed << setprecision(4) << logger.mean_fitness[i]
//              << setw(15) << logger.num_languages[i]
//              << setw(18) << fixed << setprecision(4) << logger.max_self_payoff[i]
//              << setw(18) << fixed << setprecision(4) << logger.mean_self_payoff[i]
//              << endl;
//     }
// }

int main(int argc, char **argv) {
    int num_objects = atoi(argv[1]);
    int num_sounds = atoi(argv[2]);
    // string model_name = argv[3];
    string graph_path = argv[4];
    int num_runs = atoi(argv[5]);
    string out_path_base = argv[6];
    int sample_times = atoi(argv[7]);
    // float temperature = atof(argv[8]);
    int n_trial = atoi(argv[9]);

    string model_name = "norm";
    float temperature = 0.0;
    int _log_every = 100;
    int num_trials = 1000;

    Config cfg = {
        num_objects,
        num_sounds,
        0, // n_languages will be set in SimulationGraph constructor
        sample_times,
        false, // self_communication, not used for SimulationGraph
        temperature, // temperature, not used for SimulationGraph
        _log_every
    };

    for (int i_trial; i_trial < num_trials; ++i_trial) {
        // cout << "Running trial " << i_trial + 1 << " / " << num_trials << endl;

        SimulationGraph sim(cfg, graph_path);
        sim.initialize();
        sim.run(num_runs);

        // print_logger(sim.logger);
        // Output results
        string graph_base = graph_path.substr(0, graph_path.find_last_of("/\\"));
        string graph_name = graph_path.substr(graph_path.find_last_of("/\\") + 1, graph_path.find_last_of(".") - graph_path.find_last_of("/\\") - 1);
        string out_path = out_path_base + "/" + model_name + "/" + graph_base + "/" + graph_name;

        std::filesystem::create_directories(
            out_path_base + "/" + model_name + "/" + graph_base + "/" + graph_name
        );

        std::ostringstream temp_ss;
        temp_ss.setf(std::ios::fixed);
        temp_ss.precision(1);
        temp_ss << temperature;
        string temp_str = temp_ss.str();

        out_path = out_path + "/" + "st_" + to_string(sample_times) + "_temp_" + temp_str + "_" + to_string(n_trial*num_trials+i_trial) + ".txt";
        
        ofstream file;
        file.open(out_path, ios::app);
        file << "iteration\tmax_fitness\tmean_fitness\tnum_languages\tmax_self_payoff\tmean_self_payoff\n";
        for (size_t i = 0; i < sim.logger.iteration.size(); ++i) {
            file << sim.logger.iteration[i] << "\t"
                 << sim.logger.max_fitness[i] << "\t"
                 << sim.logger.mean_fitness[i] << "\t"
                 << sim.logger.num_languages[i] << "\t"
                 << sim.logger.max_self_payoff[i] << "\t"
                 << sim.logger.mean_self_payoff[i] << "\n";
        }
        file.close();
    }
    
    return 0;
}

// g++ -static -std=c++17 run.cpp -o run



// int main(int argc, char **argv) {
//     // cout << "seed:" << seed << endl;
//     ofstream file;
//     string input_name = argv[1];
//     string output_name = argv[2];

//     int runs = atoi(argv[3]);

//     // Parameters
//     int N;
//     double s1, s2, s_mean, s_std;

//     std::vector<double> s_mean_list;
//     s_mean_list.push_back(atof(argv[4]));

//     std::vector<double> s_std_list;
//     for(int i = 5; i < argc; i++) {
//         s_std_list.push_back(atof(argv[i]));
//     }
    
//     file.open(output_name, ios::app);
//     file << "# N" << "\t";
//     file << "s_mean" << "\t";
//     file << "s_std" << "\t";
//     file << "runs" << "\t";
//     file << "counts" << "\t";
//     file << "time" << "\t";
//     file << "seed=" << seed << endl;
//     // file.close();

//     Model trial(input_name);

//     N = trial.N;
//     for (int i_s_mean = 0; i_s_mean < s_mean_list.size(); i_s_mean++) {
//         for (int i_s_std = 0; i_s_std < s_std_list.size(); i_s_std++) {
//             s_mean = s_mean_list[i_s_mean];
//             s_std = s_std_list[i_s_std];
//             s1 = s_mean - s_std;
//             s2 = s_mean + s_std;
//             if (s1 <= -1) {
//                 continue;
//             }
//             // Run
//             trial.Simulation(s1, s2, runs);
//             double ave_gen = trial.times;
//             double counts = trial.counts;
//             // int times = trial.times;
//             // double ave_gen = 0;
//             // if (counts != 0) {
//             //     ave_gen = times / counts;
//             // }
//             // file.open(output_name, ios::app);
//             file << N << "\t";
//             // file << s1 << "\t";
//             // file << s2 << "\t";
//             file << s_mean << "\t";
//             file << s_std << "\t";
//             file << runs << "\t";
//             file << counts << "\t";
//             file << ave_gen << endl;
//             // file.close();
//         }
//     }
//     file.close();
// //    std::cout << "Hello World!" << std::endl;
// }

