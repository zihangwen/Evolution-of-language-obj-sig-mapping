#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

using namespace std;

// no self loop allowed (future debug needed for node1-node1 cases)
// class Graph {
// public:
//     int popsize;
//     int *degrees, **edgelist;
//     Graph(string);
// };

// Graph::Graph(string input_name) {
//     ifstream input(input_name);

//     vector<int> out, in;
//     int node1, node2;
//     int i = 0;
//     popsize = 0;
//     while (input >> node1 >> node2)
//     {
//         popsize = (popsize < node1) ? node1: popsize;
//         popsize = (popsize < node2) ? node2: popsize;
//         out.push_back(node1);
//         in.push_back(node2);
//         ++i;
//     }
//     if (out.size() != in.size())
//         throw invalid_argument("in and out should have same length");
//     ++popsize;

//     degrees = new int[popsize];
//     int *temp = new int[popsize];

//     for (int node = 0; node < popsize; ++node) {
//         temp[node] = 0;
//         degrees[node] = 0;
//     }

//     for (auto node : out)
//         ++degrees[node];

//     for (auto node : in)
//         ++degrees[node];

//     edgelist = new int*[popsize];
//     for (int node = 0; node < popsize; ++node) {
//         edgelist[node] = new int[degrees[node]];
//     }

//     for (int j = 0; j < in.size(); ++j) {
//         int node_1 = in[j];
//         int node_2 = out[j];
//         if (node_1 == node_2) {
//             // Skip self-loops if not allowed
//             continue;
//         }
//         edgelist[node_1][temp[node_1]] = node_2;
//         edgelist[node_2][temp[node_2]] = node_1;
//         ++temp[node_1];
//         ++temp[node_2];
//     }

// //    for (int j = 0; j < popsize; ++j) {
// //        degrees[j] ++;
// //        edgelist[j][degrees[j]-1] = j;
// //    }
//     delete[] temp;
// }

class Graph {
public:
    int popsize;
    std::vector<int> degrees;
    std::vector<std::vector<int>> edgelist;
    Graph(string);
};

Graph::Graph(std::string input_name) {
    std::ifstream input(input_name);
    if (!input) {
        throw std::runtime_error("Could not open file: " + input_name);
    }

    std::vector<std::pair<int, int>> edges;
    int node1, node2;
    popsize = 0;

    // Read all edges, track max node index
    while (input >> node1 >> node2) {
        popsize = std::max(popsize, std::max(node1, node2));
        edges.emplace_back(node1, node2);
    }

    // Nodes are assumed to be 0..popsize
    ++popsize;

    // Initialize degrees and adjacency list
    degrees.assign(popsize, 0);
    edgelist.assign(popsize, std::vector<int>{});

    // Build undirected adjacency list and degrees
    for (const auto& e : edges) {
        int u = e.first;
        int v = e.second;

        if (u == v) {
            // Skip self-loops if not allowed
            continue;
        }

        edgelist[u].push_back(v);
        edgelist[v].push_back(u);

        ++degrees[u];
        ++degrees[v];
    }

    // If you want to add self-loops like in the commented-out code:
    // for (int j = 0; j < popsize; ++j) {
    //     edgelist[j].push_back(j);
    //     ++degrees[j];
    // }
}
