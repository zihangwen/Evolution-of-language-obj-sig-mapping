#pragma once
#include "Utilities.hpp"

using namespace std;

class SimpleMatrix {
public:
    int rows;
    int cols;
    vector<vector<double>> data;
    SimpleMatrix(int r, int c, bool rand=false, bool norm=false);
    void Normalize();
    SimpleMatrix Transpose();
};

SimpleMatrix::SimpleMatrix(int r, int c, bool rand, bool norm) : rows(r), cols(c) {
    data.resize(r, vector<double>(c, 0.0));
    if (rand) 
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                data[i][j] = uniform_sample(0.0, 1.0);
            }
        }
    
    if (norm) Normalize();
}

void SimpleMatrix::Normalize() {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += data[i][j];
        }
        if (sum == 0.0) {
            double u = 1.0 / double(cols);
            for (int j = 0; j < cols; j++) {
                data[i][j] = u;
            }
        } else {
            double inv = 1.0 / sum;
            for (int j = 0; j < cols; j++) {
                data[i][j] *= inv;
            }
        }
    }
}

SimpleMatrix SimpleMatrix::Transpose() {
    SimpleMatrix t_matrix(cols, rows, false, false);  // Note: dimensions are swapped
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            t_matrix.data[j][i] = data[i][j];  // Swap indices
        }
    }
    return t_matrix;
}

SimpleMatrix MatrixMultiply(const SimpleMatrix& A, const SimpleMatrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }
    SimpleMatrix C(A.rows, B.cols, false, false);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            C.data[i][j] = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                C.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }
    return C;
}

float MatrixDotProduct(const SimpleMatrix& A, const SimpleMatrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for dot product.");
    }
    float dot_product = 0.0;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            dot_product += A.data[i][j] * B.data[i][j];
        }
    }
    return dot_product;
}

bool MatrixEqual(const SimpleMatrix& A, const SimpleMatrix& B, double tol=1e-6) {
    if (A.rows != B.rows || A.cols != B.cols) {
        return false;
    }
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            if (std::abs(A.data[i][j] - B.data[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

class LanguageModel {
public:
    int obj;
    int sound;
    int language_id;
    double fitness;
    SimpleMatrix P; // obj x sound
    SimpleMatrix Q; // sound x obj (so QT is obj x sound)
    SimpleMatrix QT;

    LanguageModel(int id, int o, int s);
    void UpdateLanguage(const SimpleMatrix& sample_matrix, int lang_id);
    SimpleMatrix Samplelanguage(int sample_times);
};

LanguageModel::LanguageModel(int id, int o, int s) : language_id(id), obj(o), sound(s), fitness(0.0), P(o, s, true, true), Q(s, o, true, true), QT(s, o, false, false) {
    QT = Q.Transpose();
}

void LanguageModel::UpdateLanguage(const SimpleMatrix& sample_matrix, int lang_id) {
    language_id = lang_id;
    // Update P matrix with sample_matrix
    for (int i = 0; i < obj; ++i) {
        for (int j = 0; j < sound; ++j) {
            P.data[i][j] = sample_matrix.data[i][j];
        }
    }
    P.Normalize();
    // Update Q as transpose of P
    Q = P.Transpose();
    Q.Normalize();
    QT = Q.Transpose();
}

SimpleMatrix LanguageModel::Samplelanguage(int sample_times) {
    SimpleMatrix sample_matrix(obj, sound, false, false);
    for (int o = 0; o < obj; ++o) {
        // build cumulative distribution
        vector<double> cdf = accumulate_sum(P.data[o]);

        // Draw tokens
        for (int t = 0; t < sample_times; ++t) {
            int k = random_choice_single(cdf);
            sample_matrix.data[o][k] += 1.0;
        }
    }
    return sample_matrix;
}

bool LanguageEqual(const LanguageModel& lang1, const LanguageModel& lang2, double tol=1e-6) {
    return MatrixEqual(lang1.P, lang2.P, tol) && MatrixEqual(lang1.Q, lang2.Q, tol);
}