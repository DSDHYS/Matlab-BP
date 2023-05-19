/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include "Net.h"

using std::vector;
using std::string;

namespace Utils {
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    static double tansig(double x) {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    }

    vector<double> getFileData(const string &filename);

    vector<Sample> getTrainData(const string &filename);

    vector<Sample> getTestData(const string &filename);
    /**
     * @brief Get the Data object
     * 
     * @param buffer 
     * @return vector<Sample> 
     */
    vector<Sample> getData(vector<double> buffer);
}
