/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#pragma once

#include <vector>
#include "Config.h"

using std::vector;

struct Sample {
    vector<double> feature, label;

    Sample();

    Sample(const vector<double> &feature, const vector<double> &label);

    void display();
};

struct Node {
    double value{}, bias{}, bias_delta{};
    vector<double> weight, weight_delta;

    explicit Node(size_t nextLayerSize);
};

class Net {
private:
    Node *inputLayer[Config::INNODE]{};
    Node *hiddenLayer[Config::HIDENODE_NUM][Config::HIDENODE]{};
    Node *outputLayer[Config::OUTNODE]{};

    /**
     * Clear all gradient accumulation
     *
     * Set 'weight_delta'(the weight correction value) and
     * 'bias_delta'(the bias correction value) to 0 of nodes
     */
    void grad_zero();

    /**
     * Forward propagation
     */
    void forward();

    /**
     * Calculate the value of Loss Function
     * @param label the label of sample (vector / numeric)
     * @return loss
     */
    double calculateLoss(const vector<double> &label);

        /**
     * Forward propagation
     */



public:

    Net();


    /**
     * Using network to predict sample
     * @param feature The feature of sample (vector)
     * @return Sample with 'feature' and 'label'(predicted)
     */
    Sample predict(const vector<double> &feature);

    /**
     * Using network to predict the sample set
     * @param predictDataSet The sample set
     * @return The sample set, in which each sample has 'feature' and 'label'(predicted)
     */
    vector<Sample> predict(const vector<Sample> &predictDataSet);
    /**
     * 归一化计算
     * @param x 输入值
     * @param xmax 最大值
     * @param xmin 最小值
     * @return 归一化数据
     */
    double mapminmax(double x,double xmax,double xmin);
        /**
     * 反归一化计算
     * @param y 归一化输入值
     * @param xmax 最大值
     * @param xmin 最小值
     * @return 归一化数据
     */
    double remapminmax(double y,double xmax,double xmin);


};
