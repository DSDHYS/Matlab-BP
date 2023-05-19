/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#include "Net.h"
#include "Utils.h"
#include <random>

using namespace std;


Net::Net() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    /**
     * Initialize input layer
     */
    double input_weight[5][2]={{-2.5555,-1.8765},{-1.3927,2.8727},{2.8813   -1.2452},{-1.9704   -2.4330},{2.4498   -1.9478}};
    // double input_weight[2][2]={{0,0},{0,0}};

    for (size_t i = 0; i < Config::INNODE; ++i) {
        inputLayer[i] = new Node(Config::HIDENODE);

        for (size_t j = 0; j < Config::HIDENODE; ++j) {

            // Initialize 'weight'(the weight value)
            // from the i-th node in the input layer to the j-th node in the hidden layer

            // inputLayer[i]->weight[j] = distribution(rd);
            inputLayer[i]->weight[j] = input_weight[j][i];





            // Initialize 'weight_delta'(the weight correction value)
            // from the i-th node in the input layer to the j-th node in the hidden layer
            inputLayer[i]->weight_delta[j] = 0.f;
        }
    }



    /**
     * Initialize hidden layer
     */
    double hidden_bias[5][5]={{3.1016,1.4404,-0.0222,-1.5503,3.1329},{-1.9499,1.1396,0.0495,0.8647,1.8445},{-1.9319,0.9656,0.1601,-0.9930,1.8529}};
    double hidden_weight[2][4]={{ -0.0768,-1.8288,2.3468,0.6377},{1.3924,-1.6239,1.6746,-1.5336}};


    for(size_t l=0;l<Config::HIDENODE_NUM;++l)
    {
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            hiddenLayer[l][j] = new Node(Config::OUTNODE);

            // Initialize 'bias'(the bias value)
            // of the j-th node in the hidden layer

            hiddenLayer[l][j]->bias = hidden_bias[l][j];


            // Initialize 'bias_delta'(the bias correction value)
            // of the j-th node in the hidden layer
            hiddenLayer[l][j]->bias_delta = 0.f;
            for (size_t k = 0; k < Config::HIDENODE; ++k) {

                // Initialize 'weight'(the weight value)
                // from the j-th node in the hidden layer to the k-th node in the output layer

                // hiddenLayer[l][j]->weight[k] = distribution(rd);
                hiddenLayer[l][j]->weight[k] = hidden_weight[l][j+k*2];



                // Initialize 'weight_delta'(the weight correction value)
                // from the j-th node in the hidden layer to the k-th node in the output layer
                hiddenLayer[l][j]->weight_delta[k] = 0.f;
            }
        }
    }

    double output_bias[2]={0.3969,0.2609};

    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        outputLayer[k] = new Node(0);

        // Initialize 'bias'(the bias value)
        // of the k-th node in the output layer

        // outputLayer[k]->bias = distribution(rd);
        outputLayer[k]->bias = output_bias[k];



        // Initialize 'bias_delta'(the bias correction value)
        // of the k-th node in the output layer
        outputLayer[k]->bias_delta = 0.f;
    }
}

void Net::grad_zero() {

    // Clear 'weight_delta'(the weight correction value)
    // of all nodes in the input layer
    for (auto &nodeOfInputLayer: inputLayer) {
        nodeOfInputLayer->weight_delta.assign(nodeOfInputLayer->weight_delta.size(), 0.f);
    }

    // Clear 'weight_delta'(the weight correction value) and 'bias_delta'(the bias correction value)
    // of all nodes in the hidden layer
    for (auto &nodeOfHiddenLayer: hiddenLayer) {
        for (int i = 0; i < Config::HIDENODE_NUM; i++)
        {
            nodeOfHiddenLayer[i]->bias_delta = 0.f;
            nodeOfHiddenLayer[i]->weight_delta.assign(nodeOfHiddenLayer[i]->weight_delta.size(), 0.f);
        }
    }

    // Clear 'bias_delta'(the bias correction value)
    // of all nodes in the hidden layer
    for (auto &nodeOfOutputLayer: outputLayer) {
        nodeOfOutputLayer->bias_delta = 0.f;
    }
}

void Net::forward() {

    /**
     * The input layer propagate forward to the hidden layer.
     * MathJax formula: h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
     */
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        double sum = 0;
        for (size_t i = 0; i < Config::INNODE; ++i) {
            sum += inputLayer[i]->value * inputLayer[i]->weight[j];
        }
        // sum -= hiddenLayer[0][j]->bias;
        sum += hiddenLayer[0][j]->bias;

        // hiddenLayer[0][j]->value = Utils::sigmoid(sum);
        hiddenLayer[0][j]->value = Utils::tansig(sum);
        cout<< hiddenLayer[0][j]->value<<endl;




    }
    for(size_t l = 0; l < Config::HIDENODE_NUM-1; l++)
    {
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        double sum = 0;
        for (size_t i = 0; i < Config::HIDENODE; ++i) {
            sum += hiddenLayer[l][i]->value * hiddenLayer[l][i]->weight[j];
        }
        // sum -= hiddenLayer[j+1][l]->bias;
        sum += hiddenLayer[l+1][j]->bias;


        // hiddenLayer[j+1][l]->value = Utils::sigmoid(sum);
        hiddenLayer[l+1][j]->value = Utils::tansig(sum);
        cout<<hiddenLayer[l+1][j]->value<<endl;


    }
    }



    /**
     * The hidden layer propagate forward to the output layer.
     * MathJax formula: \hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
     */
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double sum = 0;
        for (size_t j = 0; j < Config::HIDENODE; j++) {
            sum += hiddenLayer[Config::HIDENODE_NUM-1][j]->value * hiddenLayer[Config::HIDENODE_NUM-1][j]->weight[k];
        }
        // sum -= outputLayer[k]->bias;
        sum += outputLayer[k]->bias;


        // outputLayer[k]->value = Utils::sigmoid(sum);
        // outputLayer[k]->value = Utils::tansig(sum);
        outputLayer[k]->value = sum;



    }
}

double Net::calculateLoss(const vector<double> &label) {
    double loss = 0.f;

    /**
     * MathJax formula: Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
     */
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double tmp = std::fabs(outputLayer[k]->value - label[k]);
        loss += tmp * tmp / 2;
    }

    return loss;
}



Sample Net::predict(const vector<double> &feature) {

    // load sample into the network
    for (size_t i = 0; i < Config::INNODE; ++i)
        inputLayer[i]->value = feature[i];

    forward();

    vector<double> label(Config::OUTNODE);
    for (size_t k = 0; k < Config::OUTNODE; ++k)
        label[k] = outputLayer[k]->value;

    Sample pred = Sample(feature, label);
    return pred;
}
double Net::mapminmax(double x,double xmax,double xmin)
{
    return 2*(x-xmin)/(xmax-xmin)-1;
}
double Net::remapminmax(double y,double xmax,double xmin)
{
    return (y+1)*(xmax-xmin)/2+xmin;
}

vector<Sample> Net::predict(const vector<Sample> &predictDataSet) {
    vector<Sample> predSet;

    for (auto &sample: predictDataSet) {
        Sample pred = predict(sample.feature);
        predSet.push_back(pred);
    }

    return predSet;
}

Node::Node(size_t nextLayerSize) {
    weight.resize(nextLayerSize);
    weight_delta.resize(nextLayerSize);
}

Sample::Sample() = default;

Sample::Sample(const vector<double> &feature, const vector<double> &label) {
    this->feature = feature;
    this->label = label;
}

void Sample::display() {
    printf("input : ");
    for (auto &x: feature) printf("%lf ", x);
    puts("");
    printf("output: ");
    for (auto &y: label) printf("%lf ", y);
    puts("");
}
