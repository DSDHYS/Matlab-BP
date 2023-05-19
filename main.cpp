/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#include <iostream>

#include "lib/Net.cpp"
#include "lib/Utils.cpp"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {


    // Create neural network object
    
    Net net;
    
    // Prediction of samples using neural network
    // const vector<Sample> testDataSet = Utils::getTestData("data/testdata.txt");
    vector<double> buffer ={1.5,4.4};

    // 归一化
    buffer[0]=net.mapminmax(buffer[0],3,1);
    buffer[1]=net.mapminmax(buffer[1],6,4);

    const vector<Sample> testDataSet = Utils::getData(buffer);


    vector<Sample> predSet = net.predict(testDataSet);

    // 反归一化
    for (auto &pred: predSet) {
        buffer[0]=net.remapminmax(pred.label[0],9,7);
        buffer[1]=net.remapminmax(pred.label[1],12,10);
        cout<<buffer[0]<<endl;
        cout<<buffer[1]<<endl;
        // cout<<pred.label[0]<<endl;
        // cout<<pred.label[1]<<endl;

        // pred.display();
    }

    return 0;
}
