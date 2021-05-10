#pragma once
#include <vector>
#include "Network.h"
#include "Neuron.h"

typedef vector<Neuron> Layer;

using namespace std;

class Network
{
public:
    Network(const vector<unsigned>& topology);
    void feedForward(const vector<double>& inputVals);
    void backProp(const vector<double>& targetVals);
    void getResults(vector<double>& resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
};
