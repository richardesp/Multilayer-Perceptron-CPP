/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron() {

}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
    nOfLayers = nl;

    // Initializan all values to 0
    // We have n layers
    layers = (Layer *) calloc(nOfLayers, sizeof(Layer));

    // Initializing every neuron of the layer

    // Remember that npl[0] and npl[nOflayers - 1] is the input
    // and output layer
    for (int i = 0; i < nOfLayers; ++i) {
        layers[i].nOfNeurons = npl[i];
        layers[i].neurons = (Neuron *) calloc(layers[i].nOfNeurons, sizeof(Neuron));

        // Specifiying every value of the neuron
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            layers[i].neurons[j].out = 0;
            layers[i].neurons[j].delta = 0;

            // We have one bias and nInputs from the layer_i - 1 weights

            // If we are in the input layer, the weights must be 1
            if (i == 0) {
                layers[i].neurons[j].w = NULL;

                layers[i].neurons[j].deltaW = NULL;
                layers[i].neurons[j].lastDeltaW = NULL;
                layers[i].neurons[j].wCopy = NULL;
            } else if (i == nOfLayers - 1 and j == layers[i].nOfNeurons - 1 and outputFunction == 1) {
                // If we are in the last output neuron using softmax function, we can obviate it
                layers[i].neurons[j].w = NULL;

                layers[i].neurons[j].deltaW = NULL;
                layers[i].neurons[j].lastDeltaW = NULL;
                layers[i].neurons[j].wCopy = NULL;

            }

                // The input of every layer depends on from the neurons of the prev layer
            else {
                // Weights: nNeurons from the prev layer
                // Bias: additional weight with a constant input (1)
                layers[i].neurons[j].w = (double *) calloc(layers[i - 1].nOfNeurons + 1, sizeof(double));
                layers[i].neurons[j].deltaW = (double *) calloc(layers[i - 1].nOfNeurons + 1, sizeof(double));
                layers[i].neurons[j].lastDeltaW = (double *) calloc(layers[i - 1].nOfNeurons + 1, sizeof(double));
                layers[i].neurons[j].wCopy = (double *) calloc(layers[i - 1].nOfNeurons + 1, sizeof(double));
            }
        }
    }

    return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
    freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
    for (int i = 0; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            free(layers[i].neurons[j].w);
            free(layers[i].neurons[j].deltaW);
            free(layers[i].neurons[j].lastDeltaW);
            free(layers[i].neurons[j].wCopy);
        }

        free(layers[i].neurons);
    }

    free(layers);
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++) {

                if (layers[i].neurons[j].w != NULL)
                    layers[i].neurons[j].w[k] = randomDouble(-1, 1);
            }
        }
    }
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input) {
    for (int i = 0; i < layers[0].nOfNeurons; i++) {
        layers[0].neurons[i].out = input[i]; // wi = 1
    }
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output) {
    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++) {
        output[i] = layers[nOfLayers - 1].neurons[i].out;
    }
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons; k++) {
                if (layers[i].neurons[j].w != NULL)
                    layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];

            }
        }
    }
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++) {
                if (layers[i].neurons[j].w != NULL)
                    layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];

            }
        }
    }
}

// Create a function for calculate net from a given layer
double MultilayerPerceptron::calculateNet(int layer, int neuron) {
    double net = 0;

    // If we are in the input layer, the net is the input
    if (layer == 0) {
        net = layers[layer].neurons[neuron].out;
    }

        // If we are in the hidden or output layer, the net is the sum of the
        // weights * inputs
    else {
        // For every neuron in the prev layer
        for (int i = 0; i < layers[layer - 1].nOfNeurons; i++) {
            net += layers[layer - 1].neurons[i].out * layers[layer].neurons[neuron].w[i];
        }

        // Adding the bias
        net += layers[layer].neurons[neuron].w[layers[layer - 1].nOfNeurons];
    }

    return net;
}

void MultilayerPerceptron::softmax(int layerIndex) {
    double *net = new double[layers[layerIndex].nOfNeurons], summationOfExp = 0;

    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j) {

        if (j == layers[layerIndex].nOfNeurons - 1) { // If we are in the last neuron, net its 0
            net[j] = 0;
        } else {

            Neuron *neuron = layers[layerIndex].neurons + j;
            net[j] = 0;

            for (int k = 0; k < layers[layerIndex - 1].nOfNeurons; ++k)
                net[j] += neuron->w[k + 1] *
                          layers[layerIndex -
                                 1].neurons[k].out; //We add 1 to the weight index to account for the bias w_0

            net[j] += neuron->w[layers[layerIndex - 1].nOfNeurons];

        }

        summationOfExp += exp(net[j]);
    }

    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j)
        layers[layerIndex].neurons[j].out = exp(net[j]) / summationOfExp;

    delete[] net;
}

void MultilayerPerceptron::sigmoid(int layerIndex) {
    for (int j = 0; j < layers[layerIndex].nOfNeurons; ++j) {
        Neuron *neuron = layers[layerIndex].neurons + j;
        double net = neuron->w[0];

        for (int k = 0; k < layers[layerIndex - 1].nOfNeurons; ++k)
            net += neuron->w[k + 1] *
                   layers[layerIndex - 1].neurons[k].out; //We add 1 to the weight index to account for the bias w_0

        neuron->out = 1.0 / (1.0 + exp(-net));
    }
}


// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            double net = 0;

            // Sum of the inputs * weights
            for (int k = 0; k < layers[i - 1].nOfNeurons; k++) {

                if (layers[i].neurons[j].w != NULL)
                    net += layers[i - 1].neurons[k].out * layers[i].neurons[j].w[k];

            }

            // Add the bias
            if (layers[i].neurons[j].w != NULL)
                net += layers[i].neurons[j].w[layers[i - 1].nOfNeurons];


            if (i == nOfLayers - 1) {
                if (outputFunction == 0) // Apply a sigmoid function
                    layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net));
                else {  // Apply a softmax function
                    softmax(i);
                }
            } else {
                // Apply the activation function
                layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net));
            }
        }
    }
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double *target, int errorFunction) {
    double error = 0;

    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++) {
        error += pow(target[i] - layers[nOfLayers - 1].neurons[i].out, 2);
    }

    return error / layers[nOfLayers - 1].nOfNeurons;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double *target, int errorFunction) {

    for (int i = nOfLayers - 1; i >= 0; i--) {

        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            // If we are in the last layer
            if (i == nOfLayers - 1) {

                if (outputFunction == 0) { // Sigmoid output function

                    if (errorFunction == 0) { // MSE error function
                        double out = layers[i].neurons[j].out;
                        double error = target[j] - out;

                        layers[i].neurons[j].delta = 2 * error * out * (1 - out);

                    } else {
                        if (errorFunction == 1) { // Cross Entropy error function
                            double out = layers[i].neurons[j].out;

                            double error = target[j] / out;
                            layers[i].neurons[j].delta = error * out * (1 - out);
                        }
                    }

                } else { // Softmax output function

                    if (errorFunction == 0) { // MSE error function
                        double out = layers[i].neurons[j].out;

                        // Delta must be equal to 0 to avoid create a new variable and waste memory
                        if (layers[i].neurons[j].deltaW != nullptr) {
                            layers[i].neurons[j].delta = 0;

                            for (int k = 0; k < layers[i].nOfNeurons; k++) {
                                layers[i].neurons[j].delta += ((target[k] - layers[i].neurons[k].out) *
                                                               out * ((double) (k == j) - layers[i].neurons[k].out));
                            }
                        }

                    } else {
                        if (errorFunction == 1) { // Cross entropy error function

                            double out = layers[i].neurons[j].out;

                            if (layers[i].neurons[j].deltaW != nullptr) {

                                layers[i].neurons[j].delta = 0;

                                for (int k = 0; k < layers[i].nOfNeurons; k++) {
                                    double out_k = layers[i].neurons[k].out;

                                    double error = target[k] / out_k;
                                    layers[i].neurons[j].delta += error * out * ((double) (k == j) - out_k);

                                }
                            }
                        }
                    }
                }

            } else {
                double sum = 0.0;

                for (int k = 0; k < this->layers[i + 1].nOfNeurons; k++) {
                    if (this->layers[i + 1].neurons[k].w != nullptr) {
                        sum += this->layers[i + 1].neurons[k].w[j + 1] * this->layers[i + 1].neurons[k].delta;
                    }
                }
                this->layers[i].neurons[j].delta =
                        sum * this->layers[i].neurons[j].out * (1 - this->layers[i].neurons[j].out);
            }
        }
    }

}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons; k++) {
                if (layers[i].neurons[j].deltaW != NULL)
                    layers[i].neurons[j].deltaW[k] += layers[i].neurons[j].delta * layers[i - 1].neurons[k].out;

            }

            // Bias
            if (layers[i].neurons[j].deltaW != NULL)
                layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons] += layers[i].neurons[j].delta;
        }
    }
}

void MultilayerPerceptron::weightAdjustmentOffline() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons; k++) {
                if (layers[i].neurons[j].w != NULL)
                    layers[i].neurons[j].w[k] +=
                            ((eta * layers[i].neurons[j].deltaW[k]) / nOfTrainingPatterns) +
                            ((mu * eta * layers[i].neurons[j].lastDeltaW[k]) / nOfTrainingPatterns);

            }

            // Bias
            if (layers[i].neurons[j].w != NULL)
                layers[i].neurons[j].w[layers[i - 1].nOfNeurons - 1] +=
                        ((eta * layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons - 1]) / nOfTrainingPatterns) +
                        ((mu * eta * layers[i].neurons[j].lastDeltaW[layers[i - 1].nOfNeurons - 1]) /
                         nOfTrainingPatterns);
        }
    }
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k < layers[i - 1].nOfNeurons; k++) {
                if (layers[i].neurons[j].w != NULL)
                    layers[i].neurons[j].w[k] +=
                            eta * layers[i].neurons[j].deltaW[k] + mu * eta * layers[i].neurons[j].lastDeltaW[k];
            }

            // Bias
            if (layers[i].neurons[j].w != NULL)
                layers[i].neurons[j].w[layers[i - 1].nOfNeurons - 1] +=
                        eta * layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons - 1] +
                        mu * eta * layers[i].neurons[j].lastDeltaW[layers[i - 1].nOfNeurons - 1];
        }
    }
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
    // The first layer receive constant inputs (1)
    for (int i = 1; i < nOfLayers; ++i) {
        std::cout << "Layer " << i << ": \n";
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            std::cout << "\tNeuron " << j << ": \n";
            std::cout << "\t\tWeights: ";

            // Printing all weights
            // nInputs from layer i - 1 and bias
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; ++k) {

                // Printing bias
                if (k == layers[i - 1].nOfNeurons and layers[i].neurons[j].w != NULL)
                    std::cout << "(bias) " << layers[i].neurons[j].w[k];

                else if (layers[i].neurons[j].w != NULL)
                    std::cout << layers[i].neurons[j].w[k] << ", ";
            }
            std::cout << "\n";
        }
    }
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double *input, double *target, int errorFunction) {

    feedInputs(input);
    forwardPropagate();
    backpropagateError(target, errorFunction);
    accumulateChange();

}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double *input, double *target, int errorFunction) {
    for (int i = 1; i < nOfLayers; ++i) {
        for (int j = 0; j < layers[i].nOfNeurons; ++j) {
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; ++k) {

                if (layers[i].neurons[j].lastDeltaW != NULL) {
                    layers[i].neurons[j].lastDeltaW[k] = layers[i].neurons[j].deltaW[k];
                    layers[i].neurons[j].deltaW[k] = 0.0;
                }
            }
        }
    }

    feedInputs(input);
    forwardPropagate();
    backpropagateError(target, errorFunction);
    accumulateChange();
    weightAdjustment();
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset *trainDataset, int errorFunction, bool onlineMode) {
    // Perform a training in online mode or offline mode (batch)
    int i;

    if (onlineMode) { // Online mode
        for (i = 0; i < trainDataset->nOfPatterns; i++) {
            performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
        }
    } else { // Offline mode

        // Cleaning delta weights
        for (int i = 1; i < nOfLayers; ++i) {
            for (int j = 0; j < layers[i].nOfNeurons; ++j) {
                for (int k = 0; k < layers[i - 1].nOfNeurons + 1; ++k) {
                    if (layers[i].neurons[j].lastDeltaW != NULL) {
                        layers[i].neurons[j].lastDeltaW[k] = layers[i].neurons[j].deltaW[k];
                        layers[i].neurons[j].deltaW[k] = 0.0;
                    }

                }
            }
        }

        for (i = 0; i < trainDataset->nOfPatterns; i++) {
            performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
        }
        weightAdjustmentOffline();

    }
}

double MultilayerPerceptron::getMSE(double *output, double *target, int nOfOutputs) {
    double mse = 0.0;

    for (int i = 0; i < nOfOutputs; i++) {
        mse += pow((output[i] - target[i]), 2);
    }

    return mse / nOfOutputs;
}

double MultilayerPerceptron::getCrossEntropy(const double *target) {
    double *prediction = new double[layers[nOfLayers - 1].nOfNeurons];
    getOutputs(prediction);

    double summation = .0;

    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; ++i)
        summation -= (target[i] * log(prediction[i]));

    delete[] prediction;

    return summation / layers[nOfLayers - 1].nOfNeurons;
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset *dataset, int errorFunction) {
    double error = 0.0;
    double *output = new double[dataset->nOfOutputs];

    for (int i = 0; i < dataset->nOfPatterns; i++) {
        feedInputs(dataset->inputs[i]);
        forwardPropagate();
        getOutputs(output);

        if (errorFunction == 0) // MSE
            error += getMSE(dataset->outputs[i], output, dataset->nOfOutputs);

        else if (errorFunction == 1) // Cross Entropy
            error += getCrossEntropy(dataset->outputs[i]);
    }

    delete[] output;
    return error / dataset->nOfPatterns;
}

int getMaxIndex(double *array, int size) {
    int maxIndex = 0;

    for (int i = 1; i < size; ++i) {
        if (array[i] > array[maxIndex])
            maxIndex = i;
    }

    return maxIndex;
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset *dataset) {
    double ccr = 0.0;
    double *output = new double[dataset->nOfOutputs];

    for (int i = 0; i < dataset->nOfPatterns; i++) {
        feedInputs(dataset->inputs[i]);
        forwardPropagate();
        getOutputs(output);

        if (getMaxIndex(output, dataset->nOfOutputs) == getMaxIndex(dataset->outputs[i], dataset->nOfOutputs))
            ccr++;
    }

    delete[] output;
    return ccr / dataset->nOfPatterns;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset *dataset) {
    int i;
    int j;
    int numSalidas = layers[nOfLayers - 1].nOfNeurons;
    double *salidas = new double[numSalidas];

    cout << "Id,Category" << endl;

    for (i = 0; i < dataset->nOfPatterns; i++) {

        feedInputs(dataset->inputs[i]);
        forwardPropagate();
        getOutputs(salidas);

        int maxIndex = 0;
        for (j = 0; j < numSalidas; j++)
            if (salidas[j] >= salidas[maxIndex])
                maxIndex = j;

        cout << i << "," << maxIndex << endl;

    }
}


// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void
MultilayerPerceptron::runBackPropagation(Dataset *trainDataset, Dataset *testDataset, Dataset *validationDataset,
                                         int maxiter,
                                         double *errorTrain,
                                         double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction,
                                         std::vector<double> &trainErrorVector,
                                         std::vector<double> &testErrorVector, bool verbose,
                                         bool learningRateScheduler, char *learningRateSchedulerTpe,
                                         bool printCrossValidation, bool onlineMode, bool printCurrentCCR) {
    int countTrain = 0;

    // Random assignment of weights (starting point)
    randomWeights();

    double minTrainError = 0;
    int iterWithoutImproving = 0;
    nOfTrainingPatterns = trainDataset->nOfPatterns;

    double oldEta = eta;

    // Learning
    do {

        train(trainDataset, errorFunction, onlineMode);
        double trainError = test(trainDataset, errorFunction);
        if (countTrain == 0 || trainError < minTrainError) {
            minTrainError = trainError;
            copyWeights();
            iterWithoutImproving = 0;
        } else if ((trainError - minTrainError) < 0.00001)
            iterWithoutImproving = 0;
        else
            iterWithoutImproving++;

        if (iterWithoutImproving == 50) {
            cout << "We exit because the training is not improving!!" << endl;
            restoreWeights();
            countTrain = maxiter;
        }

        countTrain++;

        cout << "Iteration " << countTrain << "\t Training error: " << trainError;
        if (printCrossValidation)
            cout << " | Validation error: " << test(validationDataset, errorFunction);
        cout << "\n";

        if (printCurrentCCR) {
            cout << "Iteration " << countTrain << "\t Training CCR: " << testClassification(trainDataset);

            if (printCrossValidation)
                cout << " | Validation CCR: " << testClassification(validationDataset);
            cout << "\n";
        }

        if (verbose) {
            trainErrorVector.push_back(trainError);

            if (printCrossValidation)
                testErrorVector.push_back(test(validationDataset, errorFunction));
        }

        if (learningRateScheduler && strcmp(learningRateSchedulerTpe, "None") != 0) {
            if (strcmp(learningRateSchedulerTpe, "exponential") == 0)
                eta = eta * 0.99;

            else if (strcmp(learningRateSchedulerTpe, "step") == 0) {
                if (countTrain % 100 == 0)
                    eta = eta * 0.5;
            } else if (strcmp(learningRateSchedulerTpe, "cosine") == 0)
                eta = oldEta * 0.5 * (1 + cos(M_PI * countTrain / maxiter));

            else if (strcmp(learningRateSchedulerTpe, "linear") == 0)
                eta = oldEta * (float) (1 - (float) countTrain / (float) maxiter);

            else
                throw std::invalid_argument("The learning rate scheduler type is not valid");
        }

        std::cout << "Current learning rate --> " << eta << std::endl;

    } while (countTrain < maxiter);

    if (iterWithoutImproving != 50)
        restoreWeights();

    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    //printNetwork();

    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;
    for (int i = 0; i < testDataset->nOfPatterns; i++) {
        double *prediction = new double[testDataset->nOfOutputs];

        // Feed the inputs and propagate the values
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        for (int j = 0; j < testDataset->nOfOutputs; j++)
            cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
        delete[] prediction;

    }

    *errorTest = test(testDataset, errorFunction);
    *errorTrain = minTrainError;
    *ccrTest = testClassification(testDataset);
    *ccrTrain = testClassification(trainDataset);

    eta = oldEta; // We restore the original learning rate

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *fileName) {
    // Object for writing the file
    ofstream f(fileName);
    if (!f.is_open())
        return false;

    // Write the number of layers and the number of layers in every layer
    f << nOfLayers;

    for (int i = 0; i < nOfLayers; i++) {
        f << " " << layers[i].nOfNeurons;
    }
    f << " " << outputFunction;
    f << endl;

    // Write the weight matrix of every layer
    for (int i = 1; i < nOfLayers; i++)
        for (int j = 0; j < layers[i].nOfNeurons; j++)
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                if (layers[i].neurons[j].w != NULL)
                    f << layers[i].neurons[j].w[k] << " ";

    f.close();

    return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *fileName) {
    // Object for reading a file
    ifstream f(fileName);

    if (!f.is_open())
        return false;

    // Number of layers and number of neurons in every layer
    int nl;
    int *npl;

    // Read number of layers
    f >> nl;

    npl = new int[nl];

    // Read number of neurons in every layer
    for (int i = 0; i < nl; i++) {
        f >> npl[i];
    }
    f >> outputFunction;

    // Initialize vectors and data structures
    initialize(nl, npl);

    // Read weights
    for (int i = 1; i < nOfLayers; i++)
        for (int j = 0; j < layers[i].nOfNeurons; j++)
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                if (!(outputFunction == 1 && (i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
                    f >> layers[i].neurons[j].w[k];

    f.close();
    delete[] npl;

    return true;
}
