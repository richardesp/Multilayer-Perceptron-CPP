# Multilayer-Perceptron-CPP
This practice consists of the implementation of a multilayer perceptron in C++. Several scripts have been added for the optimisation of hyper-parameters as well as additional arguments for training.

## Input arguments

- Argument t: Indicates the name of the file containing the training data to be used.
to be used. Without this argument, the program cannot run.
- Argument T: Indicates the name of the file containing the test data to be used. If
argument is not specified, use the training data as test data.
- Argument i: Indicates the number of iterations of the outer loop to be performed. If not specified, use 1000 iterations.
specified, use 1000 iterations.
- Argument l: Indicates the number of hidden layers of the neural network model. If not specified, use 1 hidden layer.
specified, use 1 hidden layer.
- Argument h: Indicates the number of neurons to introduce in each of the hidden layers.
hidden layers. If not specified, use 4 neurons.
- Argument e: Indicates the value of the eta parameter (η). By default, use η = 0.7.
- Argument m: Indicates the value of the mu (μ) parameter pair. By default, use μ = 1.
- Argument o: Boolean indicating whether the on-line version is to be used. If not specified, we will use the on-line version.
If not specified, we will use the off-line version.
- Argument f: Indicates the error function to be used during learning (0 for M SE error and 1 for M SE error).
for M SE error and 1 for cross-entropy). By default, use the M SE error.
- S <Scheduler type> (Scheduler argument): Specifies the type of learning rate scheduler to be applied to the programme.
- v (Verbose mode): Boolean to indicate that we want to save in a .txt file the data obtained from the experiment, saving the average MSE obtained in the training and tests, as well as the total number of iterations, error in each iteration of the training, and the current error after each iteration in tests (thanks to this it has been possible to carry out the crossvalidation convergence graphs).
- V <Splitting value> (Validation split): float to which we indicate the percentage of the training dataset from which we want to make the split (by default it takes the value of 0.2) to carry out the crossvalidation.
- c <print current CCR training and validation>: Boolean to indicate if we want to show the current CCR in each iteration in the training and validation dataset, necessary to carry out the convergence graphs of the mode.
