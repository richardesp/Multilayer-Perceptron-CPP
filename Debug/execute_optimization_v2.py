import os
import numpy as np
import sys
from alive_progress import alive_bar

# XOR PROBLEM
print(">>> NOMNIST PROBLEM <<<")

train_dataset_path = '../../datasetsLA2IMC/dat/train_nomnist.dat'
test_dataset_path = '../../datasetsLA2IMC/dat/test_nomnist.dat'
experiment_name = f"nomnist_optimized_dataset_{sys.argv[3]}"

learning_rates = np.linspace(0.1, 1, 10)
momentums = np.linspace(0.1, 1, 10)
lr_schedulers = np.array(["None", "exponential", "step", "cosine", "linear"])
normalizing_values = np.array([False, True])  # In kaggle mode we need to obtain the most real value
output_activation_functions = np.array(["softmax"])
learning_mode = np.array(["batch"])
error_functions = np.array(["cross_entropy"])

hidden_layers = np.array([int(sys.argv[1])])  # First argument
neurons_per_layer = np.array([int(sys.argv[2])])  # Second argument
iterations = 10

best_test_ccr = sys.float_info.min

total = len(learning_rates) * len(momentums) * len(lr_schedulers) * len(hidden_layers) * len(neurons_per_layer) * len(
    normalizing_values) * len(output_activation_functions) * len(learning_mode) * len(error_functions)
bar_length = 30

i = int(1)

with alive_bar(total) as bar:
    for learning_rate in learning_rates:
        for momentum in momentums:
            for lr_scheduler in lr_schedulers:
                for hidden_layer in hidden_layers:
                    for neuron in neurons_per_layer:
                        for normalizing_value in normalizing_values:
                            for output_activation_function in output_activation_functions:
                                for learning in learning_mode:
                                    for error_function in error_functions:

                                        os.system(f"./la2 -t {train_dataset_path} -T {test_dataset_path} -l {hidden_layer} -h {neuron} -i {iterations} -e {learning_rate} -m {momentum} -S {lr_scheduler} {'-n' if normalizing_value == True else ''} {'-o' if learning == 'online' else ''} {'-s' if output_activation_function == 'softmax' else ''} {'-f 1' if error_function == 'cross_entropy' else '-f 0'} -V 0.2 | grep \"Test CCR\" | grep -oP \'(?<=:).*\' | grep -Eo \".* \" | tr -d \'-\' | tr -d \'+\' > .trash_{sys.argv[3]}.out")

                                        with open(f'.trash_{sys.argv[3]}.out', 'r') as f:
                                            current_test_ccr = float(f.read())

                                        if current_test_ccr > best_test_ccr:
                                            print(f"\t>>> 🥇 New best test CCR: {current_test_ccr} vs {best_test_ccr}")
                                            best_test_ccr = current_test_ccr

                                            best_learning_rate = learning_rate
                                            best_momentum = momentum
                                            best_lr_scheduler = lr_scheduler
                                            best_hidden_layer = hidden_layer
                                            best_neuron = neuron
                                            best_normalizing_value = normalizing_value
                                            best_output_activation_function = output_activation_function
                                            best_learning = learning
                                            best_error_function = error_function

                                        else:
                                            print(f"\t>>> 🥉 Current test CCR: {current_test_ccr} vs {best_test_ccr}")

                                        """
                                        percent = 100.0 * i / total
                                        sys.stdout.write('\r')
                                        sys.stdout.write("Completed: [{:{}}] {:>3}%\n".format('=' * int(percent / (100.0 / bar_length)),
                                                                                              bar_length, int(percent)))
                                        sys.stdout.flush()

                                        i += 1
                                        """
                                        bar()

# Once finished the execution, we are going to train the model with the best parameters

# Execute the program with best params
os.system(f"./la2 -t {train_dataset_path} -T {test_dataset_path} -l {best_hidden_layer} -h {best_neuron} -i {iterations} -e {best_learning_rate} -m {best_momentum} -S {best_lr_scheduler} {'-n' if best_normalizing_value == True else ''} {'-o' if best_learning == 'online' else ''} {'-s' if best_output_activation_function == 'softmax' else ''} {'-f 1' if best_error_function == 'cross_entropy' else '-f 0'} -V 0.2 -v {experiment_name} | grep \"Test error\" | grep -oP \'(?<=:).*\' | grep -Eo \".* \" | tr -d \'-\' | tr -d \'+\' > .trash_{sys.argv[3]}.out")

with open(f'{experiment_name}_best_params.out', 'w') as f:
    f.write(f"Best learning rate: {best_learning_rate}\n")
    f.write(f"Best momentum: {best_momentum}\n")
    f.write(f"Best lr scheduler: {best_lr_scheduler}\n")
    f.write(f"Best hidden layer: {best_hidden_layer}\n")
    f.write(f"Best neuron: {best_neuron}\n")
    f.write(f"Best normalizing value: {best_normalizing_value}\n")
    f.write(f"Best output activation function: {best_output_activation_function}\n")
    f.write(f"Best learning: {best_learning}\n")
    f.write(f"Best error function: {best_error_function}\n")
    f.write(f"Best test CCR: {best_test_ccr}\n\n")
    f.write(f"Command for execute the experiment: ./la2 -t {train_dataset_path} -T {test_dataset_path} -l {best_hidden_layer} -h {best_neuron} -i {iterations} -e {best_learning_rate} -m {best_momentum} -S {best_lr_scheduler} {'-n' if best_normalizing_value == True else ''} {'-o' if best_learning == 'online' else ''} {'-s' if best_output_activation_function == 'softmax' else ''} {'-f 1' if best_error_function == 'cross_entropy' else '-f 0'} -V 0.2 -v {experiment_name}")

os.system(f"rm .trash_{sys.argv[3]}.out")
