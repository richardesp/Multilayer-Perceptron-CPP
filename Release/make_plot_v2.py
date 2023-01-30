import matplotlib.pyplot as plt
import sys
import re
import numpy as np

"""
    ITS OBLIGATORY TO PLOT THE EXECUTION FROM THE ALGORITHM IN A TEMPORAL FILE WITH > tmp.out WHEN THE BEST
    ARQUITECTURE IS CHOOSEN, THE BEST ARQUITECTURE IS THE ONE THAT MINIMIZE THE ERROR
"""


# Create a function that anottate the minimun value reached in error
def annot_min(x, y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text = "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=0")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(xmin - (x.shape[0] / 10), ymin + 0.0005),
                arrowprops={"arrowstyle": "->", "color": "gray"})


path = sys.argv[1]
with open(path, 'r') as f:
    # read every line
    lines = f.readlines()
    seeds_count = 1
    seeds = []
    training_error = []
    validation_error = []
    iterations = [1]
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):

        # Find the line that contains the iterations
        if line.startswith('SEED'):
            index += 2

            while index < len(lines):
                if lines[index].startswith('NETWORK'):
                    # The i-th seed is finished
                    iterations.pop()

                    # Plot validation and training
                    plt.clf()
                    annot_min(np.array(iterations), np.array(validation_error), ax=None)
                    plt.plot(iterations, training_error, label='Train')
                    plt.plot(iterations, validation_error, label='Test')
                    plt.title(f'Training result (Seed Number {seeds_count})')
                    plt.xlabel('Iterations')
                    plt.ylabel('Error')
                    plt.legend()
                    plt.show()

                    # Adding to seed the best validation error with the corresponding iteration
                    seeds.append((seeds_count, iterations[np.argmin(validation_error)],
                                  validation_error[np.argmin(validation_error)]))

                    training_error.clear()
                    validation_error.clear()
                    iterations.clear()
                    iterations.append(1)
                    seeds_count += 1

                    break

                new_line = re.split("Training error: (.*) Validation error: (.*)", lines[index])

                if len(new_line) == 4:
                    new_line[1] = new_line[1].replace('|', '')

                    training_error.append(float(new_line[1]))
                    validation_error.append(float(new_line[2]))
                    iterations.append(iterations[-1] + 1)

                print(lines[index])

                index += 1

    # Plot the best validation error for each seed in her corresponding iteration
    plt.clf()
    plt.bar([seed[0] for seed in seeds], [seed[2] for seed in seeds], label='Validation Error')
    plt.title('Best Validation Error for each seed')
    plt.xlabel('Seed Number')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
