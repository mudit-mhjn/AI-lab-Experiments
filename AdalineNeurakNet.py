"""
Implementing XOR operation using the adaline neural network
"""
import numpy as np

LEARNING_RATE = 0.45


# Step function
def step(x):
    if x > 0:
        return 1
    else:
        return -1


# input dataset representing the logical OR operator (including constant BIAS input of 1)
INPUTS = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

OUTPUTS = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# initialize weights randomly with mean 0
WEIGHTS = 2 * np.random.random((3, 1)) - 1
print("Random Weights before training", WEIGHTS)

errors = []

# Training loop
for epochs in range(100):

    for input_item, desired in zip(INPUTS, OUTPUTS):
        ADALINE_OUTPUT = (input_item[0] * WEIGHTS[0]) + (input_item[1] * WEIGHTS[1]) + (input_item[2] * WEIGHTS[2])
        ADALINE_OUTPUT = step(ADALINE_OUTPUT)
        ERROR = desired - ADALINE_OUTPUT

        errors.append(ERROR)

        WEIGHTS[0] = WEIGHTS[0] + LEARNING_RATE * ERROR * input_item[0]
        WEIGHTS[1] = WEIGHTS[1] + LEARNING_RATE * ERROR * input_item[1]
        WEIGHTS[2] = WEIGHTS[2] + LEARNING_RATE * ERROR * input_item[2]

print("New Weights after training", WEIGHTS)
for input_item, desired in zip(INPUTS, OUTPUTS):
    ADALINE_OUTPUT = (input_item[0] * WEIGHTS[0]) + (input_item[1] * WEIGHTS[1]) + (input_item[2] * WEIGHTS[2])
    ADALINE_OUTPUT = step(ADALINE_OUTPUT)
    print("Actual ", ADALINE_OUTPUT, "Desired ", desired)
