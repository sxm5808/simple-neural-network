import numpy as np
import matplotlib.pyplot as plt

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data for XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(1)
weights_input_hidden = 2 * np.random.random((2, 4)) - 1
weights_hidden_output = 2 * np.random.random((4, 1)) - 1

# Initialize error list
errors = []

# Training loop
for epoch in range(10000):
    input_layer = X
    hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output))

    output_error = y - output_layer
    errors.append(np.mean(np.abs(output_error)))
    output_delta = output_error * sigmoid_derivative(output_layer)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    weights_hidden_output += hidden_layer.T.dot(output_delta)
    weights_input_hidden += input_layer.T.dot(hidden_delta)

print("Final outputs:")
print(output_layer)

# Plot error over time
plt.plot(errors)
plt.title("Error over time")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.show()