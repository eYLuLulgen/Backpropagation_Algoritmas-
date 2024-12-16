import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the dataset (only odd number multiplications)
data = np.array([
    [1, 3, 3],
    [3, 5, 15],
    [5, 7, 35],
    [7, 9, 63],
    [9, 5, 45]
])

# Normalize inputs and outputs
inputs = data[:, :2] / 10  # Scale inputs to range [0, 1]
outputs = data[:, 2:] / 100  # Scale outputs to range [0, 1]

# Initialize neural network parameters
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))

bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
losses = []
for epoch in range(epochs):
    # Feedforward
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute loss (Mean Squared Error)
    loss = np.mean((outputs - predicted_output) ** 2)
    losses.append(loss)

    # Backpropagation
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate

    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
print("\nFinal Outputs:")
print(predicted_output * 100)

# Plot the loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
