import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        """
        Initialize the neural network with random weights
        
        Parameters:
        - input_size: number of input features (2 in our case)
        - hidden_size: number of neurons in the hidden layer
        - output_size: number of output neurons (1 for binary classification)
        """
        # Initialize weights with small random values
        # W1 is the weight matrix from input layer to hidden layer (shape: input_size x hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        
        # b1 is the bias vector for the hidden layer (shape: 1 x hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # W2 is the weight matrix from hidden layer to output layer (shape: hidden_size x output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        
        # b2 is the bias vector for the output layer (shape: 1 x output_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.z1 = None  # weighted input to hidden layer
        self.a1 = None  # activation of hidden layer
        self.z2 = None  # weighted input to output layer
        self.a2 = None  # activation of output layer (prediction)
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        
        This squashes input values into range (0, 1)
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
        
        This is used in backpropagation
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Parameters:
        - X: input data of shape (batch_size, input_size)
        
        Returns:
        - Output probabilities after sigmoid activation
        """
        # Linear transformation: z1 = X * W1 + b1
        # This is the matrix multiplication of inputs and weights plus bias
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function to get hidden layer activations: a1 = sigmoid(z1)
        self.a1 = self.sigmoid(self.z1)
        
        # Linear transformation for output layer: z2 = a1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Apply sigmoid to get output: a2 = sigmoid(z2)
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward pass - calculating gradients and updating weights
        
        Parameters:
        - X: input data (batch_size, input_size)
        - y: true labels (batch_size, output_size)
        - output: predicted outputs (batch_size, output_size)
        
        Returns:
        - loss: mean squared error loss
        """
        # Calculate the error/loss
        loss = np.mean(np.square(y - output))
        
        # Backpropagation
        # Calculate error at output layer
        # delta2 = (y - output) * sigmoid_derivative(output)
        # This represents the error signal at the output layer
        delta2 = (y - output) * self.sigmoid_derivative(output)
        
        # Propagate error back to hidden layer
        # delta1 = (delta2 * W2.T) * sigmoid_derivative(a1)
        # This shows how much each hidden neuron contributed to the output error
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        
        # Calculate gradients
        # dW2 = a1.T * delta2 (contribution of each hidden neuron to the error)
        dW2 = np.dot(self.a1.T, delta2)
        
        # db2 = sum of error signals (how much to adjust the bias)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # dW1 = X.T * delta1 (contribution of each input to the hidden layer error)
        dW1 = np.dot(X.T, delta1)
        
        # db1 = sum of hidden layer error signals
        db1 = np.sum(delta1, axis=0)
        
        # Update weights and biases
        # W = W + learning_rate * dW (gradient ascent to maximize accuracy)
        learning_rate = 0.1
        self.W1 += learning_rate * dW1
        self.b1 += learning_rate * db1
        self.W2 += learning_rate * dW2
        self.b2 += learning_rate * db2
        
        return loss
    
    def train(self, X, y, epochs=10000):
        """
        Train the neural network for a given number of epochs
        
        Parameters:
        - X: training data (samples, features)
        - y: target values (samples, 1)
        - epochs: number of training iterations
        
        Returns:
        - loss_history: list of losses during training
        """
        loss_history = []
        
        for i in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass and update weights
            loss = self.backward(X, y, output)
            
            # Record loss every 100 epochs
            if i % 100 == 0:
                loss_history.append(loss)
                if i % 1000 == 0:
                    print(f"Epoch {i}, Loss: {loss}")
        
        return loss_history
    
    def predict(self, X):
        """
        Make binary predictions
        
        Parameters:
        - X: input data
        
        Returns:
        - Binary predictions (0 or 1)
        """
        # Forward pass
        output = self.forward(X)
        
        # Convert probabilities to binary predictions
        predictions = (output > 0.5).astype(int)
        
        return predictions

# Create a sample dataset for apple vs orange classification
# Let's use two features: weight (normalized) and texture (where 0 is smooth, 1 is rough)
def create_sample_data():
    # Generate some example data
    # Format: [weight, texture]
    # Apples tend to be lighter and smoother (lower values)
    # Oranges tend to be heavier and rougher (higher values)
    
    # Apple samples (label 0)
    apples = np.array([
        [0.1, 0.2],  # Small and very smooth
        [0.2, 0.3],
        [0.25, 0.25],
        [0.3, 0.3],
        [0.35, 0.35],
        [0.4, 0.4],  # Medium and moderately smooth
    ])
    
    # Orange samples (label 1)
    oranges = np.array([
        [0.5, 0.6],
        [0.6, 0.7],
        [0.65, 0.75],
        [0.7, 0.8],
        [0.8, 0.8],
        [0.9, 0.9],  # Large and very rough
    ])
    
    # Combine the data
    X = np.vstack((apples, oranges))
    
    # Create labels (0 for apple, 1 for orange)
    y = np.vstack((
        np.zeros((len(apples), 1)),  # Apples labeled as 0
        np.ones((len(oranges), 1))   # Oranges labeled as 1
    ))
    
    return X, y

def visualize_data(X, y, predictions=None):
    """
    Visualize the data points and decision boundary
    
    Parameters:
    - X: feature data
    - y: true labels
    - predictions: model predictions (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot apples (class 0)
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], 
                color='red', label='Apple', marker='o')
    
    # Plot oranges (class 1)
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], 
                color='orange', label='Orange', marker='s')
    
    # If we have predictions, highlight incorrect ones
    if predictions is not None:
        incorrect = (predictions.flatten() != y.flatten())
        if np.any(incorrect):
            plt.scatter(X[incorrect, 0], X[incorrect, 1], 
                        color='black', label='Incorrect', marker='x', s=100)
    
    plt.xlabel('Weight (normalized)')
    plt.ylabel('Texture (0=smooth, 1=rough)')
    plt.title('Apple vs Orange Classification')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_decision_boundary(X, y, model):
    """
    Visualize the decision boundary of the model
    
    Parameters:
    - X: feature data
    - y: true labels
    - model: trained neural network model
    """
    plt.figure(figsize=(10, 6))
    
    # Plot apples (class 0)
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], 
                color='red', label='Apple', marker='o')
    
    # Plot oranges (class 1)
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], 
                color='orange', label='Orange', marker='s')
    
    # Create a mesh grid to visualize the decision boundary
    h = 0.01  # step size
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions for the entire grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    plt.xlabel('Weight (normalized)')
    plt.ylabel('Texture (0=smooth, 1=rough)')
    plt.title('Decision Boundary - Apple vs Orange')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_loss(loss_history):
    """
    Visualize the training loss over epochs
    
    Parameters:
    - loss_history: list of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(loss_history) * 100, 100), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create example data
    X, y = create_sample_data()
    
    # Visualize the data
    visualize_data(X, y)
    
    # Create and train the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Train the model
    loss_history = nn.train(X, y, epochs=5000)
    
    # Visualize the training loss
    visualize_loss(loss_history)
    
    # Make predictions
    predictions = nn.predict(X)
    
    # Visualize results with incorrect predictions highlighted
    visualize_data(X, y, predictions)
    
    # Visualize the decision boundary
    visualize_decision_boundary(X, y, nn)
    
    # Print final model accuracy
    accuracy = np.mean((predictions == y).astype(int))
    print(f"Final model accuracy: {accuracy * 100:.2f}%")
    
    # Show the internal weights
    print("\nFinal weights and biases:")
    print("W1 (input to hidden):")
    print(nn.W1)
    print("\nb1 (hidden bias):")
    print(nn.b1)
    print("\nW2 (hidden to output):")
    print(nn.W2)
    print("\nb2 (output bias):")
    print(nn.b2)