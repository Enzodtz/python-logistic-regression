import numpy as np
from tqdm import tqdm

class LogisticRegressionModel:

    def __init__(self):
        pass

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        """
        
        self.w = np.zeros((dim,1))
        self.b = 0

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

    def feed_forward(self, X):
        """
        This function takes the input data and maps it into Y output values
        """
        return self.sigmoid(np.dot(self.w.T, X) + self.b)

    def loss_function(self, Y, A):
        """
        Implements loss function given Y (right value)
        and A (predicted value)
        """
        return Y * np.log(A) + (1 - Y) * np.log(1 - A) 

    def cost_function(self, Y, A, m):
        """
        Implements cost function given Y (right value)
        and A (predicted value)
        """
        return - np.sum(self.loss_function(Y, A)) / m
    
    def gradient_w(self, X, Y, A, m):
        """
        Implements the gradient for the weights given 
        the Y (right value) and A (predicted value)
        """
        return  np.dot(X, (A - Y).T) / m 

    def gradient_b(self, Y, A, m):
        """
        Implements the gradient for the biases given 
        the Y (right value) and A (predicted value)
        """
        return np.sum(A - Y) / m

    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient
        """
        
        m = X.shape[1]
        
        # Forward Propagation
        A = self.feed_forward(X)
        cost = self.cost_function(Y, A, m)
        
        # Backpropagation
        dw = self.gradient_w(X, Y, A, m)
        db = self.gradient_b(Y, A, m)

        return dw, db, cost

    def optimize(self, X, Y, num_iterations, learning_rate):
        """
        This function optimizes weights and biases by applying
        the gradient descend to them, according to the learning
        rate
        """
        
        # Initialize costs variable
        if not hasattr(self, 'costs'):
            self.costs = []

        for i in tqdm(range(num_iterations)):
            # Get gradients and cost from propagation
            dw, db, cost = self.propagate(X, Y)

            # Apply gradient backwards, according to the learning rate, also named as step size
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
            
            # Record the costs
            # This is only made for statistical and review metrics, since this is not used in any calculation
            self.costs.append(cost)
        
        return dw, db

    def predict(self, X):
        '''
        Given an X value, predicts its Y value
        '''
        
        assert X.shape[0] == self.w.shape[0]

        # Get probabilities that a cat is present on the image
        A = self.feed_forward(X)

        # Map probabilities to True/False classifications
        Y_prediction = A >= 0.5
        
        return Y_prediction

    def train(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005):
        """
        Train the model with the specified iterations and learning rate
        """
        
        # If weights and biases weren't inicialized
        if not hasattr(self, 'w'):
            self.initialize_with_zeros(X_train.shape[0])
        else:
            assert self.w.shape[0] == X_train.shape[0]

        # Optimize weights and biases
        dw, db = self.optimize(X_train, Y_train, num_iterations, learning_rate)

        # Predict test/train set examples
        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X_train)

        # Get train/test Errors
        train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

        # Return model information
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
