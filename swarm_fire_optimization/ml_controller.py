import numpy as np


class SimpleNN:
    """
    Simple 1-hidden-layer neural network (NumPy only)
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        self.W1 = 0.1 * np.random.randn(in_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = 0.1 * np.random.randn(hidden_dim, out_dim)
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.H = np.tanh(self.Z1)
        return self.H @ self.W2 + self.b2

    def train(self, X, Y, lr=1e-3, epochs=2000):
        N = X.shape[0]

        for _ in range(epochs):
            Y_hat = self.forward(X)
            dY = 2 * (Y_hat - Y) / N

            dW2 = self.H.T @ dY
            db2 = dY.sum(axis=0)

            dH = dY @ self.W2.T
            dZ1 = dH * (1 - np.tanh(self.Z1) ** 2)

            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
