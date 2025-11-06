import numpy as np
from .activations import Sigmoid, Tanh, ReLU, Softmax

__all__ = ['Activation', 'Sigmoid', 'Tanh', 'ReLU', 'Softmax']

class Activation:
    """Clase base para funciones de activación"""
    def forward(self, z):
        raise NotImplementedError
    
    def backward(self, a):
        """Derivada respecto a z (no a a)"""
        raise NotImplementedError

class Sigmoid(Activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def backward(self, a):
        """Derivada: a * (1 - a) donde a = sigmoid(z)"""
        return a * (1 - a)

class Tanh(Activation):
    def forward(self, z):
        return np.tanh(z)
    
    def backward(self, a):
        """Derivada: 1 - a^2 donde a = tanh(z)"""
        return 1 - a**2

class ReLU(Activation):
    def __init__(self):
        self.z = None
    
    def forward(self, z):
        self.z = z  # Guardar z
        return np.maximum(0, z)
    
    def backward(self, grad_output):
        return grad_output * (self.z > 0).astype(float)

class Softmax(Activation):
    def forward(self, z):
        """Softmax estable"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def backward(self, a):
        """Para softmax + cross-entropy, se simplifica en la loss"""
        return a  # placeholder, se maneja en loss