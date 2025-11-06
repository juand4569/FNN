import numpy as np
from .activations import Sigmoid

class Layer:
    """Clase base para capas"""
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation=None, 
                 weight_init='xavier'):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation if activation else Sigmoid()
        
        # Inicialización pesos
        if weight_init == 'xavier':
            limit = np.sqrt(6 / (n_inputs + n_neurons))
            self.W = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif weight_init == 'he':
            self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2/n_inputs)
        else:
            self.W = np.random.randn(n_inputs, n_neurons) * 0.01
        
        self.b = np.zeros((1, n_neurons))
        
        # Cache para backward
        self.X = None
        self.z = None
        self.a = None
    
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, grad_output):
        """
        grad_output: gradiente de loss respecto a la salida de esta capa
        Returns: gradiente respecto a la entrada X
        """
        m = self.X.shape[0]
        
        # Gradiente a través de la activación
        grad_z = grad_output * self.activation.backward(self.a)
        
        # Gradientes de parámetros
        self.dW = np.dot(self.X.T, grad_z) / m
        self.db = np.sum(grad_z, axis=0, keepdims=True) / m
        
        # Gradiente hacia capa anterior
        grad_X = np.dot(grad_z, self.W.T)
        
        return grad_X
    
    def get_params(self):
        return [self.W, self.b]
    
    def get_grads(self):
        return [self.dW, self.db]