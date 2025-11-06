import numpy as np
from .layers import Dense
from .activations import Sigmoid, Softmax
from .losses import CategoricalCrossentropy

class NeuralNetwork:
    def __init__(self, architecture, loss='categorical_crossentropy'):
        """
        architecture: lista de tuplas [(n_neurons, activation), ...]
        Ejemplo: [(64, 'sigmoid'), (32, 'sigmoid'), (10, 'softmax')]
        """
        self.layers = []
        self.loss_fn = self._get_loss(loss)
        
        # Construir capas
        for i in range(len(architecture)):
            n_neurons, activation = architecture[i]
            if i == 0:
                continue  # Primer elemento es input, se define en add_layer
            
            n_inputs = architecture[i-1][0]
            act = self._get_activation(activation)
            
            layer = Dense(n_inputs, n_neurons, activation=act)
            self.layers.append(layer)
    
    def _get_activation(self, name):
        from .activations import Sigmoid, Tanh, ReLU, Softmax
        activations = {
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'relu': ReLU(),
            'softmax': Softmax()
        }
        return activations.get(name, Sigmoid())
    
    def _get_loss(self, name):
        from .losses import MSE, CategoricalCrossentropy
        losses = {
            'mse': MSE(),
            'categorical_crossentropy': CategoricalCrossentropy()
        }
        return losses.get(name, CategoricalCrossentropy())
    
    def forward(self, X):
        """Forward pass por todas las capas"""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self, y_true):
        """Backward pass"""
        y_pred = self.layers[-1].a
        m = y_true.shape[0]
        
        # Para softmax + categorical crossentropy, el gradiente es simplemente:
        if isinstance(self.layers[-1].activation, Softmax):
            grad = (y_pred - y_true) / m
            # Marcar que la última capa NO debe multiplicar por derivada de activación
            self.layers[-1].skip_activation_grad = True
        else:
            grad = self.loss_fn.backward(y_pred, y_true)
            self.layers[-1].skip_activation_grad = False
        
        # Backprop por todas las capas
        for i, layer in enumerate(reversed(self.layers)):
            if i > 0:  # Capas ocultas SÍ usan derivada de activación
                layer.skip_activation_grad = False
            grad = layer.backward(grad)
    
    def get_params(self):
        """Retorna lista plana [W1, b1, W2, b2, ...]"""
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def get_grads(self):
        """Retorna lista plana [dW1, db1, dW2, db2, ...]"""
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads
    
    def compute_loss(self, y_pred, y_true):
        return self.loss_fn.forward(y_pred, y_true)