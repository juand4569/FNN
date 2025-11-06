import numpy as np

class Optimizer:
    """Clase base para optimizadores"""
    def update(self, params, grads):
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None  # Se inicializa en primer update
        self.v = None
    
    def update(self, params, grads):
        """
        params: lista de [W1, b1, W2, b2, ...]
        grads: lista de [dW1, db1, dW2, db2, ...]
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i in range(len(params)):
            # Momento primer orden
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # Momento segundo orden
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            
            # Corrección de sesgo
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Actualización
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]