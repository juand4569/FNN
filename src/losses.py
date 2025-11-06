import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        """Gradiente de la loss respecto a y_pred"""
        raise NotImplementedError

class MSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        return 2 * (y_pred - y_true) / m

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred_clipped)) / m
    
    def backward(self, y_pred, y_true):
        """Solo se usa si NO hay softmax en la última capa"""
        m = y_true.shape[0]
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred_clipped) / m