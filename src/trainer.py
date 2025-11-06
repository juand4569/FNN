import numpy as np
from .utils import create_mini_batches, accuracy

class Trainer:
    def __init__(self, model, optimizer, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn else model.loss_fn
        
        # Historial
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=True):
        """
        Entrena el modelo con mini-batches
        """
        for epoch in range(1, epochs + 1):
            # Training
            train_losses = []
            train_accs = []
            
            for X_batch, y_batch in create_mini_batches(X_train, y_train, 
                                                         batch_size, shuffle=True):
                # Forward
                y_pred = self.model.forward(X_batch)
                loss = self.model.compute_loss(y_pred, y_batch)
                acc = accuracy(y_pred, y_batch)
                
                # Backward
                self.model.backward(y_batch)
                
                # Update
                params = self.model.get_params()
                grads = self.model.get_grads()
                self.optimizer.update(params, grads)
                
                train_losses.append(loss)
                train_accs.append(acc)
            
            # Promedios de época
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = np.mean(train_accs)
            
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            
            # Validación
            if X_val is not None and y_val is not None:
                val_pred = self.model.forward(X_val)
                val_loss = self.model.compute_loss(val_pred, y_val)
                val_acc = accuracy(val_pred, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"loss: {epoch_train_loss:.4f} - "
                          f"acc: {epoch_train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - "
                          f"val_acc: {val_acc:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"loss: {epoch_train_loss:.4f} - "
                          f"acc: {epoch_train_acc:.4f}")
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo en test set"""
        y_pred = self.model.forward(X_test)
        test_loss = self.model.compute_loss(y_pred, y_test)
        test_acc = accuracy(y_pred, y_test)
        
        return test_loss, test_acc
    
    def plot_history(self):
        """Grafica curvas de entrenamiento"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Loss Evolution')
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        if self.history['val_acc']:
            ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_title('Accuracy Evolution')
        
        plt.tight_layout()
        plt.show()