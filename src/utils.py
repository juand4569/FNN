import numpy as np

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, 
                         test_ratio=0.15, shuffle=True, random_seed=None):
    """
    Divide dataset en train/val/test
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

def one_hot_encode(y, num_classes=None):
    """Convierte etiquetas a one-hot encoding"""
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def accuracy(y_pred, y_true):
    """Calcula accuracy para clasificación"""
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)

def create_mini_batches(X, y, batch_size, shuffle=True):
    """
    Genera mini-batches
    
    Yields:
        (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]