import numpy as np
import gzip
import os
from urllib.request import urlretrieve

def download_mnist(data_dir='../data'):
    """Descarga MNIST si no existe"""
    files = {
        'train_images': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz',
        'train_labels': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz',
        'test_images': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz'
    }

    os.makedirs(data_dir, exist_ok=True)

    for key, url in files.items():
        filename = os.path.basename(url)
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(url, filepath)

def load_mnist(data_dir='../data', normalize=True):
    """Carga MNIST desde archivos descargados"""
    download_mnist(data_dir)

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = read_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = read_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    X_test = read_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = read_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test
