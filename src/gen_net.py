import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json

def generate_standard_nn(depth, hidden_units):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(hidden_units[0],)))
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))
    return model

def generate_cnn(depth, channel_sizes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))
    for channels in channel_sizes:
        model.add(layers.Conv2D(channels, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def get_graph_features(model):
    layers = model.layers
    adjacency_matrix = np.zeros((len(layers), len(layers)))
    for i in range(len(layers)-1):
        adjacency_matrix[i][i+1] = 1
    node_attributes = []
    for layer in layers:
        layer_type = type(layer).__name__
        weights = layer.count_params()
        node_attributes.append({
            'type': layer_type,
            'weights': weights
        })
    return adjacency_matrix.tolist(), node_attributes

def measure_latency(model, input_shape, device):
    latencies = []
    for _ in range(5):
        if 'Conv2D' in [layer.__class__.__name__ for layer in model.layers]:
            input_data = np.random.rand(1, *input_shape)
        else:
            input_data = np.random.rand(1, input_shape[0])
        start_time = time.time()
        with tf.device(device):
            _ = model.predict(input_data)
        end_time = time.time()
        latencies.append(end_time - start_time)
    median_latency = np.median(latencies)
    return median_latency

dataset = []

num_standard_nns = 1000
num_cnns = 1000

# Generate standard NNs
for _ in range(num_standard_nns):
    depth = random.randint(3, 10)
    hidden_units = [2**random.randint(0, 10) for _ in range(depth)]
    model = generate_standard_nn(depth, hidden_units)
    adjacency_matrix, node_attributes = get_graph_features(model)
    cpu_latency = measure_latency(model, (hidden_units[0],), '/CPU:0')
    gpu_latency = measure_latency(model, (hidden_units[0],), '/GPU:0')
    dataset.append({
        'type': 'standard_nn',
        'adjacency_matrix': adjacency_matrix,
        'node_attributes': node_attributes,
        'cpu_latency': cpu_latency,
        'gpu_latency': gpu_latency
    })

for _ in range(num_cnns):
    depth = random.randint(3, 10)
    channel_sizes = [2**random.randint(0, 8) for _ in range(depth)]
    model = generate_cnn(depth, channel_sizes)
    adjacency_matrix, node_attributes = get_graph_features(model)
    cpu_latency = measure_latency(model, (32, 32, 3), '/CPU:0')
    gpu_latency = measure_latency(model, (32, 32, 3), '/GPU:0')
    dataset.append({
        'type': 'cnn',
        'adjacency_matrix': adjacency_matrix,
        'node_attributes': node_attributes,
        'cpu_latency': cpu_latency,
        'gpu_latency': gpu_latency
    })

with open('latency_dataset.json', 'w') as f:
    json.dump(dataset, f)