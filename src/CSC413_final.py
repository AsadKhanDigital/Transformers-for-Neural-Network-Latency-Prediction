import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json
# import torch


def generate_standard_nn(depth, hidden_units):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(hidden_units[0],)))
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))
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
        input_data = np.random.rand(1, input_shape[0])
        start_time = time.time()
        with tf.device(device):
            _ = model.predict(input_data)
        end_time = time.time()
        latencies.append(end_time - start_time)
    median_latency = np.median(latencies)
    return median_latency


def encode_for_transformer(adjacency_matrix, node_attributes, max_length=10):
    """
    Encode graph features for Transformer input
    
    Args:
    - adjacency_matrix: 2D list representing connections between layers
    - node_attributes: List of dictionaries with layer attributes
    - max_length: Maximum number of layers to consider
    
    Returns:
    - Encoded input suitable for Transformer model
    """
    # 1. Adjacency Matrix Encoding
    # Flatten and pad/truncate the adjacency matrix
    adj_flat = [val for row in adjacency_matrix for val in row]
    adj_flat = adj_flat[:max_length**2] + [0] * (max_length**2 - len(adj_flat))
    
    # 2. Node Attributes Encoding
    node_encodings = []
    for node in node_attributes[:max_length]:
        # Encode layer type
        type_mapping = {
            'InputLayer': 0,
            'Dense': 1,
            'Conv2D': 2,
            # Add more layer types as needed
        }
        type_encoding = type_mapping.get(node['type'], -1)
        
        # Normalize weights (log scale to handle large variations)
        weights_encoding = np.log(node['weights'] + 1) / 10.0  # Normalize log of weights
        
        # Create multi-dimensional encoding
        node_encodings.append([
            type_encoding,  # Layer type
            weights_encoding,  # Normalized weights
            node.get('activation', 0)  # Activation type (if available)
        ])
    
    # Pad node encodings
    while len(node_encodings) < max_length:
        node_encodings.append([0, 0, 0])
    
    # Flatten node encodings
    node_enc_flat = [item for sublist in node_encodings for item in sublist]
    
    # 3. Positional Encoding
    # Create simple positional encoding based on layer index
    positional_encoding = [i / (max_length - 1) for i in range(max_length)]
    
    # 4. Combine all encodings
    combined_encoding = adj_flat + node_enc_flat + positional_encoding
    
    return combined_encoding


# Dataset generation
def generate_datasets(num_standard_nns=10):
    dataset = []

    for _ in range(num_standard_nns):
        depth = random.randint(3, 10)
        hidden_units = [2**random.randint(0, 10) for _ in range(depth)]
        model = generate_standard_nn(depth, hidden_units)
        
        # Get graph features
        adjacency_matrix, node_attributes = get_graph_features(model)
        
        # Measure latencies
        cpu_latency = measure_latency(model, (hidden_units[0],), '/CPU:0')
        gpu_latency = measure_latency(model, (hidden_units[0],), '/GPU:0')
        
        # Encode for Transformer
        transformer_input = encode_for_transformer(adjacency_matrix, node_attributes)
        
        dataset.append({
            'type': 'standard_nn',
            'adjacency_matrix': adjacency_matrix,
            'node_attributes': node_attributes,
            'cpu_latency': cpu_latency,
            'gpu_latency': gpu_latency,
            'transformer_input': transformer_input
        })

    return dataset


def main():
    # Generate dataset
    dataset = generate_datasets()

    # Save dataset
    with open('latency_dataset.json', 'w') as f:
        json.dump(dataset, f)

    print(f"Generated dataset with {len(dataset)} neural network samples")
    

if __name__ == '__main__':
    main()