import os
import numpy as np
import tensorflow as tf
import random
import time
import gc
from tensorflow.keras import backend as K

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

TOTAL_NETWORKS = 320000
MIN_DEPTH = 3
MAX_DEPTH = 10
INPUT_DIM = 128
OUTPUT_DIM = 10
HIDDEN_UNITS_POW_RANGE = (0, 14)
NUM_LATENCY_RUNS = 5
SAVE_BATCH_SIZE = 10000

DATA_SAVE_DIR = "./network_dataset"
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

ATTR_DIM = 3

def build_random_network():
    depth = random.randint(MIN_DEPTH, MAX_DEPTH)
    hidden_layers = depth - 2

    inputs = tf.keras.Input(shape=(INPUT_DIM,), name="InputLayer")
    x = inputs
    layer_units = []

    for i in range(hidden_layers):
        units = 2**random.randint(*HIDDEN_UNITS_POW_RANGE)
        x = tf.keras.layers.Dense(units, activation='relu', name=f"Dense_{i}")(x)
        layer_units.append(units)

    x = tf.keras.layers.Dense(OUTPUT_DIM, name="OutputLayer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model, layer_units

def get_graph_representation(model, layer_units):
    layers = model.layers
    num_nodes = len(layers)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes - 1):
        adjacency[i, i+1] = 1.0

    attributes = []
    for l in layers:
        if 'InputLayer' in l.__class__.__name__:
            op_type = [1,0]
            num_weights = 0
        elif isinstance(l, tf.keras.layers.Dense):
            op_type = [0,1]
            w_shape = l.weights[0].shape
            num_weights = w_shape[0]*w_shape[1] + w_shape[1]
        else:
            op_type = [0,1]
            num_weights = 0 if not l.weights else np.sum([np.prod(w.shape) for w in l.weights])
        attributes.append(op_type + [num_weights])

    attributes = np.array(attributes, dtype=np.float32)
    return adjacency, attributes

def pad_graph_representation(adjacency, attributes, max_depth=MAX_DEPTH, attr_dim=ATTR_DIM):
    current_depth = adjacency.shape[0]
    padded_adj = np.zeros((max_depth, max_depth), dtype=np.float32)
    padded_adj[:current_depth, :current_depth] = adjacency

    padded_attr = np.zeros((max_depth, attr_dim), dtype=np.float32)
    padded_attr[:current_depth, :] = attributes

    return padded_adj, padded_attr

def measure_inference_latency(model):
    dummy_input = np.random.randn(1, INPUT_DIM).astype(np.float32)
    for _ in range(5):
        model.predict(dummy_input, verbose=0)

    times = []
    for _ in range(NUM_LATENCY_RUNS):
        start = time.perf_counter()
        model.predict(dummy_input, verbose=0)
        end = time.perf_counter()
        times.append(end - start)
    median_latency = np.median(times)
    return median_latency

def generate_dataset(total_networks=320000, save_dir=DATA_SAVE_DIR, save_batch_size=10000):
    start_time = time.time()

    adjacency_list = []
    attributes_list = []
    latencies = []

    for i in range(total_networks):
        model, layer_units = build_random_network()
        adj, attr = get_graph_representation(model, layer_units)
        adj, attr = pad_graph_representation(adj, attr, max_depth=MAX_DEPTH, attr_dim=ATTR_DIM)

        latency = measure_inference_latency(model)

        adjacency_list.append(adj)
        attributes_list.append(attr)
        latencies.append(latency)

        K.clear_session()
        gc.collect()

        if (i+1) % save_batch_size == 0:
            batch_id = (i+1) // save_batch_size
            file_path = os.path.join(save_dir, f"dataset_part_{batch_id}.npz")
            np.savez_compressed(
                file_path,
                adjacency=np.array(adjacency_list, dtype=np.float32),
                attributes=np.array(attributes_list, dtype=np.float32),
                latencies=np.array(latencies, dtype=np.float32)
            )

            adjacency_list = []
            attributes_list = []
            latencies = []

            K.clear_session()
            gc.collect()

            print(f"Saved batch {batch_id}, {i+1} networks processed.")

    if adjacency_list:
        batch_id = total_networks // save_batch_size + 1
        file_path = os.path.join(save_dir, f"dataset_part_{batch_id}.npz")
        np.savez_compressed(
            file_path,
            adjacency=np.array(adjacency_list, dtype=np.float32),
            attributes=np.array(attributes_list, dtype=np.float32),
            latencies=np.array(latencies, dtype=np.float32)
        )

        K.clear_session()
        gc.collect()
        print(f"Saved final batch {batch_id}.")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total time taken to generate the dataset: {total_duration:.2f} seconds")

generate_dataset(total_networks=320000, save_batch_size=10000)
