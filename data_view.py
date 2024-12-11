import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    # Load the .npz file
    data = np.load(args.data_file)
    adjacency = data['adjacency']  # shape: (N, MAX_DEPTH, MAX_DEPTH)
    attributes = data['attributes']  # shape: (N, MAX_DEPTH, ATTR_DIM)
    latencies = data['latencies']  # shape: (N,)

    # Print basic info
    print(f"Loaded data from {args.data_file}")
    print(f"Number of samples: {len(latencies)}")
    print(f"Adjacency shape: {adjacency.shape}")
    print(f"Attributes shape: {attributes.shape}")
    print(f"Latencies shape: {latencies.shape}")

    # Print summary statistics for latencies
    print("Latency stats:")
    print(f"  Min: {latencies.min()}")
    print(f"  Max: {latencies.max()}")
    print(f"  Mean: {latencies.mean()}")
    print(f"  Median: {np.median(latencies)}")
    print(f"  Std: {latencies.std()}")

    # Show a histogram of latencies
    plt.figure(figsize=(6,4))
    plt.hist(latencies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Latency Distribution')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    if args.show_plots:
        plt.show()
    else:
        plt.savefig("latency_histogram.png", dpi=150)
        print("Saved latency_histogram.png")

    # Print some sample attributes
    sample_idx = 0
    if args.sample_index is not None:
        sample_idx = args.sample_index
    else:
        sample_idx = np.random.randint(0, len(latencies))

    print(f"\nDisplaying sample index: {sample_idx}")
    print(f"Latency: {latencies[sample_idx]}")
    print("Attributes (first few rows):")
    print(attributes[sample_idx, :5])  # print first 5 nodes' attributes

    # Visualize adjacency matrix for the sample
    plt.figure(figsize=(6,6))
    plt.imshow(adjacency[sample_idx], cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Connection')
    plt.title(f'Adjacency Matrix (Sample {sample_idx})')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.tight_layout()
    if args.show_plots:
        plt.show()
    else:
        plt.savefig("adjacency_matrix_sample.png", dpi=150)
        print("Saved adjacency_matrix_sample.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View and visualize NPZ dataset.")
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the .npz file containing adjacency, attributes, and latencies.')
    parser.add_argument('--show_plots', action='store_true',
                        help='If set, display plots interactively instead of saving them.')
    parser.add_argument('--sample_index', type=int, default=None,
                        help='Specify a particular sample index to visualize. If not set, a random sample will be chosen.')
    args = parser.parse_args()
    main(args)
