import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Avalanche simulation function with network visualization
def simulate_avalanches(G, num_steps=10000, grain_loss=1e-4, visualize_step=2000):
    # Initialize bucket sizes (equal to node degrees)
    bucket_sizes = {node: G.degree(node) for node in G.nodes()}
    sand_in_buckets = {node: 0 for node in G.nodes()}
    avalanche_sizes = []

    for step in range(num_steps):
        # Add a grain to a random node
        node = random.choice(list(G.nodes()))
        sand_in_buckets[node] += 1

        # Perform topplings
        avalanche_size = 0
        unstable_nodes = [node]

        while unstable_nodes:
            current_node = unstable_nodes.pop()
            if sand_in_buckets[current_node] >= bucket_sizes[current_node]:
                # Node topples
                avalanche_size += 1
                excess_grains = sand_in_buckets[current_node]
                sand_in_buckets[current_node] = 0  # Reset bucket

                # Distribute grains to neighbors
                for neighbor in G.neighbors(current_node):
                    sand_in_buckets[neighbor] += (1 - grain_loss)
                    if sand_in_buckets[neighbor] >= bucket_sizes[neighbor]:
                        unstable_nodes.append(neighbor)

        avalanche_sizes.append(avalanche_size)

        # Visualize the network at the specified step
        if step == visualize_step:
            visualize_network(G, sand_in_buckets, f"Network at Step {visualize_step} (Avalanche)")

    return avalanche_sizes

# Plot avalanche distribution
def plot_avalanche_distribution(avalanche_sizes, title):
    unique_sizes, counts = np.unique(avalanche_sizes, return_counts=True)
    probabilities = counts / np.sum(counts)

    plt.loglog(unique_sizes, probabilities, marker='o', linestyle='none')
    plt.xlabel('Avalanche Size (s)')
    plt.ylabel('P(s)')
    plt.title(title)
    plt.show()

# Generate random network (Erdős-Rényi)
def generate_erdos_renyi(N, avg_k):
    p = avg_k / (N - 1)
    G = nx.erdos_renyi_graph(N, p)
    return G

# Generate scale-free network
def generate_scale_free_network(N, gamma):
    # Generate a Zipf degree sequence
    degree_sequence = np.random.zipf(gamma, N)

    # Ensure that the sum of the degree sequence is even
    if sum(degree_sequence) % 2 != 0:
        # Adjust by adding 1 to a random node's degree
        degree_sequence[random.randint(0, N - 1)] += 1

    # Generate the configuration model graph
    G = nx.configuration_model(degree_sequence)
    G = nx.Graph(G)  # Convert to simple graph
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    return G

# Visualize the network
def visualize_network(G, sand_in_buckets=None, title="Network Visualization"):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    node_colors = 'blue'

    if sand_in_buckets:
        node_colors = [sand_in_buckets[node] for node in G.nodes()]
        nx.draw(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.Blues, with_labels=False)
    else:
        nx.draw(G, pos, node_size=50, node_color=node_colors, with_labels=False)

    plt.title(title)
    plt.show()

# Example usage for both types of networks
N = 500
avg_k = 2

# Erdős-Rényi network
G_er = generate_erdos_renyi(N, avg_k)
visualize_network(G_er, title="Erdős-Rényi Network (Before Avalanche)")
avalanche_sizes_er = simulate_avalanches(G_er, visualize_step=2000)
plot_avalanche_distribution(avalanche_sizes_er, "Avalanche Size Distribution (Erdős-Rényi)")

# Scale-free network (Configuration model)
G_sf = generate_scale_free_network(N, gamma=2.5)
visualize_network(G_sf, title="Scale-Free Network (Before Avalanche)")
avalanche_sizes_sf = simulate_avalanches(G_sf, visualize_step=2000)
plot_avalanche_distribution(avalanche_sizes_sf, "Avalanche Size Distribution (Scale-Free)")
