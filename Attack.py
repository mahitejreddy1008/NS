import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Generate a network with power-law degree distribution (Configuration model)
def generate_powerlaw_network(N, gamma):
    while True:
        degree_sequence = np.random.zipf(gamma, N)
        # Ensure the sum of degrees is even
        if sum(degree_sequence) % 2 == 0:
            break
    G = nx.configuration_model(degree_sequence)
    G = nx.Graph(G)  # Remove multi-edges
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    return G

# Attack the network based on a specific metric and fraction of nodes removed
def attack_network(G, metric, fraction_remove):
    N = len(G)
    num_nodes_to_remove = int(fraction_remove * N)

    if metric == 'degree':
        # Remove nodes with the highest degree
        nodes_sorted_by_metric = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    elif metric == 'clustering':
        # Remove nodes with the highest clustering coefficient
        clustering_coefficients = nx.clustering(G)
        nodes_sorted_by_metric = sorted(clustering_coefficients.items(), key=lambda x: x[1], reverse=True)

    nodes_to_remove = [node for node, value in nodes_sorted_by_metric[:num_nodes_to_remove]]
    G.remove_nodes_from(nodes_to_remove)

    # Check if there are any connected components left
    if len(G) == 0 or len(list(nx.connected_components(G))) == 0:
        return 0  # No components left

    # Get the size of the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc)

# Visualize the network
def visualize_network(G, title):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # For better layout and consistent positioning
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='blue')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(title)
    plt.show()

# Run the attack simulation and visualize the network
def conspiracy_simulation(N, gamma):
    fractions_removed = np.linspace(0, 1, 20)  # Fractions of nodes to remove
    G = generate_powerlaw_network(N, gamma)

    # Visualize the original network
    visualize_network(G, "Original Network")

    # Simulate degree-based and clustering coefficient-based attacks
    largest_components_degree = []
    largest_components_clustering = []

    for fraction in fractions_removed:
        G_copy1 = G.copy()
        G_copy2 = G.copy()

        size_giant_degree = attack_network(G_copy1, 'degree', fraction)
        size_giant_clustering = attack_network(G_copy2, 'clustering', fraction)

        largest_components_degree.append(size_giant_degree)
        largest_components_clustering.append(size_giant_clustering)

        # Visualize the network after 20% of nodes have been removed
        if fraction == 0.2:  # Visualizing at 20% removal
            visualize_network(G_copy1, "Network after 20% Degree-based Attack")
            visualize_network(G_copy2, "Network after 20% Clustering-based Attack")

    # Plot the results
    plt.plot(fractions_removed, largest_components_degree, label='Degree-based attack')
    plt.plot(fractions_removed, largest_components_clustering, label='Clustering-based attack')
    plt.xlabel('Fraction of nodes removed')
    plt.ylabel('Size of largest component')
    plt.legend()
    plt.title('Attack on Social Network: Degree vs Clustering Coefficient')
    plt.show()

# Example usage
N = 500  # Reduced for better visualization
gamma = 2.5
conspiracy_simulation(N, gamma)







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
