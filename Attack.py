
## Control Attacks

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






## Karthic avalanche
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




## NLP Mining error correction
import networkx as nx
import random
import matplotlib.pyplot as plt

# Parameters
N = 10  # Number of nodes
average_degree = 2  # Average degree
high_capacity_factor = 100  # Set a high multiplier for bucket capacity # hi there

# Function to generate an Erdős-Rényi network
def generate_erdos_renyi(N, average_degree):
    p = average_degree / (N - 1)  # Probability for Erdős-Rényi model
    G = nx.erdos_renyi_graph(N, p)
    return G

# Function to generate a scale-free network using Barabási-Albert model
def generate_scale_free(N, average_degree):
    m = max(1, average_degree // 2)  # Parameter for number of edges to attach per new node
    G = nx.barabasi_albert_graph(N, m)
    return G

# Initialize buckets (number of grains in each node)
def initialize_buckets(G):
    return {node: 0 for node in G.nodes()}

# Get the bucket capacity (equal to node degree * high_capacity_factor to prevent frequent toppling) # hi there
def get_bucket_capacity(G):
    return {node: G.degree[node] * high_capacity_factor for node in G.nodes()}  # hi there

# Simulate adding a grain and handling toppling (avalanche)
def simulate_avalanche(G, buckets, capacities):
    avalanche_sizes = []

    def topple(node):
        toppled_nodes = set()  # Track nodes that topple # hi there
        to_process = [node]

        while to_process:
            current_node = to_process.pop()

            # Prevent looping by skipping already toppled nodes in the current step # hi there
            if current_node in toppled_nodes:  # hi there
                continue  # hi there

            if buckets[current_node] >= capacities[current_node]:
                toppled_nodes.add(current_node)  # hi there

                # Topple: Redistribute grains to neighbors
                grains_to_redistribute = buckets[current_node]
                buckets[current_node] = 0  # Reset bucket after toppling

                for neighbor in G.neighbors(current_node):
                    # Redistribute evenly to neighbors # hi there
                    buckets[neighbor] += grains_to_redistribute // G.degree[current_node]  # hi there

                    # If neighbor becomes unstable, it will topple
                    if buckets[neighbor] >= capacities[neighbor] and neighbor not in toppled_nodes:  # hi there
                        to_process.append(neighbor)

        return len(toppled_nodes)  # Avalanche size

    # Randomly choose a node to add a grain
    random_node = random.choice(list(G.nodes()))
    buckets[random_node] += 1

    # Check if it needs to topple
    if buckets[random_node] >= capacities[random_node]:
        avalanche_size = topple(random_node)
        avalanche_sizes.append(avalanche_size)
    else:
        avalanche_sizes.append(0)

    return avalanche_sizes

# Simulation function
def run_simulation(G, steps):
    # Initialize the buckets and capacities
    buckets = initialize_buckets(G)
    capacities = get_bucket_capacity(G)  # hi there

    # Simulate avalanches for a number of steps
    avalanche_sizes = []
    for _ in range(steps):
        avalanche_sizes.extend(simulate_avalanche(G, buckets, capacities))

    return avalanche_sizes

# Generate networks
G_erdos_renyi = generate_erdos_renyi(N, average_degree)
G_scale_free = generate_scale_free(N, average_degree)

# Run simulations
steps = 1000  # Number of time steps

# Erdős-Rényi simulation
avalanche_sizes_erdos = run_simulation(G_erdos_renyi, steps)

# Scale-Free simulation
avalanche_sizes_scale_free = run_simulation(G_scale_free, steps)

# Display results
print("Avalanche sizes (Erdős-Rényi):", avalanche_sizes_erdos)
print("Avalanche sizes (Scale-Free):", avalanche_sizes_scale_free)

# Plot the networks
plt.figure(figsize=(10, 5))

plt.subplot(121)
nx.draw(G_erdos_renyi, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.title("Erdős-Rényi Network")

plt.subplot(122)
nx.draw(G_scale_free, with_labels=True, node_color='lightgreen', node_size=500, edge_color='gray')
plt.title("Scale-Free Network")

plt.show()

## criticl threshold without quad
import numpy as np
from scipy.special import gamma

# Define moments calculation
def calculate_moments(degree_distribution, k_values):
    pk = degree_distribution(k_values)
    k_mean = np.sum(k_values * pk)
    k2_mean = np.sum(k_values**2 * pk)
    return k_mean, k2_mean

# Critical threshold function
def critical_threshold(k_mean, k2_mean):
    return 1 - (k_mean / k2_mean)

# Power Law with Exponential Cutoff
def power_law_exponential_cutoff(k_values, gamma, k_c):
    return k_values**(-gamma) * np.exp(-k_values / k_c)

# Lognormal Distribution
def lognormal_distribution(k_values, mu, sigma):
    return (1 / (k_values * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(k_values) - mu)**2) / (2 * sigma**2))

# Delta Distribution (all nodes have same degree k_0)
def delta_distribution(k_values, k_0):
    return np.where(k_values == k_0, 1.0, 0.0)

# Set parameters and calculate for different distributions
k_values = np.arange(1, 1000, 1)  # Degree values

# a. Power Law with Exponential Cutoff
gamma = 2.5
k_c = 100
k_mean, k2_mean = calculate_moments(lambda k: power_law_exponential_cutoff(k, gamma, k_c), k_values)
fc_powerlaw = critical_threshold(k_mean, k2_mean)

# b. Lognormal Distribution
mu = 2.0
sigma = 0.5
k_mean, k2_mean = calculate_moments(lambda k: lognormal_distribution(k, mu, sigma), k_values)
fc_lognormal = critical_threshold(k_mean, k2_mean)

# c. Delta Distribution
k_0 = 10
k_mean, k2_mean = calculate_moments(lambda k: delta_distribution(k, k_0), k_values)
fc_delta = critical_threshold(k_mean, k2_mean)

# Print results
print("Critical threshold for Power Law with Exponential Cutoff: ", fc_powerlaw)
print("Critical threshold for Lognormal Distribution: ", fc_lognormal)
print("Critical threshold for Delta Distribution: ", fc_delta)

