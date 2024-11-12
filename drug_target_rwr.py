import numpy as np
import networkx as nx

# Load drug-target interaction matrix
# For simplicity, using a random adjacency matrix. Replace with your actual drug-target data.
def load_data():
    # Suppose we have 5 drugs and 5 targets
    np.random.seed(42)
    adjacency_matrix = np.random.rand(10, 10)  # 5 drugs + 5 targets, adjacency matrix
    return adjacency_matrix

# Normalize adjacency matrix (Row Normalization)
def normalize_matrix(adjacency_matrix):
    row_sums = adjacency_matrix.sum(axis=1)
    normalized_matrix = adjacency_matrix / row_sums[:, np.newaxis]
    return normalized_matrix

# Random Walk with Restart (RWR)
def random_walk_with_restart(adj_matrix, restart_prob=0.3, max_iter=100, tol=1e-6):
    n = adj_matrix.shape[0]
    initial_prob = np.zeros(n)
    initial_prob[0] = 1  # Start with the first drug (or target) node
    
    prob_vector = initial_prob.copy()
    
    for i in range(max_iter):
        new_prob_vector = (1 - restart_prob) * np.dot(adj_matrix, prob_vector) + restart_prob * initial_prob
        
        # Convergence check
        if np.linalg.norm(new_prob_vector - prob_vector) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        prob_vector = new_prob_vector
    
    return prob_vector

# Load the adjacency matrix
adj_matrix = load_data()

# Normalize the adjacency matrix
normalized_matrix = normalize_matrix(adj_matrix)

# Run Random Walk with Restart
result = random_walk_with_restart(normalized_matrix)

# Get top predicted interactions (for simplicity, returning indices with highest scores)
top_k = 5
top_indices = np.argsort(result)[-top_k:]

print(f"Top {top_k} predicted drug-target interactions (node indices): {top_indices}")
