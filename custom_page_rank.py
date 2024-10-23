import numpy as np

def modified_pagerank_linalg(adjacency_matrix, node_values, damping_factor=0.85, node_value_factor=0.15, max_iterations=100, tolerance=1e-6):
    """
    Calculates PageRank using linear algebra for efficiency.
    """

    num_nodes = len(adjacency_matrix)
    P = adjacency_matrix / np.sum(adjacency_matrix, axis=1, keepdims=True)  # Transition matrix
    P = P.transpose()
    V = node_values.reshape(-1, 1)  # Node values as a column vector
    PR = np.ones((num_nodes, 1)) / num_nodes  # Initialize PageRank

    for _ in range(max_iterations):
        PR_prev = PR.copy()
        PR = (1 - damping_factor) / num_nodes * np.ones((num_nodes, 1)) + damping_factor * P @ PR + node_value_factor * V
        if np.linalg.norm(PR - PR_prev) < tolerance:
            break

    return PR.flatten()  # Return as a 1D array
def pagerank(adj_matrix, sales_data, damping_factor=0.85, max_iterations=100, tolerance=1e-8):
    """
    Calculate the PageRank of each place.

    Parameters:
    adj_matrix (numpy array): Adjacency matrix representing the connection between places.
    sales_data (numpy array): Vector representing the sales data of each place.
    damping_factor (float): Probability at each step of moving to a random webpage.
    max_iterations (int): Maximum number of iterations.
    tolerance (float): Tolerance for convergence.

    Returns:
    pagerank (numpy array): PageRank of each place.
    """
    num_places = len(sales_data)
    pagerank = np.array([1.0 / num_places] * num_places)

    for _ in range(max_iterations):
        new_pagerank = (1 - damping_factor) * sales_data / np.sum(sales_data) + damping_factor * np.dot(adj_matrix, pagerank)
        if np.linalg.norm(new_pagerank - pagerank) < tolerance:
            break
        pagerank = new_pagerank

    pagerank = pagerank / np.linalg.norm(pagerank)
    return pagerank

import numpy as np

def pagerank_linalg(adj_matrix, node_values, connection_weights=0.15, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Calculates PageRank using linear algebra, incorporating node values and connection weights.

    Args:
      adj_matrix: A NumPy array representing the weighted adjacency matrix.
      node_values: A NumPy array representing the value of each node.
      connection_weights: A NumPy array representing the connection weights.
      damping_factor: The damping factor (probability of following a link).
      max_iterations: The maximum number of iterations (for iterative solution if needed).
      tolerance: The convergence tolerance (for iterative solution if needed).

    Returns:
      A NumPy array containing the PageRank scores for each node.
    """

    num_nodes = adj_matrix.shape[0]

    # Create the diagonal matrix D
    # D = np.diag(np.sum(adj_matrix @ node_values[:, np.newaxis], axis=1))
    D = np.linalg.norm(adj_matrix @ node_values[:, np.newaxis], axis=1)
    D[D == 0] = 1  # Avoid division by zero

    # Calculate the modified transition matrix P
    # P = adj_matrix * node_values / np.linalg.norm(adj_matrix @ D[:, np.newaxis], axis=1)[:,np.newaxis]
    P = adj_matrix * node_values / D[:, np.newaxis]

    # Initialize PageRank scores
    PR = np.ones(num_nodes) / num_nodes

    for i in range(max_iterations):
        old_pagerank_scores = PR.copy()

        # Update PageRank scores using the modified equation
        PR = ((1 - damping_factor) / num_nodes * np.ones(num_nodes)
                           + damping_factor * P.T @ PR
              + connection_weights * node_values)


        # Check for convergence
        if np.linalg.norm(PR - old_pagerank_scores) < tolerance:
            print(f"Converged! at iteration {i}")
            break

    return PR

def weighted_pagerank(adj_matrix, node_values, connection_weights, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Calculates PageRank scores for a network with weighted edges and node values.

    Args:
      adj_matrix: A NumPy array representing the weighted adjacency matrix of the network.
      node_values: A NumPy array of node values.
      connection_weights: A NumPy array of connection weights (needs clarification on how this is applied).
      damping_factor: The damping factor (probability of following a link).
      max_iterations: The maximum number of iterations.
      tolerance: The convergence tolerance.

    Returns:
      A NumPy array of PageRank scores.
    """

    num_nodes = adj_matrix.shape[0]

    # Calculate out-degree (with handling for dangling nodes)
    D = np.sum(adj_matrix @ node_values[:, np.newaxis], axis=1)
    D[D == 0] = 1

    # Calculate modified transition probabilities
    P = adj_matrix * node_values / np.linalg.norm(adj_matrix @ D[:, np.newaxis], axis=1)[:, np.newaxis]

    # Initialize PageRank scores
    PR = np.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        old_pagerank_scores = PR.copy()

        # PageRank update rule (needs clarification on connection_weights)
        PR = ((1 - damping_factor) / num_nodes * np.ones(num_nodes)
              + damping_factor * P.T @ PR * node_values * connection_weights)

        # Check for convergence
        if np.linalg.norm(PR - old_pagerank_scores) < tolerance:
            break

    return PR

if __name__ == "__main__":
    # Example usage:
    adj_matrix = np.array([
        [0, 1, 0, 1, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0]
    ])

    sales_data = np.array([10, 5, 10, 1, 5])

    # adj_matrix = np.array([
    #     [0, 1, 1, 0, 0],
    #     [1, 0, 0, 1, 0],
    #     [1, 0, 0, 1, 0],
    #     [0, 1, 1, 0, 1],
    #     [0, 0, 0, 1, 0]
    # ])
    #
    # sales_data = np.array([10, 5, 5, 10, 1])
    # sales_data = sales_data / np.sum(sales_data)

    # adj_matrix = np.array([
    #     [0,1,0],
    #     [1,0,0],
    #     [0,0,0]
    # ])
    #
    # sales_data = np.array([10, 1, 6])

    # pagerank_values = pagerank(adj_matrix, sales_data)
    #
    # print("PageRank values:")
    # for i, pagerank_value in enumerate(pagerank_values):
    #     print(f"Place {i+1}: {pagerank_value:.4f}")

    # pagerank_scores = modified_pagerank_linalg(adj_matrix, sales_data)
    # most_significant_node = np.argmax(pagerank_scores)
    #
    # print("PageRank scores:", pagerank_scores)
    # print("Most significant node:", most_significant_node)

    pagerank_scores = pagerank_linalg(adj_matrix, sales_data, max_iterations=1000)
    most_significant_node = np.argmax(pagerank_scores)
    print("PageRank scores:", pagerank_scores)
    print("Most significant node:", most_significant_node)

    # weighted_scores = weighted_pagerank(adj_matrix, sales_data, connection_weights=0.15)
    # print("Weighted PageRank scores:", weighted_scores)
    # print("Most significant node:", np.argmax(weighted_scores))



