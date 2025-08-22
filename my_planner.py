import numpy as np
import time
from collections import deque

class Node:
    def __init__(self, configuration, parent=None):
        self.configuration = configuration
        self.parent = parent


def steer(q_near, q_rand, step_size):
    """Move from q_near toward q_rand with step size limitation."""
    direction = q_rand - q_near
    dist = np.linalg.norm(direction, axis=1).max()  # use max drone distance
    if dist <= step_size:
        return q_rand
    step = direction / dist * step_size
    return q_near + step


def nearest(tree, q_rand):
    """Find nearest node in tree to q_rand."""
    min_dist = float("inf")
    nearest_node = None
    for node in tree:
        dist = np.linalg.norm(node.configuration - q_rand, axis=1).max()
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node


def extract_path(node):
    """Extract path from root to given node."""
    path = []
    while node is not None:
        path.append(node.configuration)
        node = node.parent
    path.reverse()
    return path


def concatenate_paths(path_a, path_b):
    """Concatenate two paths (path_b is already reversed)."""
    return path_a + path_b


def BiRRT(sim, max_iterations=5000, step_size=1.0, time_limit=110):
    """
    Bidirectional RRT for MultiDrone.
    Args:
        sim: MultiDrone environment
        max_iterations: maximum iterations
        step_size: step size for expansion
        time_limit: maximum runtime (s), safety < 120
    Returns:
        path (list of configurations) or None
    """
    start_time = time.time()

    # Initialize two trees
    tree_a = [Node(sim.initial_configuration)]
    tree_b = [Node(sim.goal_positions)]

    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            print("[BiRRT] Timeout")
            return None

        # Sample a random configuration in bounds
        q_rand = np.random.uniform(sim._bounds[:, 0], sim._bounds[:, 1], (sim.N, 3)).astype(np.float32)

        # Step A: expand tree_a
        q_near_a = nearest(tree_a, q_rand)
        q_new_a_config = steer(q_near_a.configuration, q_rand, step_size)

        if sim.motion_valid(q_near_a.configuration, q_new_a_config):
            q_new_a = Node(q_new_a_config, parent=q_near_a)
            tree_a.append(q_new_a)

            # Step B: expand tree_b toward q_new_a
            q_near_b = nearest(tree_b, q_new_a.configuration)
            q_new_b_config = steer(q_near_b.configuration, q_new_a.configuration, step_size)

            if sim.motion_valid(q_near_b.configuration, q_new_b_config):
                q_new_b = Node(q_new_b_config, parent=q_near_b)
                tree_b.append(q_new_b)

                # If trees are connected
                if np.linalg.norm(q_new_a.configuration - q_new_b.configuration, axis=1).max() < step_size:
                    path_a = extract_path(q_new_a)
                    path_b = extract_path(q_new_b)
                    path_b.reverse()
                    return concatenate_paths(path_a, path_b)

        # Swap trees
        tree_a, tree_b = tree_b, tree_a

    print("[BiRRT] Failed to find a path")
    return None



