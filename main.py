import time
import numpy as np
from multi_drone import MultiDrone
from my_planner import BiRRT

def run_single_test(num_drones, env_file):
    sim = MultiDrone(num_drones=num_drones, environment_file=env_file)

    print(f"[INFO] Running BiRRT for {num_drones} drones, environment={env_file}")
    start_time = time.time()

    path = BiRRT(sim, max_iterations=10000, step_size=1.5, time_limit=110)

    elapsed = time.time() - start_time
    if path is None:
        print(f"[RESULT] No path found within {elapsed:.2f}s")
        return False, elapsed, None
    else:
        path_length = sum(np.linalg.norm(path[i+1] - path[i], axis=1).max() for i in range(len(path)-1))
        print(f"[RESULT] Path found in {elapsed:.2f}s, length={path_length:.2f}")
        sim.visualize_paths(path)
        return True, elapsed, path_length


if __name__ == "__main__":
    # Example: 2 drones, environment.yaml
    #run_single_test(num_drones=2, env_file="environment.yaml")#d2o4
    #run_single_test(num_drones=2, env_file="d2o8.yaml")
    #run_single_test(num_drones=2, env_file="d2o12.yaml")
    #run_single_test(num_drones=4, env_file="d4o4.yaml")
    # run_single_test(num_drones=4, env_file="d4o8.yaml")
    #run_single_test(num_drones=4, env_file="d4o12.yaml")
    #run_single_test(num_drones=6, env_file="d6o4.yaml")
    #run_single_test(num_drones=6, env_file="d6o8.yaml")
    # run_single_test(num_drones=6, env_file="d6o12.yaml")
    # run_single_test(num_drones=8, env_file="d8o4.yaml")
    #run_single_test(num_drones=8, env_file="d8o8.yaml")
    run_single_test(num_drones=8, env_file="d8o12.yaml")



