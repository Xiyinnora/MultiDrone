import time
import numpy as np
import pandas as pd
from scipy import stats
from multi_drone import MultiDrone
from my_planner import BiRRT


def run_single_test(num_drones, env_file):
    """Run BiRRT once and return success, elapsed time, and path length (if found)."""
    sim = MultiDrone(num_drones=num_drones, environment_file=env_file)
    start_time = time.time()
    path = BiRRT(sim, max_iterations=10000, step_size=1.5, time_limit=110)
    elapsed = time.time() - start_time

    if path is None:
        return False, elapsed, None
    else:
        # 计算路径长度（逐段累加）
        path_length = sum(
            np.linalg.norm(path[i+1] - path[i], axis=1).max()
            for i in range(len(path)-1)
        )
        return True, elapsed, path_length


def run_statistics(num_runs, num_drones, env_file):
    """Run multiple experiments and compute summary statistics."""
    times = []
    lengths = []
    successes = 0

    for i in range(num_runs):
        print(f"[INFO] Run {i+1}/{num_runs} | {num_drones} drones, env={env_file}")
        success, elapsed, path_length = run_single_test(num_drones, env_file)
        times.append(elapsed)
        if success:
            successes += 1
            lengths.append(path_length)

    times = np.array(times)
    lengths = np.array(lengths)

    # 成功率
    success_rate = successes / num_runs * 100

    # 时间统计（所有实验都算）
    time_mean = np.mean(times)
    time_median = np.median(times)
    time_ci = stats.t.interval(0.95, len(times)-1, loc=time_mean, scale=stats.sem(times))

    # 路径长度统计（只算成功的实验）
    if len(lengths) > 0:
        length_mean = np.mean(lengths)
        length_median = np.median(lengths)
        length_ci = stats.t.interval(0.95, len(lengths)-1, loc=length_mean, scale=stats.sem(lengths))
    else:
        length_mean = length_median = None
        length_ci = (None, None)

    return {
        "Num Drones": num_drones,
        "Environment": env_file,
        "Success Rate (%)": round(success_rate, 1),
        "Time Mean (s)": round(time_mean, 2),
        "Time Median (s)": round(time_median, 2),
        "Time 95% CI": (round(time_ci[0], 2), round(time_ci[1], 2)),
        "Length Mean": round(length_mean, 2) if length_mean is not None else "N/A",
        "Length Median": round(length_median, 2) if length_median is not None else "N/A",
        "Length 95% CI": (
            (round(length_ci[0], 2), round(length_ci[1], 2))
            if length_mean is not None else "N/A"
        )
    }


if __name__ == "__main__":
    num_runs = 30  # 每种配置跑 30 次

    # 实验矩阵：无人机数量 × 环境文件
    experiments = [
        (2, "environment.yaml"),
        (2, "d2o8.yaml"),
        (2, "d2o12.yaml"),
        (4, "d4o4.yaml"),
        (4, "d4o8.yaml"),
        (4, "d4o12.yaml"),
        (6, "d6o4.yaml"),
        (6, "d6o8.yaml"),
        (6, "d6o12.yaml"),
        (8, "d8o4.yaml"),
        (8, "d8o8.yaml"),
        (8, "d8o12.yaml"),
    ]

    results = []
    for num_drones, env_file in experiments:
        print(f"\n=== Running {num_drones} drones in {env_file} ===")
        stats_dict = run_statistics(num_runs, num_drones, env_file)
        results.append(stats_dict)

    # 转成 DataFrame 表格
    df = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    print("\n===== Summary Table =====")
    print(df)

    # 保存到 CSV 文件
    df.to_csv("biRRT_experiment_summary.csv", index=False)
    print("\n[INFO] Results saved to biRRT_experiment_summary.csv")
