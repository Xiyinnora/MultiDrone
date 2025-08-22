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








# import time              # 用于计时和超时控制
# import heapq             # A* 搜索需要的优先队列
# import numpy as np       # 数值计算
# from scipy.spatial import cKDTree  # 高效的k-NN搜索
# from collections import defaultdict  # 存储邻居关系
# from multi_drone import MultiDrone  # 已经实现的仿真环境类
#
# # -------------------------
# # 工具函数
# # -------------------------
# def flatten_config(q):
#     """
#     将配置 (K,3) 转成一维向量 (3K,)
#     便于做距离计算或KDTree查询
#     """
#     return q.reshape(-1)
#
# def euclidean_dist(a, b):
#     """
#     计算两份配置向量的欧几里得距离
#     输入可以是已经 flatten 的 1D 向量
#     """
#     a = np.asarray(a).reshape(-1)
#     b = np.asarray(b).reshape(-1)
#     return np.linalg.norm(a - b)  # sqrt(sum((a-b)^2))
#
# # -------------------------
# # 采样函数
# # -------------------------
# def sample_uniform_configuration(sim: MultiDrone):
#     """
#     均匀随机采样 K 个无人机的位置
#     sim._bounds 存储空间上下界
#     返回 shape (K,3) 的 numpy 数组
#     """
#     K = sim.N  # 无人机数量
#     lower = sim._bounds[:, 0]  # x,y,z 下界
#     upper = sim._bounds[:, 1]  # x,y,z 上界
#     # 对每个无人机独立采样
#     samples = np.random.uniform(lower, upper, size=(K, 3))
#     return samples.astype(np.float32)
#
# def bridge_sample(sim: MultiDrone, max_trials=50):
#     """
#     桥采样用于穿越窄缝：
#     1. 随机采样 q1, q2，要求二者碰撞
#     2. 取中点 q_mid，如果 q_mid 无碰撞，则返回
#     否则继续尝试 max_trials 次
#     """
#     K = sim.N
#     for _ in range(max_trials):
#         q1 = sample_uniform_configuration(sim)  # 随机采样 q1
#         q2 = sample_uniform_configuration(sim)  # 随机采样 q2
#         # 桥采样要求两端碰撞
#         if sim.is_valid(q1) or sim.is_valid(q2):
#             continue
#         q_mid = 0.5 * (q1 + q2)  # 中点
#         if sim.is_valid(q_mid):
#             return q_mid  # 成功找到狭缝点
#     return None  # 失败
#
# # -------------------------
# # 可视化 / 代表性检测
# # -------------------------
# def is_representative(sim: MultiDrone, q, V, visibility_threshold=10):
#     """
#     Visibility PRM 检测：拒绝那些可以看到太多已有节点的样本
#     V: 已经采样好的节点列表 (每个节点 shape=(K,3))
#     visibility_threshold: 允许可见节点数阈值
#     """
#     visible_count = 0
#     for v in V:
#         if local_planner(sim, q, v):  # 两个配置之间直线可达
#             visible_count += 1
#             if visible_count >= visibility_threshold:  # 可视节点过多 → 拒绝
#                 return False
#     return True  # 可代表，保留
#
# # -------------------------
# # 局部规划器
# # -------------------------
# def local_planner(sim: MultiDrone, q0, q1):
#     """
#     直线路径检测：从 q0 到 q1 是否可行
#     利用 MultiDrone.sim.motion_valid 检查无人机整个队列运动是否有效
#     """
#     return sim.motion_valid(np.asarray(q0, dtype=np.float32),
#                             np.asarray(q1, dtype=np.float32))
#
# # -------------------------
# # k-NN 邻居搜索
# # -------------------------
# def build_kdtree(V):
#     """
#     建立 KDTree 用于快速查找 k 最近邻
#     V: 节点列表，每个节点 shape=(K,3)
#     """
#     if len(V) == 0:
#         return None
#     data = np.vstack([flatten_config(v) for v in V])  # 每个节点展平
#     return cKDTree(data)
#
# def nearest_neighbors(kdtree, V, q, k):
#     """
#     查询配置 q 在 V 中的 k 最近邻节点
#     返回节点列表（shape=(K,3)）
#     """
#     if kdtree is None:
#         return []
#     qf = flatten_config(q)
#     dists, idxs = kdtree.query(qf, k=min(k, len(V)))  # 最近邻索引
#     if np.isscalar(idxs):
#         idxs = [idxs]
#     return [V[i] for i in idxs]
#
# # -------------------------
# # A* 图搜索
# # -------------------------
# def astar_search(V, E_adj, start, goal):
#     """
#     A* 搜索路径
#     V: 节点列表
#     E_adj: 邻接表 {idx: [neighbor_idx,...]}
#     start, goal: 配置
#     返回路径列表 [q0, q1,...,goal]
#     """
#     # 找 start/goal 对应索引
#     start_idx = None
#     goal_idx = None
#     for i, v in enumerate(V):
#         if np.allclose(v, start):
#             start_idx = i
#         if np.allclose(v, goal):
#             goal_idx = i
#     if start_idx is None or goal_idx is None:
#         return None
#
#     # A* open set: 优先队列 (fscore, idx)
#     open_set = []
#     gscore = {start_idx: 0.0}  # 从 start 到当前节点代价
#     fscore = {start_idx: euclidean_dist(flatten_config(V[start_idx]),
#                                         flatten_config(V[goal_idx]))}  # 启发式
#     heapq.heappush(open_set, (fscore[start_idx], start_idx))
#     came_from = {}  # 用于路径重建
#
#     while open_set:
#         _, current = heapq.heappop(open_set)
#         if current == goal_idx:  # 到达目标
#             # 回溯路径
#             path_idx = []
#             node = current
#             while node in came_from:
#                 path_idx.append(node)
#                 node = came_from[node]
#             path_idx.append(start_idx)
#             path_idx.reverse()
#             return [V[i] for i in path_idx]
#
#         for neigh in E_adj.get(current, []):  # 遍历邻居
#             tentative_g = gscore[current] + euclidean_dist(flatten_config(V[current]),
#                                                            flatten_config(V[neigh]))
#             if tentative_g < gscore.get(neigh, float('inf')):
#                 came_from[neigh] = current
#                 gscore[neigh] = tentative_g
#                 fscore[neigh] = tentative_g + euclidean_dist(flatten_config(V[neigh]),
#                                                              flatten_config(V[goal_idx]))
#                 heapq.heappush(open_set, (fscore[neigh], neigh))
#
#     return None  # 无法到达
#
# # -------------------------
# # Improved PRM 主函数
# # -------------------------
# def improved_prm(sim: MultiDrone,
#                  num_samples=900,
#                  k_neighbors=15,
#                  drone_radius=0.3,
#                  sample_bridge_prob=0.3,
#                  visibility_threshold=6,
#                  bridge_trials=40,
#                  timeout_seconds=60):
#     """
#     改进 PRM 主函数，带桥采样 + Visibility PRM + kNN + A*
#     返回路径列表或 None
#     """
#     start = sim.initial_configuration.astype(np.float32)  # 起点配置
#     goal = sim.goal_positions.astype(np.float32)          # 目标配置
#     K = sim.N
#     drone_radius = drone_radius if drone_radius is not None else sim._drone_radius
#
#     V = []  # 节点列表
#     start_idx = None
#     goal_idx = None
#
#     t0 = time.time()  # 计时
#
#     # 检查起点和终点是否有效
#     if not sim.is_valid(start):
#         raise RuntimeError("Start configuration is invalid!")
#     if not sim.is_valid(goal):
#         raise RuntimeError("Goal configuration is invalid!")
#
#     # 将 start/goal 添加进节点集
#     V.append(start)
#     V.append(goal)
#     start_idx = 0
#     goal_idx = 1
#
#     # Step 1: 采样节点
#     while len(V) < num_samples and (time.time() - t0) < timeout_seconds:
#         if np.random.rand() < sample_bridge_prob:
#             q = bridge_sample(sim, max_trials=bridge_trials)  # 桥采样
#             if q is None:
#                 continue
#         else:
#             q = sample_uniform_configuration(sim)  # 普通均匀采样
#
#         if not sim.is_valid(q):
#             continue  # 冲突 -> 丢弃
#
#         if not is_representative(sim, q, V, visibility_threshold=visibility_threshold):
#             continue  # 可视化冗余 -> 丢弃
#
#         V.append(q)  # 添加节点
#
#     # Step 2: 建图 (kNN + 局部规划)
#     kdtree = build_kdtree(V)
#     n = len(V)
#     E_adj = defaultdict(list)  # 邻接表
#     for i, q in enumerate(V):
#         neighbors = nearest_neighbors(kdtree, V, q, k_neighbors + 1)  # +1 防止包含自己
#         for nbr in neighbors:
#             found_idx = None
#             for j, vv in enumerate(V):
#                 if np.allclose(vv, nbr):
#                     found_idx = j
#                     break
#             if found_idx is None or found_idx == i:
#                 continue
#             if local_planner(sim, q, V[found_idx]):
#                 E_adj[i].append(found_idx)
#                 E_adj[found_idx].append(i)  # 无向图
#
#     # Step 3: A* 搜索
#     path = astar_search(V, E_adj, start, goal)
#     return path
#
#
