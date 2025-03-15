import os
import numpy as np
import math
import carla


def load_and_merge_trajectories(folder_path):
    """
    读取文件夹下所有 .npy 文件，提取每个文件的 ego_trajectory [(frame, x, y, z), ...]。
    仅在检测到该文件中有安全违规时，将其整段轨迹加入合并列表，用于后续Coverage计算。

    返回 (all_xy, num_files)，只保留 x, y。
    """
    all_xy = []

    # 找到所有 .npy 文件
    npy_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ]

    for file in npy_files:
        data = np.load(file, allow_pickle=True).item()
        raw_trajectory = data["ego_trajectory"]  # [(frame, x, y, z), ...]
        violation = data["violations"]

        # 如果出现我们关注的违规类型 (示例: side_collision、multi_collision)
        if any(violation[key] == 1 for key in ["side_collision", "multi_collision"]):
            for point in raw_trajectory:
                _, x, y, _ = point
                all_xy.append([x, y])

    all_xy = np.array(all_xy)  # shape: (N, 2)
    num_files = len(npy_files)
    print(f"[INFO] from {num_files} files get {len(all_xy)} violated waypoints")
    return all_xy, num_files


def build_scenario_vectors(folder_path):
    """
    针对每个出现违规的文件，构造一个“场景表征向量”，包含:
      - Ego 车辆初始位置 x_ego, y_ego
      - 周围车辆的初始位置 (x, y) 若周围车数少于最大值则后面补零
    返回:
      raw_vectors: list[ (ego_x, ego_y, [(sx1, sy1), (sx2, sy2), ...]) ]
      valid_files: 有违规的文件列表
    """
    raw_vectors = []
    valid_files = []

    npy_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ]

    for file in npy_files:
        data = np.load(file, allow_pickle=True).item()
        violation = data["violations"]

        # 如果出现需要的违规
        if any(violation[key] == 1 for key in ["side_collision", "multi_collision", "object_collision", "Time-out"]):
            ego_trajectory = data["ego_trajectory"]
            if len(ego_trajectory) == 0:
                continue

            # 取 Ego 初始位置 (示例: 第0帧)
            _, ego_x, ego_y, _ = ego_trajectory[0]

            # 周围车辆 (这里假设 data["surrounding_vehicles"] 里能拿到初始位置)
            surrounding_vehicles = data.get("surrounding_vehicles", [])
            sur_positions = []
            for veh in surrounding_vehicles:
                # 若存了 trajectory，可取 veh["trajectory"][0]
                # 这里仅示例写成 veh.get("x"), veh.get("y")
                sx = veh.get("x", 0.0)
                sy = veh.get("y", 0.0)
                sur_positions.append((sx, sy))

            raw_vectors.append((ego_x, ego_y, sur_positions))
            valid_files.append(file)

    return raw_vectors, valid_files


def unify_scenario_vectors(raw_vectors):
    """
    把每个场景 (ego_x, ego_y, [ (sx1, sy1), (sx2, sy2), ... ]) 补零对齐到相同长度，返回 numpy array。

    最终每行形状: [ ego_x, ego_y, n_sur, sx1, sy1, sx2, sy2, ..., sxM, syM ]
      其中 M = 所有场景中周围车数量的最大值。若某场景实际周围车数 < M 则补零。
    """
    # 先找最大周围车数量
    max_sur_num = 0
    for ego_x, ego_y, sur_positions in raw_vectors:
        max_sur_num = max(max_sur_num, len(sur_positions))

    scenario_list = []
    for ego_x, ego_y, sur_positions in raw_vectors:
        n_sur = len(sur_positions)
        row = [ego_x, ego_y, n_sur]
        for sx, sy in sur_positions:
            row.append(sx)
            row.append(sy)
        # 如果数量不够，就补零
        diff = max_sur_num - n_sur
        if diff > 0:
            row.extend([0.0] * (2 * diff))  # 每辆车 (sx, sy) 两个维度
        scenario_list.append(row)

    scenario_matrix = np.array(scenario_list, dtype=float)
    return scenario_matrix


def compute_average_distance(matrix):
    """
    给定形状 (N, D) 的场景向量矩阵，计算它们的平均两两欧几里得距离。
    公式: (1 / n^2) * ΣᵢΣⱼ d( vec_i, vec_j ).
    实现方式: dist_matrix = 每对 i,j 的距离，然后 dist_matrix.mean().
    返回 (avg_dist, dist_matrix)
    """
    n = len(matrix)
    if n == 0:
        return 0.0, None
    if n == 1:
        return 0.0, np.zeros((1, 1))

    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(matrix[i] - matrix[j])

    avg_dist = dist_matrix.mean()
    return avg_dist, dist_matrix


def get_road_grid(client, sampling_distance=1.0):
    """
    从 CARLA 中获取道路(可行驶)的离散坐标点（间隔为 sampling_distance）。
    转换为 (int_x, int_y) 存入一个 set 以便做 Coverage 统计。
    """
    world = client.load_world('Town01')
    carla_map = world.get_map()

    waypoints = carla_map.generate_waypoints(sampling_distance)
    road_grid = set()
    for wp in waypoints:
        x = wp.transform.location.x
        y = wp.transform.location.y
        grid_x = int(round(x))
        grid_y = int(round(y))
        road_grid.add((grid_x, grid_y))

    print(f"[INFO] generate {len(road_grid)} grids)")
    return road_grid


def calculate_coverage(road_grid, all_traj_xy):
    """
    基于 1m 网格来计算覆盖率:
    - road_grid: {(int_x, int_y), ...} 所有可行驶格子
    - all_traj_xy: (N, 2) 违规轨迹点
    返回 (coverage_ratio, covered_count, total_count).
    """
    visited_grid = set()
    total_count = len(road_grid)
    if total_count == 0:
        return 0.0, 0, 0

    if len(all_traj_xy) == 0:
        return 0.0, 0, total_count

    for x, y in all_traj_xy:
        gx, gy = int(round(x)), int(round(y))
        if (gx, gy) in road_grid:
            visited_grid.add((gx, gy))

    covered_count = len(visited_grid)
    coverage_ratio = covered_count / total_count
    return coverage_ratio, covered_count, total_count


def get_max_road_distance(client, sampling_distance=1.0):
    """
    生成 Town04 地图中可行驶道路上的所有 Waypoints（间隔 sampling_distance），
    找出它们之间的最大欧几里得距离并返回。

    注意: O(n^2) 实现，采样太密会非常耗时。
    """
    world = client.load_world('Town01')
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(sampling_distance)

    coords = [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints]
    n = len(coords)
    if n < 2:
        return 0.0

    max_dist = 0.0
    # 直接两层循环
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist > max_dist:
                max_dist = dist
    return max_dist


def main(folder_path, carla_host='127.0.0.1', carla_port=2000, sampling_distance=1.0):
    """
    1. 连接 CARLA
    2. 计算覆盖率
    3. 计算场景间距离 (Ego + SurroundingVehicles 初始位置)
    4. 计算地图可行驶的最大距离
    """
    # 1. 连接 CARLA
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(10.0)

    # 2. 覆盖率计算
    all_traj_xy, num_files = load_and_merge_trajectories(folder_path)
    road_grid = get_road_grid(client, sampling_distance=sampling_distance)
    coverage_ratio, covered_count, total_count = calculate_coverage(road_grid, all_traj_xy)

    print(f"\n[RESULT] Total grid: {total_count}")
    print(f"[RESULT] covered grid: {covered_count}")
    print(f"[RESULT] Trajectory Coverage: {coverage_ratio:.2%}")

    # 3. 场景间的距离 (保留之前的距离计算功能, 不删除)
    raw_vectors, valid_files = build_scenario_vectors(folder_path)
    scenario_matrix = unify_scenario_vectors(raw_vectors)
    avg_dist, dist_matrix = compute_average_distance(scenario_matrix)

    print(f"\n[RESULT] Safety violation case number: {len(valid_files)}")
    print(f"[RESULT] initial position distance: {avg_dist:.4f}")
    # 如果想查看距离矩阵，可以取消注释:
    # print("距离矩阵:\n", dist_matrix)

    # 4. 地图上最大可行驶距离 (Town04)
    max_road_dist = get_max_road_distance(client, sampling_distance=sampling_distance)
    print(f"\n[RESULT] map (sample={sampling_distance}m) maximum distance: {max_road_dist:.2f} m")


if __name__ == "__main__":
    folder_path = "trajectory/MART_town01_1"  # 你的轨迹文件夹
    sampling_distance = 10  # 生成道路waypoints的采样间隔(单位:米)

    main(
        folder_path=folder_path,
        carla_host='127.0.0.1',
        carla_port=2000,
        sampling_distance=sampling_distance
    )
