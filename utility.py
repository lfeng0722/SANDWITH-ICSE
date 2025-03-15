import numpy as np
import xml.etree.ElementTree as ET
import math
import carla
from typing import List, Set


def map_size(world):
    # 加载.xodr文件
    xodr_path = "Town01.xodr"
    tree = ET.parse(xodr_path)
    root = tree.getroot()

    # 初始化范围
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # 遍历几何信息
    for road in root.findall('road'):
        for geometry in road.find('planView').findall('geometry'):
            x = float(geometry.get('x'))
            y = float(geometry.get('y'))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    return min_x, max_x , min_y, max_y

def get_xy_speed(vehicle):
    """
    Get the X and Y speed components of a vehicle in m/s.
    """
    velocity = vehicle.get_velocity()
    x_speed = velocity.x  # Forward/Backward speed (m/s)
    y_speed = velocity.y  # Lateral speed (m/s)
    return x_speed, y_speed

def position_scaler(position, x_min, x_max, y_min, y_max):
    # Example values for min and max positions

    # Original positions
    npc_position_x, npc_position_y = position

    # Scaling to [0, 2]
    x_range = x_max - x_min
    y_range = y_max - y_min
    scaled_x = ((npc_position_x - x_min) / x_range) * 2
    # print(min_x)
    scaled_y = ((npc_position_y - y_min) / y_range) * 2

    return scaled_x, scaled_y

def state_encoder(ego_vehicle, vehicles, ego_vehicle_position, agent_positions, max_x_diff, max_y_diff):
    state = []
    # 计算所有车辆的速度，避免重复调用
    vehicle_speeds = [get_xy_speed(vehicle) for vehicle in vehicles]
    ego_vel_x, ego_vel_y = get_xy_speed(ego_vehicle)

    for i, agent_position in enumerate(agent_positions):
        vel_x, vel_y = vehicle_speeds[i]

        # 计算其他代理的位置和速度
        other_agents = [pos for j, pos in enumerate(agent_positions) if j != i]
        other_vels = [vehicle_speeds[j] for j in range(len(vehicles)) if j != i]

        # 确保至少有两个代理（否则填充默认值）
        while len(other_agents) < 2:
            other_agents.append((0, 0))
            other_vels.append((0, 0))

        agent_state = (
            agent_position[0] / max_x_diff,
            agent_position[1] / max_y_diff,
            vel_x,
            vel_y,
            other_agents[0][0] / max_x_diff,
            other_agents[0][1] / max_y_diff,
            other_vels[0][0],
            other_vels[0][1],
            other_agents[1][0] / max_x_diff,
            other_agents[1][1] / max_y_diff,
            other_vels[1][0],
            other_vels[1][1],
            ego_vehicle_position[0] / max_x_diff,
            ego_vehicle_position[1] / max_y_diff,
            ego_vel_x,
            ego_vel_y
        )
        state.append(agent_state)

    return state


def has_passed_destination(vehicle, destination_location, threshold_distance=10):
    """
    Returns True if the vehicle is either:
      (a) within threshold_distance of the destination *and* its speed is 0 (or near 0), OR
      (b) the destination is behind the vehicle (dot < 0).
    """
    import math

    # 1. 获取车辆当前位置以及速度
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_velocity = vehicle.get_velocity()
    speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2)

    # 2. 计算车辆到目标点的距离向量
    direction_to_destination = carla.Vector3D(
        destination_location.x - vehicle_location.x,
        destination_location.y - vehicle_location.y
    )
    distance_to_destination = math.sqrt(
        direction_to_destination.x ** 2 + direction_to_destination.y ** 2
    )

    # 3. 计算车辆的“前向”朝向（基于yaw）
    yaw = math.radians(vehicle_transform.rotation.yaw)
    forward_vector = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

    # 4. 计算与目标点方向向量的点乘 dot
    dot = (direction_to_destination.x * forward_vector.x +
           direction_to_destination.y * forward_vector.y)

    # 条件 (a): 近距离且速度几乎为0
    near_and_stopped = (distance_to_destination < threshold_distance and speed == 0 )

    # 条件 (b): 目的地在车辆后方
    behind_vehicle = (dot < 0 and distance_to_destination > threshold_distance)

    return near_and_stopped, behind_vehicle


import numpy as np
from scipy.spatial.distance import euclidean
import random

def calculate_population_distance(pop1, pop2):
    """
    计算两个 population 之间的距离，包括：
    - EGO 车辆位置
    - surrounding 车辆位置
    - 车辆数量
    alpha 控制车辆数量的权重
    """
    # 计算 EGO 车辆位置的欧几里得距离
    ego_dist = euclidean(
        (pop1["ego_transform"].location.x, pop1["ego_transform"].location.y),
        (pop2["ego_transform"].location.x, pop2["ego_transform"].location.y)
    )

    # 计算 surrounding 车辆位置的平均距离（考虑最小公共数量）
    min_vehicle_num = min(len(pop1["surrounding_transforms"]), len(pop2["surrounding_transforms"]))
    surrounding_dists = [
        euclidean(
            (p1.location.x, p1.location.y),
            (p2.location.x, p2.location.y)
        )
        for p1, p2 in zip(pop1["surrounding_transforms"][:min_vehicle_num],
                          pop2["surrounding_transforms"][:min_vehicle_num])
    ]

    # 计算车辆数量的影响
    vehicle_num_diff = abs(pop1["vehicle_num"] - pop2["vehicle_num"]) / max(pop1["vehicle_num"], pop2["vehicle_num"])

    # 计算最终距离
    avg_distance = (ego_dist + sum(surrounding_dists)) / (min_vehicle_num + 1)  # 位置部分
    total_distance = avg_distance + vehicle_num_diff  # 加上车辆数量的影响
    return total_distance


def average_population_distance(population, generation):
    """
    计算一个 population 与整个 generation 之间的平均距离
    """
    distances = [calculate_population_distance(population["position_info"], pop["position_info"]) for pop in generation]
    return np.mean(distances)


def parents_selection(fitness, population, population_size):
    processed_fitness = []
    for i in range(population_size):
        processed_fitness.append(
            [fitness["safety_violation"][i], fitness["diversity"][i], fitness["ART_trigger_time"][i]])
    rank, weight = non_dominated_sorting_initial(processed_fitness)
    parents = random.choices(population, weights=weight, k=population_size)
    return parents

def next_gen_selection(fitness, population, population_size):
    processed_fitness = []
    for i in range(population_size):
        processed_fitness.append(
            [fitness["safety_violation"][i+10], fitness["diversity"][i+10], fitness["ART_trigger_time"][i+10]])
    sorted_population = non_dominated_sorting_with_weights(processed_fitness, population)
    population = sorted_population[0:population_size]
    return population

def non_dominated_sorting_initial(solutions: List[List[float]]) -> (List[Set[int]], List[float]):
    def dominates(sol1: List[float], sol2: List[float]) -> bool:
        """Check if sol1 dominates sol2."""
        return all(x <= y for x, y in zip(sol1, sol2)) and any(x < y for x, y in zip(sol1, sol2))

    # Non-dominated sorting
    n = len(solutions)
    dominated_by = [set() for _ in range(n)]
    dominates_count = [0 for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(solutions[i], solutions[j]):
                    dominated_by[i].add(j)
                elif dominates(solutions[j], solutions[i]):
                    dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for sol in fronts[i]:
            for dominated_sol in dominated_by[sol]:
                dominates_count[dominated_sol] -= 1
                if dominates_count[dominated_sol] == 0:
                    next_front.append(dominated_sol)
        i += 1
        fronts.append(next_front)

    fronts = [set(front) for front in fronts if front]

    # Assigning weights
    weights = [0] * n
    for i, front in enumerate(fronts):
        weight = 1 / (i + 1)
        for sol in front:
            weights[sol] = weight

    return fronts, weights

def non_dominated_sorting_with_weights(solutions: List[List[float]],population) -> List[List[float]]:
    def dominates(sol1: List[float], sol2: List[float]) -> bool:
        """Check if sol1 dominates sol2."""
        return all(x <= y for x, y in zip(sol1, sol2)) and any(x < y for x, y in zip(sol1, sol2))

    def calculate_crowding_distance(front_solutions: List[List[float]]) -> List[float]:
        """Calculate the crowding distance of each solution in the front."""
        if not front_solutions:
            return []

        size = len(front_solutions)
        distances = [0.0 for _ in range(size)]
        for m in range(len(front_solutions[0])):
            sorted_indices = sorted(range(size), key=lambda x: front_solutions[x][m])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            for i in range(1, size - 1):
                distances[sorted_indices[i]] += (
                            front_solutions[sorted_indices[i + 1]][m] - front_solutions[sorted_indices[i - 1]][m])

        return distances

    # Non-dominated sorting
    n = len(solutions)
    dominated_by = [set() for _ in range(n)]
    dominates_count = [0 for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(solutions[i], solutions[j]):
                    dominated_by[i].add(j)
                elif dominates(solutions[j], solutions[i]):
                    dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for sol in fronts[i]:
            for dominated_sol in dominated_by[sol]:
                dominates_count[dominated_sol] -= 1
                if dominates_count[dominated_sol] == 0:
                    next_front.append(dominated_sol)
        i += 1
        fronts.append(next_front)

    fronts = [set(front) for front in fronts if front]

    # Sorting within each front based on crowding distance
    sorted_population_indices = []
    for front in fronts:
        front_list = list(front)
        front_solutions = [solutions[i] for i in front_list]
        crowding_distances = calculate_crowding_distance(front_solutions)
        # Sort the front based on crowding distance (descending order)
        sorted_front_indices = sorted(front_list, key=lambda i: crowding_distances[front_list.index(i)], reverse=True)
        sorted_population_indices.extend(sorted_front_indices)

    sorted_population = [population[i] for i in sorted_population_indices]

    return sorted_population