import random
import math
import numpy as np
from utility import average_population_distance
action_space = np.array([
    [0, 3],
    [1, 3],
    [2, 3],
    [0, 0],
    [1, 0],
    [2, 0]
], dtype=np.int32)

def distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

# ========== 1) 分别写两个采样函数: position 和 action ==========

def sample_position_info(carla_map):
    """
    用于采样关于位置的信息: 比如 vehicle_num, EGO位置, surrounding位置...
    这里只是一个简单示例.
    """
    v_num = random.choice([3])  # 这里可改
    while True:
        spawn_points = carla_map.get_spawn_points()
        # 随机选EGO点
        ego_idx = random.randint(0, len(spawn_points) - 1)
        ego_spawning = spawn_points.pop(ego_idx)

        # 寻找EGO附近的spawn点
        nearby_spawn_points = [
            sp for sp in spawn_points
            if distance(ego_spawning.location, sp.location) <= 20000
        ]
        if len(nearby_spawn_points) >= v_num:
            break
    selected_spawns = random.sample(nearby_spawn_points, v_num)
    return {
        "vehicle_num": v_num,
        "ego_transform": ego_spawning,
        "surrounding_transforms": selected_spawns
    }
def sample_action_info(vehicle_num, seq_len=50):
    """
    用于采样关于action的信息: 比如每辆surrounding车都有一个动作序列
    """
    actions_per_vehicle = []
    for _ in range(vehicle_num):
        seq = []
        for _ in range(seq_len):
            seq.append(action_space[random.randint(0, len(action_space)-1)])
        actions_per_vehicle.append(seq)
    return actions_per_vehicle

# ========== 2) 定义一个 GA 类，把 position/action 都放在同一个individual里 ==========

class CombinedGA:
    """
    每个个体由两部分组成:
      individual = {
        "position_info": {...},
        "action_info": [...]
      }
    其中 position_info 和 action_info 的 sample 是分开的
    但在 crossover / mutation 的时候，通过一个 enable_action_ga 开关来决定是否也处理 action 部分
    """

    def __init__(self, carla_map, population_size=10, generations=5, seq_len=5, enable_action_ga=False):
        """
        :param carla_map: world.get_map()
        :param population_size: GA种群大小
        :param generations: 迭代次数
        :param seq_len: 每辆车的动作序列长度(示例)
        :param enable_action_ga: 是否对 action_info 也做 crossover/mutation 的开关
        """
        self.carla_map = carla_map
        self.safe_set = []
        self.population_size = population_size
        self.generations = generations
        self.seq_len = seq_len
        self.enable_action_ga = enable_action_ga

        self.population = []  # 列表, 每个元素是 { "position_info":..., "action_info": ...}
        self.best_individual = None
        self.best_fitness = float('-inf')

    def sample_initial_population(self):
        """
        初始化种群: 分别sample position + action，再组合
        """
        for _ in range(self.population_size):
            pos_info = sample_position_info(self.carla_map)
            if self.enable_action_ga:
                act_info = sample_action_info(pos_info["vehicle_num"], self.seq_len)
                individual = {
                    "position_info": pos_info,
                    "action_info": act_info
                }
            else:
                individual = {
                    "position_info": pos_info,
                }
            self.population.append(individual)



    # ---------- 关键：crossover / mutation 里根据开关决定是否处理 action ----------

    def crossover_individuals(self, parent1, parent2, crossover_rate=0.8):
        """
        对两个个体进行交叉。个体数据结构举例：

        parent = {
            'position_info': {
                'vehicle_num': 2,
                'ego_transform': <carla.libcarla.Transform object>,
                'surrounding_transforms': [<Transform>, <Transform>]
            }
        }

        要点：
        1. vehicle_num 和 surrounding_transforms 绑定在一起，跟着走。
        2. ego_transform 独立选择。

        如果随机数大于交叉率，则不发生交叉，直接返回父代。
        """
        # 如果不满足交叉概率，直接返回父代（无变化）
        if random.random() >= crossover_rate:
            return parent1, parent2

        # -----------------------------
        # 若进行交叉，则构造子代
        # -----------------------------
        child1 = {'position_info': {}}
        child2 = {'position_info': {}}

        # ============== 1. 决定子代的 vehicle_num & surrounding_transforms ==============
        # 先随机决定 child1 的“vehicle_num + surrounding_transforms”来自哪一个父代

        child1['position_info']['vehicle_num'] = parent2['position_info']['vehicle_num']
        child1['position_info']['surrounding_transforms'] = parent2['position_info']['surrounding_transforms']
        child1['position_info']['ego_transform'] = parent1['position_info']['ego_transform']

        child2['position_info']['vehicle_num'] = parent1['position_info']['vehicle_num']
        child2['position_info']['surrounding_transforms'] = parent1['position_info']['surrounding_transforms']
        child2['position_info']['ego_transform'] = parent2['position_info']['ego_transform']

        return child1, child2

    def mutation(self, population, mutation_rate=0.2):
        if random.random() < mutation_rate:
            candidate = []
            dist_set =[]
            for _ in range(10):
                pos_info = sample_position_info(self.carla_map)
                individual = {
                    "position_info": pos_info,
                }
                candidate.append(individual)
            for item in candidate:
                dist = average_population_distance(item, self.safe_set)
                dist_set.append(dist)
            max_idx = max(range(len(dist_set)), key=lambda i: dist_set[i])
            return candidate[max_idx]
        else:
            return population
    def resample(self):
        pos_info = sample_position_info(self.carla_map)
        individual = {
            "position_info": pos_info,
        }
        return individual


