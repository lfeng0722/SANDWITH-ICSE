import math
import random
from itertools import product


def action_trans(vehicle, action):
    """
    根据动作名称对车辆执行相应操作的示例函数。
    """
    if action == 'break':
        vehicle.decelerate()
    elif action == 'accelerate':
        vehicle.accelerate()
    elif action == 'right_change_acc':
        vehicle.request_lane_change_accel('right')
    elif action == 'right_change_dec':
        vehicle.request_lane_change_decel('right')
    elif action == 'left_change_acc':
        vehicle.request_lane_change_accel('left')
    elif action == 'left_change_dec':
        vehicle.request_lane_change_decel('left')


class ARTSelector4D:
    def __init__(self, scale_pos=1.0, scale_action=1.0):
        """
        - scale_pos: 用于缩放 (p_x, p_y)
        - scale_action: 用于缩放 (a_x, a_y)

        这里的 risk_set_1, safe_set_1 (4维)；risk_set_2, safe_set_2(8维)；...
        存储的 (p_x, p_y, a_x, a_y, ...) 都是「相对坐标 + 动作」拼成的4N维向量。
        """
        self.scale_pos = scale_pos
        self.scale_action = scale_action

        # 6 种离散动作: 名称 -> (dx, dy)
        self.actions = {
            'break': (0, -1),
            'accelerate': (0, 1),
            'right_change_acc': (1, 1),
            'right_change_dec': (1, -1),
            'left_change_acc': (-1, 1),
            'left_change_dec': (-1, -1)
        }
        # 反向映射: (dx, dy) -> 动作名称
        self.code_to_action = {v: k for k, v in self.actions.items()}

        # 风险/安全集合
        self.risk_set_1 = []
        self.safe_set_1 = []
        self.risk_set_2 = []
        self.safe_set_2 = []
        self.risk_set_3 = []
        self.safe_set_3 = []
        self.risk_set_4 = []
        self.safe_set_4 = []

    def distance_4N(self, vec1, vec2, num_vehicle):
        """
        计算长度=4 * num_vehicle 的欧几里得距离, 并考虑缩放.
        vec1, vec2: (p_x1, p_y1, a_x1, a_y1, p_x2, p_y2, a_x2, a_y2, ...)
        """
        total_sq = 0.0
        for i in range(num_vehicle):
            base = 4 * i
            # 位置
            p1x = vec1[base + 0]
            p1y = vec1[base + 1]
            p2x = vec2[base + 0]
            p2y = vec2[base + 1]
            # 动作
            a1x = vec1[base + 2]
            a1y = vec1[base + 3]
            a2x = vec2[base + 2]
            a2y = vec2[base + 3]

            # 缩放
            p1x_s = p1x / self.scale_pos
            p1y_s = p1y / self.scale_pos
            p2x_s = p2x / self.scale_pos
            p2y_s = p2y / self.scale_pos

            a1x_s = a1x / self.scale_action
            a1y_s = a1y / self.scale_action
            a2x_s = a2x / self.scale_action
            a2y_s = a2y / self.scale_action

            total_sq += (p1x_s - p2x_s) ** 2 + (p1y_s - p2y_s) ** 2 \
                        + (a1x_s - a2x_s) ** 2 + (a1y_s - a2y_s) ** 2

        return math.sqrt(total_sq)

    def build_global_vector(self, ego_position, vehicle_list, actions_list):
        ego_x, ego_y = ego_position
        vec = []
        for v, (dx, dy) in zip(vehicle_list, actions_list):
            loc = v.get_location()  # v.get_location() 返回一个 Location 对象
            vx = loc.x
            vy = loc.y
            px = vx - ego_x
            py = vy - ego_y
            vec.extend([px, py, dx, dy])
        return tuple(vec)

    def parse_global_vector_to_actions(self, global_vec):
        """
        将 4N 维向量的动作部分 (ax, ay) 提取出来, 映射成动作名称列表
        """
        length = len(global_vec)
        num_vehicle = length // 4
        actions_list = []
        for i in range(num_vehicle):
            base = 4 * i
            ax = global_vec[base + 2]
            ay = global_vec[base + 3]
            act_name = self.code_to_action.get((ax, ay), "Unknown")
            actions_list.append(act_name)
        return actions_list

    def choose_actions_for_all_vehicles(self, ego_position, vehicle_list):
        """
        主函数:
        1) 判断有多少车辆 => 选 risk_set_x / safe_set_x
        2) 如果 risk_set 和 safe_set 都为空, 就随机构造 4N向量 也返回
        3) 如果不为空, 就穷举所有组合进行距离比较
        4) 返回 (最优动作列表, 最优4N向量)
        """
        num_vehicle = len(vehicle_list)
        # 挑选对应的 risk_set / safe_set
        if num_vehicle == 1:
            risk_set = self.risk_set_1
            safe_set = self.safe_set_1
        elif num_vehicle == 2:
            risk_set = self.risk_set_2
            safe_set = self.safe_set_2
        elif num_vehicle == 3:
            risk_set = self.risk_set_3
            safe_set = self.safe_set_3
        else:
            risk_set = self.risk_set_4
            safe_set = self.safe_set_4
        # ---【改动处】---
        if not risk_set and not safe_set:
            # 如果两个集合都为空，则:
            # 1) 随机给每辆车选一个动作
            random_actions = []
            for _ in vehicle_list:
                action_name = random.choice(list(self.actions.keys()))
                random_actions.append(action_name)

            # 2) 将这些动作转换为 (dx, dy) 格式
            random_action_vecs = [self.actions[a] for a in random_actions]

            # 3) 用 build_global_vector 构造 4N 向量
            random_vec = self.build_global_vector(ego_position, vehicle_list, random_action_vecs)

            # 4) 返回 (动作列表, 4N 向量)
            return random_actions, random_vec

        # 如果进入这里, 表示 risk_set 或 safe_set 不为空
        from itertools import product
        possible_actions = list(self.actions.values())  # 6个 (dx, dy)
        num_vehicle = len(vehicle_list)
        all_combos = product(possible_actions, repeat=num_vehicle)

        candidates = []
        for combo in all_combos:
            # combo 形如: ((dx1, dy1), (dx2, dy2), ...)
            # 构造 4N 向量(相对于 ego)
            global_vec = self.build_global_vector(ego_position, vehicle_list, combo)
            candidates.append(global_vec)

        # 计算 dist_safe, dist_risk
        scored_candidates = []
        for gv in candidates:
            if safe_set:
                dist_safe = min(self.distance_4N(gv, s, num_vehicle) for s in safe_set)
            else:
                dist_safe = None
            if risk_set:
                dist_risk = min(self.distance_4N(gv, r, num_vehicle) for r in risk_set)
            else:
                dist_risk = None
            scored_candidates.append((gv, dist_safe, dist_risk))

        # 排序: dist_safe降序, dist_risk升序
        def sort_key(item):
            _, ds, dr = item
            if ds is None and dr is None:
                return (1e9, 1e9)
            elif ds is None:
                return (0, dr)
            elif dr is None:
                return (-ds, 0)
            else:
                return (-ds, dr)

        scored_candidates.sort(key=sort_key)

        best_global_vec, best_ds, best_dr = scored_candidates[0]
        best_actions_list = self.parse_global_vector_to_actions(best_global_vec)

        return best_actions_list, best_global_vec
