import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
import random
import time
import base64
import requests
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from collections import deque
from metadrive.utils import print_source
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils import generate_gif
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
from metadrive.component.lane.straight_lane import StraightLane
# env=MetaDriveEnv(dict(map="S", traffic_density=0))
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
import copy
import json

action_space = np.array([[0,3], [1,3], [2,3], [0,0], [1,0], [2,0]])

class CAREnv:
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.v_max = 1
        self.agents = [f'agent{i+1}' for i in range(num_agents)]
        self.ego_vehicle = None
        self.history_positions = [[] for _ in range(num_agents)]
        #
        # self.action_space = {
        #     'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        #     'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        #     'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        # }  # action represents [a_x,a_y]
        # self.observation_space = {
        #     'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
        #     'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
        #     'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        # }


        self.vehicles = []
        self.max_x_diff = None
        self.max_y_diff = None

    def reset(self):
        SEED = random.randint(1, 1000)
        random.seed(SEED)
        # road_type = ['S','O','y','T','C']
        road = 'S'
        # road = random.choice(road_type)
        # print(road)
        #
        # with open('road_config.json', 'r') as config_file:
        #     road_configs = json.load(config_file)
        # road_config = road_configs[road]
        # spawn_point = random.sample(road_config,3)
        # print(spawn_point)

        self.env = MultiAgentMetaDrive(dict(
            map_config=dict(config=road, type="block_sequence", lane_num=4),
            traffic_density=0,
            is_multi_agent=True,
            num_agents=self.num_agents + 1,  # 由于 agent0 固定，所以要 +1
            discrete_action=True,
            use_multi_discrete=True,
            agent_policy=LaneChangePolicy,
            random_spawn_lane_index=True,
            agent_configs={
                "agent0": dict(use_special_color=True, enable_reverse=True),  # 固定的 agent0
                # **{
                #     f"agent{i + 1}": dict(spawn_lane_index=(spawn_point[i][0], spawn_point[i][1], 0))
                #     for i in range(self.num_agents)
                # }
            }
        ))

        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        self.env.reset()

        xmin, xmax, ymin, ymax = self.env.current_map.get_boundary_point()
        self.max_x_diff = xmax - xmin
        self.max_y_diff = ymax - ymin

        self.vehicles = []
        all_vehicles = self.env.engine.get_objects()

        for k, v in all_vehicles.items():
            self.vehicles.append(v)
        self.ego_vehicle = self.vehicles.pop(0)

        for i in range(self.num_agents):
            self.multi_current_pos.append(self.vehicles[i].position)
            self.multi_current_vel.append(np.zeros(2))  # initial velocity = [0,0]
        self.multi_current_pos.append(self.ego_vehicle.position)

        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):

        last_d2target = []
        # print(actions)
        # time.sleep(0.1)
        collision = []
        out_of_road = []
        pos_taget = self.multi_current_pos[-1]
        # print(actions)
        selected_actions = [action_space[np.argmax(out)] for out in actions]
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            last_d2target.append(np.linalg.norm(pos - pos_taget))

            self.multi_current_vel[i] = selected_actions[i]


        p = self.env.engine.get_policy(self.ego_vehicle.name)
        try:
            a0 = [p.act(True)[0], p.act(True)[1]]
        except Exception as e:
            a0 = [0, 0]

        agent_actions = {
            'agent0': a0
        }
        for i, agent in enumerate(self.env.engine.agents.keys()):
            # print(agent)
            if agent in self.agents:
                agent_actions[agent] = self.multi_current_vel[i-1]
        # print(agent_actions)

        _, _, tm, _, info = self.env.step(agent_actions)


        for i in range(self.num_agents):
            self.multi_current_pos[i]=self.vehicles[i].position
        self.multi_current_pos[-1] = self.ego_vehicle.position

        # 遍历每个 agent，检查是否在 info 中且有 crash_vehicle 信息
        for agent in self.agents:
            if agent in info and "crash_vehicle" in info[agent] and info[agent]["crash_vehicle"]:
                collision.append(True)
            else:
                collision.append(False)
        for agent in self.agents:
            if agent in info and "out_of_road" in info[agent] and info[agent]["out_of_road"]:
                out_of_road.append(True)
            else:
                out_of_road.append(False)
        rewards, dones = self.cal_rewards_dones(collision, out_of_road, last_d2target)
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def get_multi_obs(self):
        total_obs = []

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.vehicles[i].velocity
            S_uavi = [
                pos[0] / self.max_x_diff,
                pos[1] / self.max_y_diff,
                vel[0],
                vel[1]
            ]  # dim 4
            # print(S_uavi)
            S_team = []  # dim 4 for 3 agents 1 target
            S_target = []  # dim 2
            for j in range(self.num_agents):
                if j != i:
                    pos_other = self.multi_current_pos[j]
                    vel_other = self.vehicles[j].velocity
                    S_team.extend([pos_other[0] / self.max_x_diff, pos_other[1] / self.max_y_diff, vel_other[0], vel_other[1]])

            pos_target = self.multi_current_pos[-1]
            # d = np.linalg.norm(pos - pos_target)
            # theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
            ego_vel = self.ego_vehicle.velocity
            S_target.extend([pos_target[0] / self.max_x_diff, pos_target[1] / self.max_y_diff, ego_vel[0],ego_vel[1]])

            single_obs = [S_uavi, S_team, S_target]

            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def cal_rewards_dones(self, IsCollied, out_of_road, last_d):
        """
        计算多机围捕环境下的所有智能体的奖励和是否结束 dones

        参数：
          IsCollied:   list(bool)，长度 = self.num_agents, 指示每个agent是否碰撞
          out_of_road: list(bool)，长度 = self.num_agents, 指示每个agent是否越界
          last_d:      list(float)，长度 = self.num_agents, 表示上一时刻各agent到被捕者的距离

        返回：
          rewards: np.array(float)，长度 = self.num_agents, 每个agent的即时奖励
          dones:   list(bool)，长度 = self.num_agents, 每个agent是否结束
        """
        # 创建空列表/数组
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 一些超参数(可以自行调节)
        mu1 = 0.7  # r_near：单机靠近目标的奖励系数
        mu2 = 0.4  # r_safe：安全(碰撞)惩罚系数
        mu3 = 0.01  # r_multi_stage：多阶段围捕奖励系数
        mu4 = 5.0  # r_finish：最终完成围捕奖励系数

        # 定义两个阈值（和你原代码一致）
        d_capture = 3.5
        d_limit = 8.0

        # ========== 1. 单机靠近目标的奖励/惩罚 ==========
        # 假设最后一个是被捕者，前面都是追捕者(或载具)
        for i in range(self.num_agents):
            pos_i = self.multi_current_pos[i]
            pos_e = self.multi_current_pos[-1]  # 被捕者坐标
            d_i = np.linalg.norm(pos_e - pos_i)

            # 如果该载具比上一时刻更接近被捕者，则给正奖励，否则负
            if d_i < last_d[i]:
                rewards[i] += mu1 * 1.0
            else:
                rewards[i] -= mu1 * 1.0

        # ========== 2. 碰撞和越界的惩罚 ==========
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10.0  # 碰撞严重惩罚
                dones[i] = True  # 碰撞的agent直接结束
            else:
                r_safe = 0.0
            rewards[i] += mu2 * r_safe

            if out_of_road[i]:
                r_out = -100.0
            else:
                r_out = 0.0
            rewards[i] += mu2 * r_out

        # ========== 3. 多阶段围捕奖励 ==========

        # 3.1 拆分追捕者和被捕者
        # 这里默认前 (k = self.num_agents-1) 个是追捕者，最后1个是被捕者
        pursuers = self.multi_current_pos[:-1]  # shape: (k, 2)
        pe = self.multi_current_pos[-1]  # 被捕者坐标
        k = len(pursuers)  # 追捕者数量

        # 定义一个辅助判断函数，避免浮点误差
        def almost_equal(a, b, eps=1e-9):
            return abs(a - b) < eps

        # 3.2 计算多边形面积 (对应旧代码里 S4)
        S_polygon = cal_polygon_area(pursuers)

        # 3.3 计算“子面积之和” Sum_S (对应旧代码里 S1 + S2 + S3)
        Sum_S = 0.0
        for i in range(k):
            j = (i + 1) % k
            Sum_S += cal_triangle_S(pursuers[i], pursuers[j], pe)

        # 3.4 计算距离
        distances = [np.linalg.norm(p - pe) for p in pursuers]
        Sum_d = sum(distances)
        # 上一时刻追捕者到被捕者距离总和（last_d仅取前k个追捕者）
        Sum_last_d = sum(last_d[:k])

        # 多阶段逻辑(Stage-1,2,3)，和原先代码相同，只是扩展到 k 个追捕者
        # Stage-1: track
        if (Sum_S > S_polygon
                and Sum_d >= d_limit
                and all(d >= d_capture for d in distances)):

            r_track = - Sum_d / max(distances)
            rewards[:k] += mu3 * r_track

        # Stage-2: encircle
        elif (Sum_S > S_polygon
              and (Sum_d < d_limit or any(d >= d_capture for d in distances))):

            # 这里分母从 3 改成 k，以适应“任意 k”
            r_encircle = - (1.0 / k) * math.log((Sum_S - S_polygon) + 1.0)
            rewards[:k] += mu3 * r_encircle

        # Stage-3: capture
        elif (almost_equal(Sum_S, S_polygon)  # Sum_S与S_polygon几乎相等
              and any(d > d_capture for d in distances)):

            # 这里的 3 同样改为 k
            r_capture = math.exp((Sum_last_d - Sum_d) / (k * self.v_max))
            rewards[:k] += mu3 * r_capture

        # ========== 4. 围捕成功的结束条件 ==========
        # 原来： if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3])
        if (almost_equal(Sum_S, S_polygon)
                and all(d <= d_capture for d in distances)):
            # 围捕成功，大额奖励
            rewards[:k] += mu4 * 10.0
            dones = [True] * self.num_agents

        return rewards, dones


    # def close(self):
    #     plt.close()
