#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import carla
import math
import time
import random
import csv  # Used for writing CSV files
import torch
import numpy as np
from utility import average_population_distance, parents_selection, next_gen_selection
from collections import deque

# ===== Adjust these imports according to your project's actual situation =====
from world import MultiVehicleDemo
from utility import position_scaler, state_encoder, has_passed_destination
from ART_fuzzer import ARTSelector4D, action_trans
from MADDPG import MADDPG
from fuzzer_set_mapper import FSM_mapper
from offline_searcher import CombinedGA

# Define action space
action_space = np.array([
    [0, 3],
    [1, 3],
    [2, 3],
    [0, 0],
    [1, 0],
    [2, 0]
], dtype=np.int32)

action_space_world = (
    'break',
    'accelerate',
    'right_change_acc',
    'right_change_dec',
    'left_change_acc',
    'left_change_dec'
)

# Hyperparameters
TIME_STEP = 0.05
population_size = 10


def main(map, result_record):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    if map != 'Town01':
        for actor in world.get_actors():
            if isinstance(actor, carla.TrafficLight):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)

    # Initialize MultiVehicleDemo
    external_ads = True
    demo = MultiVehicleDemo(world, external_ads)

    # ===== Newly added: camera-related global variables or methods =====
    import cv2
    import os
    from queue import Queue

    # Prepare camera blueprint
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # You may change the resolution here if needed
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '360')
    camera_bp.set_attribute('fov', '90')

    def camera_callback(image, image_queue):
        """
        Convert the image data from camera.listen callback to NumPy array
        and store it into the queue.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_queue.put(array)

    # Create an instance of ARTSelector4D (optional)
    ART_selector = ARTSelector4D(scale_pos=900.0, scale_action=1.0)

    # Wrap safe_set / risk_set in deques (if needed)
    MAX_ART_SIZE = 100000
    ART_selector.safe_set_1 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.safe_set_2 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.safe_set_3 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.safe_set_4 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.risk_set_1 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.risk_set_2 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.risk_set_3 = deque([], maxlen=MAX_ART_SIZE)
    ART_selector.risk_set_4 = deque([], maxlen=MAX_ART_SIZE)

    # Get map boundaries
    map_bounds = demo.get_map_bounds()
    min_x, max_x, min_y, max_y = map_bounds
    max_diff_x = max_x - min_x
    max_diff_y = max_y - min_y

    # Some control flags
    ART_ebabled = True
    Random_enabled = False
    GA_enabled = True
    recording_result = result_record

    # Statistics
    Test_buget = 400
    number_game = 1
    side_total = 0
    red_light_cnt = 0

    obj_total = 0
    multi_collision_total = 0
    timeout = 0
    cross_solid_count = 0
    TOP5 = 0
    speed_cnt = 0
    # Track vehicles that have received ART actions and have not yet ended
    ART_action_tracker = {}

    try:
        if GA_enabled:
            current_generation = 1
            GA = CombinedGA(carla_map=world.get_map())
            GA.sample_initial_population()
            scenario_confs = GA.population

            fitness = {
                "safety_violation": [],  # {veh_id: [(step, x, y, z), ...], ...}
                "diversity": [],         # Will be filled at the end of the round with side/rear/obj etc.
                "ART_trigger_time": []
            }

        for attempt_idx in range(Test_buget):
            ART_trigger_time = 0
            abnormal_case = False  # Indicates whether this round is abnormal

            # ===== (1) Spawn vehicles =====
            if not GA_enabled:
                scenario_conf = demo.sample_testseed()
            else:
                if len(scenario_confs) == population_size:
                    scenario_conf = scenario_confs[(number_game - 1) % population_size]["position_info"]
                else:
                    scenario_conf = scenario_confs[((number_game - 1) % population_size) + population_size]["position_info"]

            # Load MARL weight
            if not Random_enabled:
                n_agents = scenario_conf["vehicle_num"]
                n_actions = 6
                actor_dims = [(n_agents + 1) * 4] * n_agents
                critic_dims = sum(actor_dims)
                maddpg_agents = MADDPG(
                    actor_dims, critic_dims, n_agents, n_actions,
                    fc1=128, fc2=128,
                    alpha=0.00001, beta=0.02, scenario='MARL',
                    chkpt_dir=''
                )
                maddpg_agents.load_checkpoint()

            success = demo.setup_vehicles_with_collision(scenario_conf)
            if not success:
                abnormal_case = True
                print("[ERROR] Failed to generate vehicle, skip this episode.")
                # continue

            print(f"[INFO] Vehicle is generated successfully and controlled ({attempt_idx + 1} round).")

            vehicle_list = [demo.get_controller(i) for i in range(demo.vehicle_num)]

            # If not external_ads, then let ego autopilot
            if not demo.external_ads:
                tm = client.get_trafficmanager(8000)
                demo.ego_vehicle.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(demo.ego_vehicle, 5)

            # Used to detect whether EGO stays still for a long time
            stationary_time = 0.0
            stationary_threshold = 25
            speed_threshold = 0.2
            timeout_cnt = 0
            speeding = 0

            demo.count = attempt_idx + 1
            if external_ads:
                for mod in demo.modules:
                    demo.enable_module(mod)

            # Let the world stabilize a bit
            time.sleep(2)

            # ========== (1.1) Create a dict to record the trajectory and violations in this round ==========
            attempt_data = {
                "vehicle_trajectories": {},  # {veh_id: [(step, x, y, z), ...], ...}
                "ego_trajectory": [],        # [(step, x, y, z), (step, x, y, z), ...]
                "violations": {}             # Will be filled after the round ends
            }

            # ======================================================
            # (Core change) Spawn the camera at the spectator's position,
            # instead of attaching it to the EGO vehicle
            # ======================================================
            if recording_result:
                image_queue = Queue()
                episode_frames = []  # Used to store image frames (NumPy arrays) for the entire round

                # Get the current spectator transform
                spectator = world.get_spectator()
                spectator_transform = spectator.get_transform()

                # Spawn a camera at this transform
                camera_transform = carla.Transform(
                    spectator_transform.location,
                    spectator_transform.rotation
                )
                camera = world.spawn_actor(camera_bp, camera_transform)
                camera.listen(lambda data: camera_callback(data, image_queue))

                # Clear any frames in the queue
                while not image_queue.empty():
                    image_queue.get_nowait()

                save_dir = f"recording/episode_{number_game}"
                os.makedirs(save_dir, exist_ok=True)

            # ========== (2) Start the step loop ============
            start_loc = None
            for step in range(50000):
                world.wait_for_tick()
                if abnormal_case:
                    break

                if step < 500:
                    ego_vel = demo.ego_vehicle.get_velocity()
                    speed_ego = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
                    if speed_ego > 5:
                        print("[WARN] EGO overspeed, abnormal case, end early.")
                        abnormal_case = True

                if step == 500 and demo.external_ads:
                    start_loc = demo.ego_vehicle.get_location()
                    demo.set_destination()

                # Execute actions/detections after 1050 frames
                if step > 500:
                    v_limt = demo.ego_vehicle.get_speed_limit()

                    # If recording, retrieve the latest frame from the camera queue if available
                    if recording_result:
                        frame = image_queue.get()
                        episode_frames.append(frame)
                        cv2.imwrite(os.path.join(save_dir, f"frame_{step}.png"), frame)
                        # If you move the spectator each frame, you can also update the camera accordingly
                        spectator_transform = spectator.get_transform()
                        camera.set_transform(spectator_transform)

                    inserted = False
                    # ========== (2.1) Store trajectory (example: store every 10 frames) ==========
                    if step % 10 == 0:
                        # Record NPC vehicle trajectories
                        for vehicle_id, controller in enumerate(vehicle_list):
                            veh = demo.vehicles[vehicle_id]
                            loc = veh.get_location()
                            attempt_data["vehicle_trajectories"].setdefault(vehicle_id, []).append(
                                (step, loc.x, loc.y, loc.z)
                            )
                        # Record EGO vehicle trajectory
                        ego_loc = demo.ego_vehicle.get_location()
                        attempt_data["ego_trajectory"].append(
                            (step, ego_loc.x, ego_loc.y, ego_loc.z)
                        )

                    signals_list, ego_collision, all_collision, cross_solid_line, red_light = demo.tick()

                    # EGO stays still detection
                    ego_vel = demo.ego_vehicle.get_velocity()
                    speed_ego = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
                    if speed_ego > v_limt:
                        speeding = 1
                    if speed_ego < speed_threshold:
                        stationary_time += TIME_STEP
                    else:
                        stationary_time = 0.0

                    if stationary_time >= stationary_threshold and not all_collision:
                        timeout_cnt = 1
                        print(f"[MAIN] EGO has been motionless for {stationary_time:.1f} seconds.")
                    if stationary_time >= stationary_threshold:
                        break

                    # Check if EGO reached the destination
                    if demo.ego_destination is not None:
                        near_dest, pass_dest = has_passed_destination(demo.ego_vehicle, demo.ego_destination)
                        if step != 0 and demo.external_ads and near_dest:
                            print("Arrived, ending episode.")
                            break
                        else:
                            if pass_dest:
                                for mod in demo.modules:
                                    demo.enable_module(mod)
                                print("Pass destination error.")
                                abnormal_case = True

                    positions = demo.get_vehicle_positions()
                    ego_position = np.array([demo.ego_vehicle.get_location().x,
                                             demo.ego_vehicle.get_location().y], dtype=np.float32)

                    # Detect if the ART action has ended
                    if ART_ebabled:
                        for i, signals in enumerate(signals_list):
                            if signals is None:
                                continue
                            veh = demo.vehicles[i]
                            if veh in ART_action_tracker and ART_action_tracker[veh]["in_progress"]:
                                done_flag = False
                                if (signals["lane_change_done"] or signals["acceleration_done"] or
                                        signals["deceleration_done"] or signals["brake_done"] or
                                        signals["release_brake_done"]):
                                    done_flag = True

                                if done_flag:
                                    cur_loc = veh.get_location()
                                    ego_loc = demo.ego_vehicle.get_location()
                                    lane_width = demo.map.get_waypoint(ego_loc).lane_width
                                    dist_ego = math.dist([cur_loc.x, cur_loc.y],
                                                         [ego_loc.x, ego_loc.y])
                                    vectors = ART_action_tracker[veh]["vector"]
                                    if not inserted:
                                        if dist_ego > 2 * lane_width:
                                            vec_len_div_4 = len(vectors) / 4
                                            if vec_len_div_4 == 1:
                                                ART_selector.safe_set_1.append(vectors)
                                            elif vec_len_div_4 == 2:
                                                ART_selector.safe_set_2.append(vectors)
                                            elif vec_len_div_4 == 3:
                                                ART_selector.safe_set_3.append(vectors)
                                            else:
                                                ART_selector.safe_set_4.append(vectors)
                                        else:
                                            vec_len_div_4 = len(vectors) / 4
                                            if vec_len_div_4 == 1:
                                                ART_selector.risk_set_1.append(vectors)
                                            elif vec_len_div_4 == 2:
                                                ART_selector.risk_set_2.append(vectors)
                                            elif vec_len_div_4 == 3:
                                                ART_selector.risk_set_3.append(vectors)
                                            else:
                                                ART_selector.risk_set_4.append(vectors)
                                        inserted = True

                                    ART_action_tracker[veh]["in_progress"] = False

                        # == ART + MARL actions ==
                        lane_width = demo.map.get_waypoint(demo.ego_vehicle.get_location()).lane_width
                        # Filter vehicles close to EGO for ART testing
                        ART_vehicle_list = []
                        ART_action_list = []
                        for i, pos in enumerate(positions):
                            dist_e = math.dist([pos.x, pos.y], ego_position)
                            if dist_e <= 2 * lane_width:
                                ART_vehicle_list.append(demo.vehicles[i])
                                ART_action_list.append(vehicle_list[i])

                        # If there are vehicles in ART range and ART is enabled
                        if ART_vehicle_list:
                            art_action, vectors = ART_selector.choose_actions_for_all_vehicles(
                                ego_position, ART_vehicle_list
                            )
                            for i in range(len(art_action)):
                                veh = ART_vehicle_list[i]
                                veh_vel = veh.get_velocity()
                                speed_veh = math.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2 + veh_vel.z ** 2)
                                if speed_veh > 0:
                                    if veh not in ART_action_tracker or not ART_action_tracker[veh]["in_progress"]:
                                        ART_trigger_time += 1
                                        action_trans(ART_action_list[i], art_action[i])
                                        ART_action_tracker[veh] = {
                                            "vector": vectors,
                                            "in_progress": True
                                        }

                    # Example of letting spectator follow the EGO vehicle
                    spectator = world.get_spectator()
                    trans = demo.ego_vehicle.get_transform()
                    loc = trans.location
                    yaw_deg = trans.rotation.yaw
                    yaw_rad = math.radians(yaw_deg)
                    offset_x = -10.0
                    offset_z = 5.0
                    cam_x = loc.x + offset_x * math.cos(yaw_rad)
                    cam_y = loc.y + offset_x * math.sin(yaw_rad)
                    cam_z = loc.z + offset_z
                    spectator.set_transform(carla.Transform(
                        carla.Location(cam_x, cam_y, cam_z),
                        carla.Rotation(pitch=-20, yaw=yaw_deg)
                    ))

                    # Let other vehicles use MARL
                    if not Random_enabled:
                        car_position = [np.array([pos.x, pos.y]) for pos in positions]
                        states = state_encoder(
                            demo.ego_vehicle,
                            demo.vehicles,
                            ego_position,
                            car_position,
                            max_diff_x,
                            max_diff_y
                        )
                        actions = maddpg_agents.choose_action(states)
                        selected_actions = [action_space[np.argmax(out)] for out in actions]
                        MARL_action = [FSM_mapper(a) for a in selected_actions]
                        if ART_ebabled:
                            for i, veh in enumerate(demo.vehicles):
                                if veh not in ART_action_tracker or not ART_action_tracker[veh]["in_progress"]:
                                    idx = demo.vehicles.index(veh)
                                    action_trans(vehicle_list[idx], MARL_action[idx])
                        else:
                            for i, veh in enumerate(demo.vehicles):
                                idx = demo.vehicles.index(veh)
                                action_trans(vehicle_list[idx], MARL_action[idx])
                    else:
                        for i, veh in enumerate(demo.vehicles):
                            idx = demo.vehicles.index(veh)
                            action_trans(vehicle_list[idx], random.choice(action_space_world))

            for _ in range(300):
                world.wait_for_tick()

            # Destroy the camera for this round
            if recording_result:
                if camera is not None:
                    camera.stop()
                    camera.destroy()

            # ========== (3) End of round checks, is it abnormal? ==========
            end_loc = demo.ego_vehicle.get_location()
            if start_loc is not None:
                distance_to_start = math.dist([start_loc.x, start_loc.y], [end_loc.x, end_loc.y])
            else:
                distance_to_start = 0.0

            print("Moved distance: ", distance_to_start)
            if distance_to_start < 1:
                print("[WARNING] EGO start and destination are < 1m apart, abnormal!")
                abnormal_case = True

            if not abnormal_case:
                side_total += demo.side_collision_count_vehicle
                obj_total += demo.collision_count_obj
                multi_collision_total += demo.multi_vehicle_collision_count
                red_light_cnt += red_light
                speed_cnt += speeding
                cross_solid_count += cross_solid_line
                timeout += timeout_cnt

                Safety_failure = side_total + obj_total + multi_collision_total + red_light_cnt + speed_cnt
                progress_failure = timeout
                total_failure = Safety_failure + progress_failure
                if total_failure >= 10 and TOP5 == 0:
                    TOP5 = number_game

                attempt_data["violations"] = {
                    "side_collision": demo.side_collision_count_vehicle,
                    "object_collision": demo.collision_count_obj,
                    "multi_collision": demo.multi_vehicle_collision_count,
                    "cross_solid_line": cross_solid_line,
                    "Time-out": timeout_cnt,
                    "break red light": red_light,
                    "speeding": speed_cnt
                }

                if recording_result:
                    # If there is a safety violation or dangerous behavior, keep the entire set of frames
                    if demo.multi_vehicle_collision_count > 0:
                        new_name = f"recording/multi_vehicle_collision_episode_{number_game}"
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found multi collision, images saved: {new_name}")

                    if demo.side_collision_count_vehicle > 0:
                        new_name = f"recording/side_cnt_episode_{number_game}"
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found side collision, images saved: {new_name}")

                    elif demo.collision_count_obj > 0:
                        new_name = f"recording/obj_cnt_episode_{number_game}"
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found object collision, images saved: {new_name}")

                    elif cross_solid_line > 0:
                        new_name = f"recording/cross_solid_line_episode_{number_game}"
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found cross solid line, images saved: {new_name}")

                    elif red_light > 0:
                        new_name = f"recording/red_light_episode_{number_game}"
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found red light violation, images saved: {new_name}")

                    elif timeout_cnt > 0:
                        new_name = f"recording/timeout_cnt_episode_{number_game}"
                        os.makedirs(save_dir, exist_ok=True)
                        os.rename(save_dir, new_name)
                        print(f"[INFO] Found timeout, images saved: {new_name}")

                    else:
                        print("No violation, removing directory.")

                print("Number of game:", number_game)

                if GA_enabled:
                    print("This is generation", current_generation)

                    if (demo.side_collision_count_vehicle == 0 and
                        demo.multi_vehicle_collision_count == 0 and
                        timeout_cnt == 0 and
                        demo.collision_count_obj == 0):
                        indi = {
                            "position_info": scenario_conf,
                        }
                        GA.safe_set.append(indi)
                        fitness["safety_violation"].append(1)
                    else:
                        fitness["safety_violation"].append(-1)

                    fitness["ART_trigger_time"].append(-ART_trigger_time)

                    if (number_game % population_size == 0 and
                        number_game % (2 * population_size) != 0):
                        for individual in scenario_confs:
                            eu_dis = average_population_distance(individual, scenario_confs)
                            fitness["diversity"].append(-eu_dis)
                        parents = parents_selection(fitness, scenario_confs, population_size)
                        for i in range(0, population_size, 2):
                            parent_1, parent_2 = parents[i], parents[i + 1]
                            child_1, child_2 = GA.crossover_individuals(parent_1, parent_2)
                            child_1 = GA.mutation(child_1)
                            child_2 = GA.mutation(child_2)
                            scenario_confs.append(child_1)
                            scenario_confs.append(child_2)
                        fitness['diversity'] = []

                    if number_game % (2 * population_size) == 0:
                        for individual in scenario_confs:
                            eu_dis = average_population_distance(individual, scenario_confs)
                            fitness["diversity"].append(-eu_dis)
                        scenario_confs = next_gen_selection(fitness, scenario_confs, population_size)
                        for key in fitness:
                            fitness[key].clear()
                        current_generation += 1

                number_game += 1

            else:
                if recording_result:
                    shutil.rmtree(save_dir)
                # If abnormal, try to resample for GA
                if GA_enabled:
                    if len(scenario_confs) == population_size:
                        scenario_confs[(number_game - 1) % population_size] = GA.resample()
                    else:
                        scenario_confs[((number_game - 1) % population_size) + population_size] = GA.resample()

                print(f"[INFO] {attempt_idx + 1} round is abnormal, not counted.")

            # Destroy vehicles
            demo.destroy_all()

    except KeyboardInterrupt:
        print("User manually ended the process.")

    except Exception as e:
        print("Error:", e)

    finally:
        print("Multi-vehicle collision: ", multi_collision_total)
        print("Side collision: ", side_total)
        print("Object Collision: ", obj_total)
        print("Timeout: ", timeout)
        print("Cross solid line: ", cross_solid_count)
        print("Red light break: ", red_light_cnt)
        print("Speeding: ", speed_cnt)

        if number_game > 0:
            print("Safety failure:", Safety_failure / number_game)
            print("Progress failure:", progress_failure / number_game)
        print("TOP-10:", TOP5)

        # Cleanup
        demo.destroy_all()
        demo.close_connection()
        print("Cleanup done.")


if __name__ == "__main__":
    map = 'Town04'
    main(map=map, result_record=False)
