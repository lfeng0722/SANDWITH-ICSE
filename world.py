import requests
import carla
from carla_controller import LaneKeepAndChangeController
import random
from websocket import create_connection, WebSocketException
import json
import threading
import math
import logging
import time
import numpy as np
import collections
import os

max_search_distance_for_destination = 200  # BFS search maximum distance
step_dist_for_destination = 2.0            # BFS step distance (meters)
max_search_distance_for_spawns = 50.0      # Used for multi-lane spawn selection
step_for_spawns = 1.0

log = logging.getLogger(__name__)

def fetch_localization_variable(url="http://127.0.0.1:5000/var"):
    """
    Fetch the latest localization data from the container via HTTP GET request
    :param url: The Flask interface address, default is 127.0.0.1:5000/var on this machine.
    :return: A JSON format dictionary of localization data, or None (in case of failure)
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # If the response status code is not 200, an exception will be raised
        data = response.json()
        return data
    except Exception as e:
        print("Error occurred while fetching variable data:", e)
        return None

def distance(loc1, loc2):
    """Simple Euclidean distance"""
    return math.sqrt(
        (loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2
    )

class MultiVehicleDemo:
    """
    Main functions:
      1) Generate ego + N automated vehicles (with LaneKeepAndChangeController)
      2) Attach collision sensors to all vehicles
         - When the ego vehicle collides => determine "ego initiates collision" or "others colliding with ego"
         - When other vehicles collide => apply emergency brake, but do not terminate
      3) In tick(), return (signals_list, ego_collision, self.collision, ego_cross_solid_line, ego_run_red_light)
      4) Provide set_destination() => use BFS to find the furthest waypoint in the same direction, save as self.ego_destination
      5) Provide get_controller(idx) => get the controller of the idx-th automated vehicle
    """

    def __init__(self, world, external_ads, websocket_url="ws://localhost:8888/websocket",
                 gps_offset=carla.Vector3D(x=1.0, y=0.0, z=0.5)):
        self.world = world
        self.population_size = 10
        self.map = world.get_map()
        self.ego_vehicle = None
        self.multi_vehicle_collision_count = 0
        self.vehicles = []            # Store automated vehicles
        self.controllers = None       # N LaneKeepAndChangeController
        self.url = websocket_url
        self.gps_offset = gps_offset
        self.ws = None
        self.vehicle_num = None
        self.ws_thread = None
        self.ws_running = False
        self.ws_receive_buffer = []
        self.ego_spawning_point = None
        self.ego_destination = None   # Set by set_destination
        self.collision = False
        self.external_ads = external_ads
        self.count = 0
        self.turn_on = False
        self.modules = [
            'Localization',
            'Routing',
            'Prediction',
            'Planning',
            'Control'
        ]
        self.side_collision_count_vehicle = 0  # Number of side collisions
        self.rear_collision_count_vehicle = 0  # Number of rear-end collisions
        self.collision_count_obj = 0

        # Flag whether the ego actively collided with someone
        self.ego_collision = False

        # Map boundary
        self.map_bounds = self._compute_map_bounds()

        # Collision sensor list
        self.collision_sensors = []

        # ----- LaneInvasion related -----
        self.ego_cross_solid_line = False  # Whether EGO crosses the solid line
        self.lane_invasion_sensor_ego = None

        # ----- Red-light-running detection related -----
        self.ego_run_red_light = False  # Whether EGO is detected to run a red light

        if self.external_ads:
            self._connect_websocket()

    # ========== Basic functions ==========

    def _connect_websocket(self):
        try:
            self.ws = create_connection(self.url)
            self.ws_running = True
            print(f"[INFO] Connected to WebSocket server: {self.url}")
            # Start a thread to receive messages
            self.ws_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.ws_thread.start()
        except WebSocketException as e:
            print(f"[ERROR] Unable to connect to WebSocket server: {e}")
            self.ws = None

    def _receive_messages(self):
        while self.ws_running:
            try:
                result = self.ws.recv()
                if result:
                    self.ws_receive_buffer.append(result)
            except WebSocketException as e:
                print(f"[ERROR] Error while receiving WebSocket message: {e}")
                self.ws_running = False
            except Exception as e:
                print(f"[ERROR] Unknown error: {e}")
                self.ws_running = False

    def _compute_map_bounds(self):
        """
        Simply get the x,y range of the map by map.generate_waypoints(2.0)
        """
        wps = self.map.generate_waypoints(2.0)
        if not wps:
            print("[WARN] generate_waypoints is empty. No map data?")
            return (0, 0, 0, 0)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for wp in wps:
            loc = wp.transform.location
            if loc.x < min_x: min_x = loc.x
            if loc.x > max_x: max_x = loc.x
            if loc.y < min_y: min_y = loc.y
            if loc.y > max_y: max_y = loc.y
        print(f"[INFO] map x range=({min_x:.1f},{max_x:.1f}), y range=({min_y:.1f},{max_y:.1f})")
        return (min_x, max_x, min_y, max_y)

    def get_map_bounds(self):
        return self.map_bounds

    # ========== Vehicle creation logic ==========

    def setup_vehicles(self, scenario_conf):
        """
        1) Select a point on the map as the ego spawn point
        2) Then find several nearby spawn points => generate corresponding number of automated vehicles
        3) Then it will be called in setup_vehicles_with_collision() + attach collision sensors
        """

        self.vehicle_num = scenario_conf["vehicle_num"]
        self.controllers = [None] * self.vehicle_num
        print('vehicle number: ', self.vehicle_num)
        self.ego_spawning_point = scenario_conf["ego_transform"]
        selected_spawns = scenario_conf["surrounding_transforms"]
        blueprint_library = self.world.get_blueprint_library()

        # Generate ego vehicle
        if not self.external_ads:
            bp_ego = blueprint_library.find("vehicle.tesla.model3")
            bp_ego.set_attribute("color", "0,0,255")  # Blue
            self.ego_vehicle = self.world.try_spawn_actor(bp_ego, self.ego_spawning_point)
        else:
            # If there is an external ADS, directly find an existing mkz_2017 as the EGO
            all_actors = self.world.get_actors()
            candidate_vehicles = all_actors.filter("vehicle.*")
            for v in candidate_vehicles:
                if "mkz_2017" in v.type_id:
                    self.ego_vehicle = v
                    break
            if not self.ego_vehicle:
                raise ValueError("Could not find the vehicle of type 'mkz_2017'. Please ensure the ego vehicle is spawned properly.")

            self.ego_vehicle.set_transform(self.ego_spawning_point)

        if self.ego_vehicle:
            # Generate multiple automated vehicles
            for i in range(self.vehicle_num):
                bp = blueprint_library.find("vehicle.tesla.model3")
                veh = self.world.try_spawn_actor(bp, selected_spawns[i])
                if veh:
                    self.vehicles.append(veh)
                else:
                    print(f"[WARN] Unable to spawn vehicle {i+1}, possibly due to position conflict.")
                    return False

            # N vehicles with LaneKeepAndChangeController
            for i in range(self.vehicle_num):
                ctrl = LaneKeepAndChangeController(self.vehicles[i])
                self.controllers[i] = ctrl

            return True
        else:
            return False

    def sample_testseed(self):
        """
        Randomly generate a scene test seed
        """
        v_num = random.choice([2, 3, 4])  # can be changed here
        while True:
            spawn_points = self.map.get_spawn_points()
            # Randomly select EGO spawn point
            ego_idx = random.randint(0, len(spawn_points)-1)
            ego_spawning = spawn_points.pop(ego_idx)

            # Find spawn points near EGO
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

    def _is_valid_side_lane(self, wp, side_wp):
        """
        Determine if the left and right lanes are Driving and have the same direction (same sign) as the current lane.
        """
        if not side_wp:
            return False
        if side_wp.lane_type != carla.LaneType.Driving:
            return False
        if wp.lane_id * side_wp.lane_id <= 0:
            return False
        return True

    def setup_vehicles_with_collision(self, scenario_conf):
        """
        External interface:
        1) call setup_vehicles first
        2) if successful => _setup_collision_sensors
        """
        success = self.setup_vehicles(scenario_conf)
        if success:
            self._setup_collision_sensors()
        return success

    # ========== Collision sensor logic + LaneInvasion sensor logic ==========

    def _setup_collision_sensors(self):
        """
        Attach collision sensors to the ego and all automated vehicles
        """
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # ego collision sensor
        if self.ego_vehicle:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor_ego = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
            # Pass this sensor reference in the callback
            sensor_ego.listen(lambda event, v=self.ego_vehicle, s=sensor_ego: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor_ego)
            print(f"[INFO] Ego Vehicle {self.ego_vehicle.id} collision sensor attached: {sensor_ego.id}")

            # ----- Attach LaneInvasionSensor to Ego vehicle -----
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.lane_invasion_sensor_ego = self.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor_ego.listen(self.lane_invasion_callback)
            print(f"[INFO] Ego Vehicle {self.ego_vehicle.id} lane invasion sensor attached: {self.lane_invasion_sensor_ego.id}")

        # other vehicles' collision sensors
        for veh in self.vehicles:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=veh)
            # Pass this sensor reference in the callback
            sensor.listen(lambda event, v=veh, s=sensor: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor)
            print(f"[INFO] Vehicle {veh.id} collision sensor attached: {sensor.id}")

    def lane_invasion_callback(self, event):
        """
        Triggered when the ego vehicle crosses the lane line. Determine whether it involves a solid lane marking.
        """
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid
            ]:
                # Once a solid line marking is detected => means crossing a solid line
                self.ego_cross_solid_line = True
                print("[INFO] EGO vehicle crosses a solid line!")
                break

    def collision_callback(self, event, vehicle, sensor):
        """
        Vehicle collision callback:
          - vehicle == self.ego_vehicle: Indicates EGO is involved in a collision
            1) Distinguish if the EGO initiates collision or is hit
            2) If EGO initiates collision, determine 'rear-end or side' + 'vehicle or object'
            3) Determine if it is a multi-vehicle collision
          - vehicle in self.vehicles and != self.ego_vehicle: Apply emergency brake to other NPC vehicles
        """
        # If already counted, you can decide whether to return immediately or not
        if self.collision_count_obj == 1 or self.multi_vehicle_collision_count == 1 or self.side_collision_count_vehicle == 1:
            pass

        if vehicle == self.ego_vehicle:
            # EGO Vehicle collision
            self.collision = True
            other_actor = event.other_actor
            actor_type_id = other_actor.type_id
            is_vehicle = actor_type_id.startswith("vehicle.")

            if not is_vehicle:
                # Collided with a stationary object
                print("[INFO] EGO collided with a stationary object => counting as a collision with a stationary object")
                self.collision_count_obj = 1
            else:
                # Collided with another vehicle
                self.ego_collision = True  # Mark EGO as initiator
                ego_transform = self.ego_vehicle.get_transform()
                ego_loc = ego_transform.location
                lane_width = self.map.get_waypoint(ego_loc).lane_width
                all_vehicles = self.world.get_actors().filter("vehicle.*")

                count_vehicles_in_lane = 0
                for v in all_vehicles:
                    if v.id == self.ego_vehicle.id:
                        continue
                    v_loc = v.get_transform().location
                    dist_2d = math.sqrt((v_loc.x - ego_loc.x) ** 2 + (v_loc.y - ego_loc.y) ** 2)
                    if dist_2d < lane_width:
                        count_vehicles_in_lane += 1

                if count_vehicles_in_lane >= 2:
                    print("[INFO] EGO multi-vehicle collision => multi_vehicle_collision_count = 1")
                    self.multi_vehicle_collision_count = 1
                else:
                    print("[INFO] EGO collided with another vehicle => side collision or rear-end collision")
                    self.side_collision_count_vehicle = 1
        else:
            # Other vehicles' collision => emergency braking
            if vehicle in self.vehicles:
                idx = self.vehicles.index(vehicle)
                controller = self.controllers[idx]
                if controller:
                    controller.brake()

            current_control = vehicle.get_control()
            stop_control = carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=current_control.steer
            )
            vehicle.apply_control(stop_control)

        # Collision sensor destroys itself after one collision
        sensor.stop()
        try:
            sensor.destroy()
            print(f"[INFO] Collision sensor {sensor.id} destroyed (works only once).")
        except:
            pass

        if sensor in self.collision_sensors:
            self.collision_sensors.remove(sensor)

    # ========== Red-light-running detection logic ==========

    def _detect_run_red_light(self):
        """
        Check if EGO continues to move during a red light (simple example: if the light is red and speed > 0.1, it is considered running a red light)
        """
        if not self.ego_vehicle:
            return False
        tlight = self.ego_vehicle.get_traffic_light()
        if tlight is None:
            return False

        if tlight.get_state() == carla.TrafficLightState.Red:
            vel = self.ego_vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            if speed > 0.1:
                return True
        return False

    # ========== tick & return signals ==========

    def tick(self):
        """
        Each frame:
         1) Let all automated vehicles execute LaneKeepAndChangeController.run_step()
         2) Check if EGO ran a red light
         3) Return (signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light)
        """
        signals_list = [None]*self.vehicle_num
        for i in range(self.vehicle_num):
            ctrl = self.controllers[i]
            if ctrl:
                control, signals = ctrl.run_step()
                self.vehicles[i].apply_control(control)
                signals_list[i] = signals
            else:
                signals_list[i] = None

        # If not yet recorded running a red light, check it
        if not self.ego_run_red_light:
            if self._detect_run_red_light():
                self.ego_run_red_light = True
                print("[INFO] EGO ran a red light!")

        return signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light

    # ========== Utility functions for main script ==========

    def reconnect(self):
        """
        Closes the websocket connection and re-creates it so that data can be received again
        """
        self.ws.close()
        self.ws = create_connection(self.url)
        return

    def check_module_status(self, modules):
        """
        Checks if all modules in a provided list are enabled
        """
        module_status = self.get_module_status()
        for module, status in module_status.items():
            if not status and module in modules:
                log.warning("Warning: Apollo module {} is not running!!!".format(module))
                self.enable_module(module)
                time.sleep(1)

    def get_module_status(self):
        """
        Returns a dict where the key is the name of the module
        and value is a bool based on the module's current status
        """
        self.reconnect()
        data = json.loads(self.ws.recv())  # first recv => SimControlStatus
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        # In an actual scenario, parse data and return module statuses, here just an example
        return {}

    def get_controller(self, idx):
        """
        Get the controller of the idx-th automated vehicle (0~N-1)
        """
        if idx < 0 or idx >= len(self.controllers):
            print(f"[WARN] get_controller: Index {idx} is out of range (0~{len(self.controllers)-1})!")
            return None
        return self.controllers[idx]

    def get_vehicle_positions(self):
        """
        Return the positions of all automated vehicles (excluding the ego vehicle) as a list
        """
        positions = []
        for v in self.vehicles:
            loc = v.get_location()
            positions.append(loc)
        return positions

    def destroy_all(self):
        """
        Destroy all sensors and vehicles at the end
        """
        # # 1) First set the sensor callbacks to empty functions to prevent any unhandled events from calling logic
        # for s in self.collision_sensors:
        #     s.listen(lambda event: None)

        if self.lane_invasion_sensor_ego:
            self.lane_invasion_sensor_ego.listen(lambda event: None)

        # 2) In synchronous/async mode, tick or sleep to wait for the underlying callbacks to clear
        for _ in range(3):
            self.world.wait_for_tick()

        # 3) stop and destroy
        for s in self.collision_sensors:
            try:
                s.stop()
                s.destroy()
            except:
                pass
        self.collision_sensors.clear()

        if self.lane_invasion_sensor_ego:
            try:
                self.lane_invasion_sensor_ego.stop()
                self.lane_invasion_sensor_ego.destroy()
            except:
                pass
            self.lane_invasion_sensor_ego = None

        for _ in range(3):
            self.world.wait_for_tick()

        # Finally destroy the vehicles
        for v in self.vehicles:
            try:
                v.destroy()
            except:
                pass
        self.vehicles.clear()

        # If there's still an ego_vehicle
        if not self.external_ads and self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except:
                pass
            self.ego_vehicle = None

        # Reset states
        self.collision = False
        self.ego_collision = False
        self.multi_vehicle_collision_count = 0
        self.rear_collision_count_vehicle = 0
        self.side_collision_count_vehicle = 0
        self.collision_count_obj = 0
        self.ego_run_red_light = False
        self.world.wait_for_tick()
        # self.world.tick()

    def enable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "START_MODULE", "value": module})
        )
        return

    def disable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "STOP_MODULE", "value": module})
        )
        return

    # ========== Destination setting example ==========

    def set_destination(self):
        """
        In the ego vehicle's current lane, do a simple BFS to find the furthest waypoint in the same direction => self.ego_destination
        If there's a WebSocket, you can send a RoutingRequest (optional)
        """
        if not self.ego_vehicle:
            print("[ERROR] ego_vehicle not spawned, cannot set_destination.")
            return

        # 1) Get the waypoint of the ego's position
        ego_loc = self.ego_vehicle.get_location()
        start_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if not start_wp:
            print("[ERROR] ego_vehicle waypoint is None, cannot set_destination.")
            return

        import collections
        queue = collections.deque()
        visited = set()
        queue.append((start_wp, 0.0))
        same_direction_wps = []

        init_lane_id = start_wp.lane_id
        init_lane_sign = 1 if init_lane_id >= 0 else -1

        while queue:
            cur_wp, dist_so_far = queue.popleft()
            if cur_wp in visited:
                continue
            visited.add(cur_wp)

            same_direction_wps.append((cur_wp, dist_so_far))

            if dist_so_far > max_search_distance_for_destination:
                continue

            nxt_wps = cur_wp.next(step_dist_for_destination)
            for nxt_wp in nxt_wps:
                nxt_lane_sign = 1 if nxt_wp.lane_id >= 0 else -1
                if nxt_lane_sign == init_lane_sign:
                    dist_increment = cur_wp.transform.location.distance(nxt_wp.transform.location)
                    new_dist = dist_so_far + dist_increment
                    if new_dist <= (max_search_distance_for_destination + step_dist_for_destination):
                        queue.append((nxt_wp, new_dist))

        if not same_direction_wps:
            print("[WARNING] No same-direction waypoints found => set_destination failed")
            return

        # find the furthest waypoint
        furthest_wp, furthest_dist = max(same_direction_wps, key=lambda x: x[1])
        self.ego_destination = furthest_wp.transform.location
        print(f"[INFO] set_destination: target point (x={self.ego_destination.x:.2f}, y={self.ego_destination.y:.2f}), dist={furthest_dist:.1f}m")

        # If you have WebSocket => sendRoutingRequest (optional)
        apollo_data = fetch_localization_variable()
        if self.ws and apollo_data is not None and 'position' in apollo_data:
            try:
                yaw_deg = self.ego_vehicle.get_transform().rotation.yaw
                yaw_rad = math.radians(yaw_deg)

                msg = {
                    "type": "SendRoutingRequest",
                    "start": {
                        "x": apollo_data['position']['x'],
                        "y": apollo_data['position']['y'],
                        "z": apollo_data['position']['z'],
                        "heading": -yaw_rad,
                    },
                    "end": {
                        "x": self.ego_destination.x,
                        "y": -self.ego_destination.y,
                        "z": apollo_data['position']['z'],
                    },
                    "waypoint": "[]",
                }
                self.ws.send(json.dumps(msg))
                print("[INFO] Routing request sent:", json.dumps(msg))
            except WebSocketException as e:
                print(f"[ERROR] WebSocket error while sending RoutingRequest: {e}")
            except Exception as e:
                print(f"[ERROR] Internal error in set_destination: {e}")

    def close_connection(self):
        """
        If there's a websocket connection, close it at the end
        """
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
                print("[INFO] WebSocket connection closed.")
            except Exception as e:
                print(f"[ERROR] Error occurred while closing WebSocket connection: {e}")
