import carla
import time
import json
import os
import math
import numpy as np
import tempfile
from utils.vlm_agent import DummyVLM
try:
    from agents.navigation.basic_agent import BasicAgent
except Exception:
    BasicAgent = None


class CarlaController:
    def __init__(self, host="localhost", port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.vehicle = None
        self.rgb_camera = None
        self.depth_camera = None
        self.sensors = []

        # Last frames
        self.last_rgb_image = None
        self.last_depth_raw = None
        self.last_depth_width = None
        self.last_depth_height = None

        # Vision-language interface
        self.vlm = DummyVLM()

    # =============================================================
    # VEHICLE & SENSOR SETUP
    # =============================================================
    def spawn_vehicle(self):
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned vehicle at {spawn_point.location}")
        return self.vehicle

    def spawn_rgbd(self, prefix='output', image_size_x=800, image_size_y=600, fov=90, sensor_tick=None):
        if self.vehicle is None:
            raise RuntimeError("Vehicle must be spawned before attaching sensors")

        blueprint_lib = self.world.get_blueprint_library()
        rgb_bp = blueprint_lib.find('sensor.camera.rgb')
        depth_bp = blueprint_lib.find('sensor.camera.depth')

        for bp in (rgb_bp, depth_bp):
            bp.set_attribute('image_size_x', str(image_size_x))
            bp.set_attribute('image_size_y', str(image_size_y))
            bp.set_attribute('fov', str(fov))
            if sensor_tick is not None:
                bp.set_attribute('sensor_tick', str(sensor_tick))

        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_dir = f"{prefix}_rgb"
        depth_dir = f"{prefix}_depth"
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        self.rgb_camera = self.world.spawn_actor(rgb_bp, spawn_point, attach_to=self.vehicle)
        self.depth_camera = self.world.spawn_actor(depth_bp, spawn_point, attach_to=self.vehicle)

        def _rgb_callback(image):
            self.last_rgb_image = image
            image.save_to_disk(os.path.join(rgb_dir, "%06d.png" % image.frame))

        def _depth_callback(image):
            self.last_depth_raw = bytes(image.raw_data)
            self.last_depth_width = image.width
            self.last_depth_height = image.height
            vis = image.copy()
            vis.convert(carla.ColorConverter.Depth)
            vis.save_to_disk(os.path.join(depth_dir, "%06d.png" % image.frame))

        self.rgb_camera.listen(_rgb_callback)
        self.depth_camera.listen(_depth_callback)

        self.sensors += [self.rgb_camera, self.depth_camera]
        print(f"Spawned RGB-D sensors -> {rgb_dir}, {depth_dir}")
        return self.rgb_camera, self.depth_camera

    # =============================================================
    # DEPTH REPROJECTION
    # =============================================================
    def carla_depth_to_meters(self, image, width=None, height=None):
        if isinstance(image, carla.Image):
            raw = image.raw_data
            w, h = image.width, image.height
        elif isinstance(image, (bytes, bytearray)) and width and height:
            raw, w, h = image, width, height
        else:
            raise ValueError("Unsupported depth format")

        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        rgb = arr[:, :, :3].astype(np.uint32)
        depth24 = rgb[:, :, 0] + (rgb[:, :, 1] << 8) + (rgb[:, :, 2] << 16)
        depth = depth24.astype(np.float32) / float(256**3 - 1)
        return depth * 1000.0

    def reproject_pixel_to_world(self, u, v, depth_image, camera_actor, depth_width=None, depth_height=None):
        if isinstance(depth_image, carla.Image):
            depth_map = self.carla_depth_to_meters(depth_image)
            width, height = depth_image.width, depth_image.height
        elif isinstance(depth_image, (bytes, bytearray)):
            depth_map = self.carla_depth_to_meters(depth_image, depth_width, depth_height)
            width, height = depth_width, depth_height
        else:
            raise ValueError("Unsupported depth format")

        z = float(depth_map[int(v), int(u)])
        if z <= 0.0:
            raise ValueError("Invalid depth")

        fov = float(camera_actor.attributes.get('fov', 90.0))
        cx, cy = width / 2.0, height / 2.0
        fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fx
        z_cam = z

        local = carla.Location(x=z_cam, y=-x_cam, z=-y_cam)
        world = camera_actor.get_transform().transform(local)
        return world.x, world.y, world.z

    # =============================================================
    # ROAD-RULES HELPERS
    # =============================================================
    def get_valid_waypoint_ahead(self, transform, distance=20.0):
        """
        Return a legal CARLA waypoint ahead of the given transform.
        """
        map_ = self.world.get_map()
        wp = map_.get_waypoint(transform.location, project_to_road=True)

        travelled = 0.0
        while travelled < distance:
            next_wps = wp.next(1.0)
            if not next_wps:
                break
            wp = next_wps[0]
            travelled += 1.0

        wp = map_.get_waypoint(wp.transform.location, project_to_road=True)
        return wp

    def drive_to_waypoint_road_rules(self, target_location):
        """
        Drive using BasicAgent while following road rules.
        """
        if not BasicAgent:
            print("BasicAgent not available.")
            return False

        agent = BasicAgent(self.vehicle, target_speed=25.0)
        agent.set_destination((target_location.x, target_location.y, target_location.z))

        print(f"[ROAD RULES] Driving to {target_location}")

        while True:
            self.world.tick()
            control = agent.run_step()
            self.vehicle.apply_control(control)

            if self.vehicle.get_location().distance(target_location) < 1.5:
                print("[ROAD RULES] Target reached.")
                break

        return True

    # =============================================================
    # ACTION HANDLER
    # =============================================================
    def act(self, action):
        try:
            # -----------------------------------------
            # STOP command
            # -----------------------------------------
            if isinstance(action, str):
                if action == "STOP":
                    print("STOP command received.")
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    return
                else:
                    print(f"Unknown action string: {action}")
                    return

            # -----------------------------------------
            # Pixel click → world target → road projection
            # -----------------------------------------
            elif isinstance(action, list) and len(action) >= 2:

                u, v = int(action[0]), int(action[1])

                if not self.last_depth_raw:
                    print("No depth image available.")
                    return

                wx, wy, wz = self.reproject_pixel_to_world(
                    u, v,
                    self.last_depth_raw,
                    self.rgb_camera,
                    self.last_depth_width,
                    self.last_depth_height
                )
                print(f"[ACT] Pixel ({u},{v}) → World ({wx:.2f}, {wy:.2f}, {wz:.2f})")

                world_loc = carla.Location(wx, wy, wz)
                map_ = self.world.get_map()

                # Snap clicked point onto road
                wp_nearest = map_.get_waypoint(world_loc, project_to_road=True)

                # Compute ahead waypoint
                wp_ahead = self.get_valid_waypoint_ahead(wp_nearest.transform, distance=20.0)

                road_target = wp_ahead.transform.location
                print(f"[ACT] Road-projected target: {road_target}")

                return self.drive_to_waypoint_road_rules(road_target)

        except Exception as e:
            print("Error executing action:", e)

    # =============================================================
    # VLM TICK LOOP
    # =============================================================
    def tick(self):
        if not self.vlm or not self.last_rgb_image:
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_name = tmp.name
        tmp.close()

        try:
            self.last_rgb_image.save_to_disk(tmp_name)
            action = self.vlm.interpret_image(tmp_name)
            self.act(action)
        except Exception as e:
            print("VLM tick failed:", e)
        finally:
            try:
                os.remove(tmp_name)
            except:
                pass

    # =============================================================
    # CLEANUP
    # =============================================================
    def destroy(self):
        for s in getattr(self, 'sensors', []):
            try:
                s.stop()
                s.destroy()
            except:
                pass
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass
        print("Vehicle and sensors destroyed.")
