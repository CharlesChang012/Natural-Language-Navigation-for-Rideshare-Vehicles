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

        # Vision-language model interface
        self.vlm = DummyVLM()

    # ========== VEHICLE & SENSOR SETUP ========== #
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

    # ========== DEPTH REPROJECTION ========== #
    def carla_depth_to_meters(self, image, width=None, height=None):
        if isinstance(image, carla.Image):
            raw = image.raw_data
            w, h = image.width, image.height
        elif isinstance(image, (bytes, bytearray)) and width and height:
            raw, w, h = image, width, height
        elif isinstance(image, np.ndarray):
            return image.astype(np.float32)
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
            depth_map = np.array(depth_image, dtype=np.float32)
            height, width = depth_map.shape

        z = float(depth_map[int(v), int(u)])
        if z <= 0.0:
            raise ValueError("Invalid depth")

        fov = float(camera_actor.attributes.get('fov', 90.0))
        cx, cy = width / 2.0, height / 2.0
        fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        fy = fx

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        local = carla.Location(x=z_cam, y=-x_cam, z=-y_cam)
        world = camera_actor.get_transform().transform(local)
        return world.x, world.y, world.z

    # ========== NAVIGATION ========== #
    def drive_to_location(self, target_xyz):
        if not BasicAgent:
            print("BasicAgent not available.")
            return False
        if not self.vehicle:
            print("Vehicle not spawned.")
            return False

        target_loc = carla.Location(*target_xyz)
        agent = BasicAgent(self.vehicle, target_speed=20.0)  # km/h
        agent.set_destination((target_loc.x, target_loc.y, target_loc.z))

        while True:
            world_snapshot = self.world.wait_for_tick()
            control = agent.run_step()
            self.vehicle.apply_control(control)

            # Check distance to target
            vehicle_loc = self.vehicle.get_location()
            dist = vehicle_loc.distance(target_loc)
            if dist < 2.0:
                print("Reached target!")
                break
        return True
    
    # ========== ACTION HANDLING ========== #
    def act(self, action, pixel_dest=None):
        try:
            if action in ('goto_pixel', 'goto_image_point'):
                if not pixel_dest or len(pixel_dest) < 2:
                    print('goto_pixel requires [u, v]')
                    return

                u, v = int(pixel_dest[0]), int(pixel_dest[1])
                if not self.last_depth_raw:
                    print('No depth image yet.')
                    return

                # Reproject target pixel to world coordinates
                world_x, world_y, world_z = self.reproject_pixel_to_world(
                    u, v, self.last_depth_raw, self.rgb_camera,
                    self.last_depth_width, self.last_depth_height)

                print(f"Pixel ({u},{v}) → World ({world_x:.2f},{world_y:.2f},{world_z:.2f})")
                target = (world_x, world_y, world_z)

                # Drive to the computed world location
                success = self.drive_to_location(target)
                return

        except Exception as e:
            print("Error executing action:", e)

    # ========== AUTOMATIC TICK (VLM LOOP) ========== #
    def tick(self):
        """Call this each simulation step to let VLM decide next action."""
        if not self.vlm or not self.last_rgb_image:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_name = tmp.name
        tmp.close()
        try:
            self.last_rgb_image.save_to_disk(tmp_name)
            out = self.vlm.interpret_image(tmp_name)
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    out = {"action": out}
            if out and "action" in out:
                self.act(out["action"], out.get("pixel_dest"))
        except Exception as e:
            print("VLM tick failed:", e)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    def destroy(self):
        for s in getattr(self, 'sensors', []):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
        print("Vehicle and sensors destroyed.")