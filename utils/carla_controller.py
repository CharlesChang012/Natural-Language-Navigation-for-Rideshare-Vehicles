import carla
import time
import os
import math
import numpy as np
import tempfile
import cv2
from PIL import Image, ImageDraw, ImageFont
import ollama
import re
import json
from datetime import datetime

class VisionNavigationSystem:
    def _clean_prompt(self, prompt: str) -> str:
        p = prompt.lower()
        for t in ["navigate to the", "go to the", "head to the", "navigate to"]:
            p = p.replace(t, "")
        return p.strip()

class RoadAwareNavigator(VisionNavigationSystem):
    def _draw_grid(self, pil_img, rows=5, cols=5):
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        W, H = img.size
        
        step_x = W / cols
        step_y = H / rows
        
        grid_map = {}

        font_size = int(min(W, H) / 25)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        cell_id = 1
        for r in range(rows):
            for c in range(cols):
                x0 = c * step_x
                y0 = r * step_y
                x1 = x0 + step_x
                y1 = y0 + step_y
                
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                
                center_x = int((x0 + x1) / 2)
                center_y = int((y0 + y1) / 2)
                
                text = str(cell_id)
                bbox = draw.textbbox((0,0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                draw.rectangle(
                    [center_x - text_w/2 - 5, center_y - text_h/2 - 5, 
                     center_x + text_w/2 + 5, center_y + text_h/2 + 5], 
                    fill="black"
                )
                draw.text(
                    (center_x - text_w/2, center_y - text_h/2), 
                    text, fill="white", font=font
                )
                
                grid_map[cell_id] = [center_x, center_y]
                cell_id += 1
                
        return img, grid_map

    def navigate(self, image: carla.Image, prompt: str, debug: bool = False):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1] # BGRA to RGB
        pil_img = Image.fromarray(rgb)
        

        grid_img, grid_map = self._draw_grid(pil_img, rows=5, cols=5)
        
        cleaned = self._clean_prompt(prompt)
        
        #prompt
        system_prompt = f"""
        You are a navigation assistant. You are looking at a driver's view with a RED GRID overlay.
        Each grid cell has a WHITE NUMBER.

        Your Task:
        Identify the grid cell number that contains the: "{cleaned}"

        Rules:
        1. Select the cell containing the center of the object.
        2. If the object spans multiple cells, pick the one closest to the road surface.
        3. If the object is not visible, reply with ID: 0.

        Output Format:
        Return ONLY the JSON: {{"id": <number>}}
        """

        # Save the GRID image for the VLM to see
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            grid_img.save(path)

        if debug:
            print(f"[VLM] Sending grid image: {path}")

        try:
            res = ollama.chat(
                model="qwen3-vl:235b-cloud", 
                messages=[{"role": "user", "content": system_prompt, "images": [path]}],
                format="json"
            )
        except Exception as e:
            print(f"[ERROR] Ollama failed: {e}")
            return {"action": "STOP"}
        finally:
            try: os.remove(path)
            except: pass

        text = res["message"]["content"]
        if debug:
            print("[VLM RAW]", text)

        try:
            if "{" in text:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    cell_id = int(data.get("id", 0))
                else:
                    cell_id = 0
            else:
                nums = re.findall(r"\d+", text)
                cell_id = int(nums[0]) if nums else 0

            if cell_id in grid_map:
                u, v = grid_map[cell_id]
                return {"action": [u, v]}
            else:
                return {"action": "STOP"}

        except Exception as e:
            print(f"[ERROR] Parsing VLM response: {e}")
            return {"action": "STOP"}


# CARLA interface
class CarlaController:
    def __init__(self, host="localhost", port=2000,
                 bev_height=50.0, bev_resolution=600):

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        self.spectator = self.world.get_spectator()

        self.vehicle = None
        self.rgb = None
        self.depth = None
        self.bev_cam = None

        self.last_rgb = None
        self.last_depth = None
        self.depth_W = None
        self.depth_H = None

        self.bev_img = None
        self.bev_height = bev_height
        self.bev_res = bev_resolution
        
        self.start_loc = None
        self.target_loc = None
        self.reprojected_loc = None

        self.navigator = RoadAwareNavigator()

        from agents.navigation.basic_agent import BasicAgent
        self.BasicAgent = BasicAgent


    # def spawn_vehicle(self):
    #     bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    #     sps = self.world.get_map().get_spawn_points()

    #     for i, sp in enumerate(sps):
    #         v = self.world.try_spawn_actor(bp, sp)
    #         if v:
    #             self.vehicle = v
    #             loc = sp.location
    #             self.start_loc = (float(loc.x), float(loc.y), float(loc.z))
    #             print(f"[SPAWN] Vehicle spawned at index {i}: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
                
    #             # Move spectator immediately to car
    #             self.update_spectator()
    #             return

    #     raise RuntimeError("No spawn point could spawn a vehicle.")

    def spawn_vehicle(self):
            bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
            
            specific_loc = carla.Location(x=100.73, y=112.2, z=1.73 + 0.5) 
            specific_rot = carla.Rotation(yaw=-70) # Change yaw if you want it facing a different way
            target_transform = carla.Transform(specific_loc, specific_rot)

            print(f"[SPAWN] Attempting to spawn at custom coordinates: {specific_loc}")
            self.vehicle = self.world.try_spawn_actor(bp, target_transform)

            if self.vehicle:
                self.start_loc = (float(specific_loc.x), float(specific_loc.y), float(specific_loc.z))
                
                self.vehicle.set_simulate_physics(True)
                
                self.update_spectator()
                print("[SPAWN] Success.")
                return
            
            print("[ERROR] Could not spawn at custom coordinates. The spot might be occupied.")
            raise RuntimeError("Spawn failed at custom coordinates.")

    def update_spectator(self):
        """Moves the main simulator window to follow the car"""
        if not self.vehicle: 
            return

        t = self.vehicle.get_transform()
        fwd = t.get_forward_vector()
        cam_loc = t.location - (fwd * 10) + carla.Location(z=5)
        
        cam_rot = t.rotation
        cam_rot.pitch = -15
        
        self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def spawn_rgbd(self):
        lib = self.world.get_blueprint_library()

        rgb_bp = lib.find("sensor.camera.rgb")
        depth_bp = lib.find("sensor.camera.depth")

        for b in (rgb_bp, depth_bp):
            b.set_attribute("image_size_x", "800")
            b.set_attribute("image_size_y", "600")
            b.set_attribute("fov", "90")

        tf = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.rgb = self.world.spawn_actor(rgb_bp, tf, attach_to=self.vehicle)
        self.depth = self.world.spawn_actor(depth_bp, tf, attach_to=self.vehicle)

        self.rgb.listen(lambda img: setattr(self, "last_rgb", img))
        self.depth.listen(lambda img: self._store_depth(img))

    def _store_depth(self, img):
        self.last_depth = np.frombuffer(img.raw_data, dtype=np.uint8)
        self.depth_W = img.width
        self.depth_H = img.height

    def decode_depth(self):
        """
        Correctly handles BGRA layout for CARLA depth.
        Formula: (R + G * 256 + B * 256^2) / (256^3 - 1) * 1000
        """
        raw = self.last_depth.reshape((self.depth_H, self.depth_W, 4))
        
        B = raw[:, :, 0].astype(np.float32)
        G = raw[:, :, 1].astype(np.float32)
        R = raw[:, :, 2].astype(np.float32)

        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0**3 - 1)
        return normalized * 1000.0

    def pixel_to_world(self, u, v):
        depth_m = self.decode_depth()

        u = int(max(0, min(u, self.depth_W - 1)))
        v = int(max(0, min(v, self.depth_H - 1)))

        Z = float(depth_m[v, u])
        
        if Z > 900: Z = 100.0

        cam = self.rgb
        tf = cam.get_transform()

        fov = float(cam.attributes["fov"])
        W = float(self.depth_W)
        H = float(self.depth_H)

        cx = W / 2
        cy = H / 2
        fx = W / (2 * math.tan(math.radians(fov / 2)))

        X_cam = Z
        Y_cam = (u - cx) * Z / fx
        Z_cam = (cy - v) * Z / fx 

        cam_point = carla.Location(x=float(X_cam), y=float(Y_cam), z=float(Z_cam))
        return tf.transform(cam_point)

    def spawn_bev_cam(self, fov=90):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.bev_res))
        bp.set_attribute("image_size_y", str(self.bev_res))
        bp.set_attribute("fov", str(fov))

        vloc = self.vehicle.get_location()
        tf = carla.Transform(
            carla.Location(vloc.x, vloc.y, self.bev_height),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )

        if self.bev_cam:
            try: self.bev_cam.destroy()
            except: pass

        self.bev_cam = self.world.spawn_actor(bp, tf)
        self.bev_cam.listen(lambda img: setattr(self, "bev_img", img))
        print("[BEV] Spawned.")

    def update_bev_cam(self):
        if not self.vehicle or not self.bev_cam: return
        vloc = self.vehicle.get_location()
        tf = carla.Transform(
            carla.Location(vloc.x, vloc.y, self.bev_height),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )
        self.bev_cam.set_transform(tf)

    def world_to_bev_pixel(self, world_loc, H, W, fov):
        """
        Maps World X (Forward) to Image V (Up/Down)
        Maps World Y (Right) to Image U (Left/Right)
        """
        if not self.bev_cam: return (W//2, H//2)
        
        cam_loc = self.bev_cam.get_transform().location
        
        if isinstance(world_loc, tuple):
            wx, wy = world_loc[0], world_loc[1]
        else:
            wx, wy = world_loc.x, world_loc.y
        
        d_world_x = float(wx - cam_loc.x)
        d_world_y = float(wy - cam_loc.y)

        ground_width = 2 * self.bev_height * math.tan(math.radians(fov / 2))
        ppm = W / ground_width

        px = int(W/2 + d_world_y * ppm)       
        py = int(H/2 - d_world_x * ppm)       
        
        return px, py

    def render_bev(self):
        if self.bev_img is None: return

        arr = np.frombuffer(self.bev_img.raw_data, dtype=np.uint8)
        img = arr.reshape((self.bev_img.height, self.bev_img.width, 4))[:, :, :3].copy()

        H, W = img.shape[:2]
        fov = float(self.bev_cam.attributes["fov"])

        def draw_point(loc, color, label):
            if loc is not None:
                px = self.world_to_bev_pixel(loc, H, W, fov)
                cv2.circle(img, px, 6, color, -1)
                cv2.putText(img, label, (px[0]+10, px[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        draw_point(self.start_loc, (0, 255, 0), "Start")
        draw_point(self.vehicle.get_location(), (255, 0, 0), "Car") 
        draw_point(self.target_loc, (0, 0, 255), "Target") 
        draw_point(self.reprojected_loc, (0, 255, 255), "Raw")

        cv2.putText(img, "BEV: Green=Start | Red=Target | Blue=Car", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("BEV", img)
        cv2.waitKey(1)

    def drive_to(self, loc):
        agent = self.BasicAgent(self.vehicle, target_speed=20)
        
        if isinstance(loc, tuple):
            dest = carla.Location(x=loc[0], y=loc[1], z=loc[2])
        else:
            dest = loc
            
        agent.set_destination(dest)
        print(f"[DRIVE] Destination: ({dest.x:.2f}, {dest.y:.2f}, {dest.z:.2f})")

        while True:
            self.world.tick()
            self.update_bev_cam()
            self.update_spectator() # camera follows car
            self.render_bev()

            ctrl = agent.run_step()
            self.vehicle.apply_control(ctrl)

            if self.vehicle.get_location().distance(dest) < 2.0:
                print("[DRIVE] Arrived.")
                self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
                return

    #
    def run(self, prompt, debug=True):
        if self.vehicle is None: raise RuntimeError("Call spawn_vehicle first.")
        if self.bev_cam is None: self.spawn_bev_cam()

        for _ in range(30): 
            self.world.tick()
            self.update_spectator()

        print("[NAV] Analyzing scene...")
        nav = self.navigator.navigate(self.last_rgb, prompt, debug)
        
        if nav["action"] == "STOP":
            print("[VLM] Target not found.")
            return

        u, v = nav["action"]

        arr = np.frombuffer(self.last_rgb.raw_data, dtype=np.uint8).reshape(
            (self.last_rgb.height, self.last_rgb.width, 4)
        )
        rgb = arr[:, :, :3][:, :, ::-1].copy()
        cv2.circle(rgb, (u, v), 8, (0, 0, 255), -1)
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        os.makedirs("output", exist_ok=True)
        fname = f"output/target_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(fname, bgr)
        print(f"[SAVE] Target identified. Image: {fname}")

        # Reproject
        world_pt = self.pixel_to_world(u, v)
        self.reprojected_loc = (float(world_pt.x), float(world_pt.y), float(world_pt.z))
        
        # Snap projected poitn to road
        wp = self.world.get_map().get_waypoint(world_pt, project_to_road=True, lane_type=carla.LaneType.Driving)
        self.target_loc = (float(wp.transform.location.x), float(wp.transform.location.y), float(wp.transform.location.z))
        
        print(f"[NAV] Snapped Target: {self.target_loc}")

        self.drive_to(self.target_loc)

    def destroy(self):
        try: cv2.destroyAllWindows()
        except: pass

        for a in [self.rgb, self.depth, self.bev_cam, self.vehicle]:
            try:
                if a: a.destroy()
            except: pass


def main():
    controller = CarlaController()
    try:
        controller.spawn_vehicle()
        controller.spawn_rgbd()
        
        # Example
        prompt = "Navigate to the bin"
        controller.run(prompt, debug=True)

    except KeyboardInterrupt:
        print("Cancelled by user.")
    finally:
        controller.destroy()
        print("Cleaned up actors.")

if __name__ == "__main__":
    main()