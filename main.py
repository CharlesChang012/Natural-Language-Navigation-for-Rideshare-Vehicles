import time
import random
from utils.carla_controller import CarlaController


def main():
    # Initialize CARLA controller and VLM
    controller = CarlaController(host="localhost", port=2000)

    try:
        # Spawn a vehicle and attach RGB-D sensors
        controller.spawn_vehicle()
        controller.spawn_rgbd(prefix="test_output", image_size_x=800, image_size_y=600, fov=90, sensor_tick=0.1)

        world = controller.world
        print("Starting simulation...")

        # Let the simulation run for ~60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            world.tick()          # advance one simulation step (if running in synchronous mode)
            goal_reached = controller.tick()     # get image -> VLM -> act()
            if goal_reached:
                print("Goal reached, stopping simulation.")
                break
            time.sleep(0.1)       # small delay to simulate control rate (10 Hz)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        controller.destroy()
        print("Simulation ended and cleaned up.")

if __name__ == "__main__":
    main()
