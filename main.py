import time
from utils.carla_controller import CarlaController

def main():
    controller = CarlaController()

    try:
        print("\nSpawning vehicle...")
        controller.spawn_vehicle()
        time.sleep(1.0)

        print("Spawning RGB + Depth cameras...")
        controller.spawn_rgbd()
        time.sleep(1.0)

        print("Spawning BEV camera...")
        controller.spawn_bev_cam()
        time.sleep(1.0)

        print("\nSimulation ready. Enter natural language navigation commands.")
        print("Example: 'navigate to the nearest streetlight'")
        print("Type 'quit' or 'exit' to stop.")

        # loop
        while True:
            prompt = input("\nCommand: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            if len(prompt) == 0:
                continue

            controller.run(prompt)

    finally:
        print("\nCleaning up actors")
        controller.destroy()
        print("Done.")

if __name__ == "__main__":
    main()
