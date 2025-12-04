import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    spectator = world.get_spectator()

    try:
        while True:
            # Get location and rotation
            t = spectator.get_transform()
            l = t.location
            r = t.rotation
            
            # Print formatted output (overwriting the line so it doesn't scroll)
            print(f"\rLoc: (x={l.x:.2f}, y={l.y:.2f}, z={l.z:.2f}) | Rot: (pitch={r.pitch:.2f}, yaw={r.yaw:.2f})", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")

if __name__ == '__main__':
    main()