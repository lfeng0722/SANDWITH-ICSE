import carla
import time
import yaml


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    print("🚀 Tick Manager 正在推进仿真...")

    try:
        while True:
            world.tick()  # 统一控制 tick
            time.sleep(0.05)  # 控制 tick 频率，避免 CPU 过载
    finally:
        print("Tick Manager 终止")


if __name__ == "__main__":
    main()