import argparse
import logging
from multi_camera_session import MultiCameraSession


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--cameras", type=int, nargs='+', help="List of cameras to use")
  parser.add_argument("-l", "--log-level", type=str, default="DEBUG", help="Logging level")
  return parser.parse_args()


def main(cameras: list[int]) -> None:
  logging.basicConfig(level=args.log_level)

  loop_runner = MultiCameraSession(args.cameras)

  loop_runner.calibrate()






if __name__ == "__main__":
  args = parse_args()
  main(
    args.cameras,
  )