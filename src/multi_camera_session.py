import cv2
import time
import numpy as np
from camera import Camera
from utils import get_logger

class MultiCameraSession():
  def __init__(self, cameras):
    self.LOG = get_logger(__name__)

    self.cameras = [Camera(camera) for camera in cameras]

    self.fuducial = cv2.imread('cal_images/marker1.png')

    self.last_time = time.time()

  def get_frames(self) -> list[np.ndarray]:
    frames = []
    for camera in self.cameras:
      frames.append(camera.get_frame())
    return frames

  def get_fps(self) -> float:
    current_time = time.time()
    fps = 1 / (current_time - self.last_time)
    self.last_time = current_time
    return fps

  def add_fps_to_frame(self, frame: np.ndarray) -> np.ndarray:
    fps = self.get_fps()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

  def run(self) -> None:
    while True:
      frames = self.get_frames()

      for i, frame in enumerate(frames):
        frame = self.add_fps_to_frame(frame)
        cv2.imshow(f"Camera {i}", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows

  def calibrate(self) -> None:
    self.LOG.info("Calibrating cameras")

    while True:
      fud_images = []
      for camera in self.cameras:
        fud_images.append(
          camera.find_fuducial(self.fuducial)
        )

      for i, frame in enumerate(fud_images):
        frame = self.add_fps_to_frame(frame)
        cv2.imshow(f"Camera {i}", frame)
      if cv2.waitKey(1) & 0xFF == ord('q') or all([camera.fuducial_status for camera in self.cameras]):
        break
