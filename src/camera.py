import cv2
from utils import get_logger

import numpy as np

IMG_SIZE = 512
GOOD_ORB_MATCH_THRESHOLD = 50
STATIC_THRESHOLD = 10

class Camera:
  def __init__(self, camera_id):
    self.LOG = get_logger(__name__)

    self.LOG.info(f"Initializing Camera with ID: {camera_id}")
    self.camera = cv2.VideoCapture(camera_id)

    self.fuducial_projection = None
    self.fuducial_static = 0
    self.fuducial_status = False
    self.location = None

  def get_frame(self):
    success, frame = self.camera.read()
    #frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return frame

  def find_fuducial(self, fuducial):
    frame = self.get_frame()
    final = frame

    orb = cv2.ORB_create(nfeatures=50, edgeThreshold=12)

    kp1, des1 = orb.detectAndCompute(fuducial,None)
    kp2, des2 = orb.detectAndCompute(frame,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des1 is not None and des2 is not None:
      good_matches = []
      matches = bf.match(des1,des2)

      matches = sorted(matches, key = lambda x:x.distance)

      for m in matches:
        if m.distance < GOOD_ORB_MATCH_THRESHOLD:
          good_matches.append(m)

      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

      final = cv2.drawMatches(fuducial,kp1,frame,kp2,good_matches, None)

      if len(src_pts) >=4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        hf,wf = fuducial.shape[:2]
        pts = np.float32([ [0,0],[0,hf-1],[wf-1,hf-1],[wf-1,0] ]).reshape(-1,1,2)

        try:
          dst = cv2.perspectiveTransform(pts,M)
          dst += (wf, 0)
        except:
          dst = pts

        final = cv2.polylines(final, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
        self.update_fuducial(dst)
    return final

  def update_fuducial(self, fud_pts):
    """a function to check if the fuducial has stayed still for enough frames and update 
    the fuducial status"""
    if self.fuducial_projection is None:
      self.fuducial_projection = fud_pts
      self.fuducial_static = 0
    else:
      if np.allclose(self.fuducial_projection, fud_pts, atol=5):
        self.fuducial_static += 1
      else:
        self.fuducial_static = 0
        self.fuducial_projection = fud_pts

    if self.fuducial_static >= STATIC_THRESHOLD:
      self.fuducial_status = True

  def check_camera(self):
    return self.camera.isOpened()

  def calibrate_camera(self):
    pass

  def __del__(self):
    self.camera.release()