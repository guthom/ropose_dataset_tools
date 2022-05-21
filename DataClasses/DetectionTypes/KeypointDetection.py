from typing import List

class KeypointDetection(object):

    def __init__(self, keypoints: List[List[float]], confs: List[float], detectedClass: int=None):
        self.keypoints = keypoints
        self.confs = confs
        self.detectionsClass = detectedClass

