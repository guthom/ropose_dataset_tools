from typing import List
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox

class KeypointDetection(object):

    def __init__(self, keypoints: List[List[float]], confs: List[float], detectedClass: int=None):
        self.keypoints = keypoints
        self.confs = confs
        self.detectionsClass = detectedClass

