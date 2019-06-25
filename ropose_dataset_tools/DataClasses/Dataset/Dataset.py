from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from guthoms_helpers.base_types.Pose3D import Pose3D
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
import numpy as np
from typing import List


class Dataset:
    def __init__(self):
        self.worldTransforms = []
        self.rgbFrame: Image = None
        self.depthFrame: Image = None
        self.yoloData: YoloData = None
        self.backgroundMask: np.array = None
        self.backgroundHeatmap: np.array = None
        self.annotations = None
        self.valid: bool = True
        self.sensorName: str = None
        self.metadata: Metadata = None

    def Clear(self):
        self.backgroundMask = None
        self.backgroundHeatmap = None



