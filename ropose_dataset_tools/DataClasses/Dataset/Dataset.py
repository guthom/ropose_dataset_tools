from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from ropose_dataset_tools.DataClasses.BaseTypes.Pose import Pose
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
import numpy as np
from typing import List


class Dataset:

    worldTransforms: List[Pose] = None
    rgbFrame: Image = None
    depthFrame: Image = None
    yoloData: YoloData = None
    backgroundMask: np.array = None
    backgroundHeatmap: np.array = None
    annotations = None
    valid: bool = True
    sensorName: str = None
    metadata: Metadata = None

    def __init__(self):
        self.worldTransforms = []
        self.rgbFrame: Image = None
        self.depthFrame: Image = None
        self.backgroundMask: np.array = None
        self.backgroundHeatmap: np.array = None
        self.annotations = None
        self.valid: bool = True
        self.sensorName: str = None
        self.metadata: Metadata = None

    def Clear(self):
        self.backgroundMask = None
        self.backgroundHeatmap = None



