from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
from typing import Optional
import numpy as np
import ropose_dataset_tools.config as config
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

    def GetBackgroundMaskURDF(self) -> Optional[np.array]:

        #TODO Allow backgorund generation with CAD Data!

        if not config.useURDFForBackground:
            return None

        if not DirectoryHelper.DirExists(config.urDescriptionBackgroundPath):
            raise Exception("CAD-Data etc for the robot does not exist!")

        raise Exception("Not Implemented!")



