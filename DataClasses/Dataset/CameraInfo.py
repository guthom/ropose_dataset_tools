from typing import Dict, List
import numpy as np
import simplejson as json

class CameraInfo(object):

    #distortion modell
    D: List[float] = None

    #intrinsic camera matrix
    K: type(np.matrix) = None

    #rectification matrix
    R: List[float] = None

    #projection matrix
    P: List[float] = None

    width: float = None
    height: float = None
    binning_x: float = None
    binning_y: float = None

    filePath: str = None

    cameraName: str = None

    @classmethod
    def fromJson(cls, jsonData: Dict):

        info = cls()

        info.D = jsonData["D"]

        temp = jsonData["K"]
        info.K = np.matrix([temp[0:3], temp[3:6], temp[6:9]])

        temp = jsonData["R"]
        info.R = np.matrix([temp[0:3], temp[3:6], temp[6:9]])

        temp = jsonData["P"]
        info.P = np.matrix([temp[0:4], temp[4:8], temp[8:12]])

        info.width = jsonData["width"]
        info.height = jsonData["height"]
        info.binning_x = jsonData["binning_x"]
        info.binning_y = jsonData["binning_y"]
        info.cameraName = jsonData["CameraName"]
        info.filePath = jsonData["filePath"]
        return info
