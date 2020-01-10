from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Vector2D import Vector2D
from typing import Optional
import numpy as np

class YoloDetection(object):

    def __init__(self, boundingBox: BoundingBox=None, predictedClass: int=None, confidence: float=None):
        self.boundingBox = boundingBox
        self.predictedClass = predictedClass
        self.confidence = confidence

    @classmethod
    def FromPredictionTensor(cls, prediction):
        vec1 = Vector2D(int(prediction[0].item()), int(prediction[1].item()))
        vec2 = Vector2D(int(prediction[2].item()), int(prediction[3].item()))
        bb = BoundingBox.FromTwoPoints(vec1, vec2)

        return cls(bb, int(prediction[6].item()), prediction[4].item())

    def ToPredictionTensor(self, confidence: float = 0.0, image: Optional[np.array] = None):
        #yolo bounding box is normed with imge dimensions (range 0 - 1.0)
        if image is not None:
            bb = self.boundingBox.NormWithImage(image)
        else:
            bb = self.boundingBox
        ret = [float(confidence), float(self.predictedClass), bb.midX, bb.midY, bb.width, bb.height]
        return np.array(ret)

    def Match(self, target: 'YoloDetection', minimumIoU: float = 0.5):
        iou = self.boundingBox.CalculateIoU(target.boundingBox)
        return iou >= minimumIoU and self.predictedClass == target.predictedClass
