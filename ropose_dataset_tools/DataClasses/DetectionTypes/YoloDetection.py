from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Vector2D import Vector2D

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

    def Match(self, target: 'YoloDetection', minimumIoU: float = 0.5):
        iou = self.boundingBox.CalculateIoU(target.boundingBox)
        return iou >= minimumIoU and self.predictedClass == target.predictedClass
