from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox

class YoloDetection(object):

    def __init__(self, boundingBox: BoundingBox=None, predictedClass: int=None, confidence: float=None):
        self.boundingBox = boundingBox
        self.predictedClass = predictedClass
        self.confidence = confidence

    @classmethod
    def FromPredictionTensor(cls, prediction):
        bb = BoundingBox(int(prediction[0].item()), int(prediction[1].item()),
                         int(prediction[2].item()), int(prediction[3].item()))

        return cls(bb, int(prediction[6].item()), prediction[4].item())

    def Match(self, target: 'YoloDetection', minimumIoU: float = 0.5):
        iou = self.boundingBox.CalculateIoU(target.boundingBox)
        return iou >= minimumIoU and self.predictedClass == target.predictedClass
