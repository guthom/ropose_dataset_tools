from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.common_stuff.Timer import Timer
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
from typing import List, Tuple, Dict
import ropose_dataset_tools.config as config

class YoloData(object):
    def __init__(self):
        self.boundingBoxes: List[BoundingBox] = []
        self.resizedBoundingBoxes: List[BoundingBox] = []
        self.classIDs: List[int] = []
        self.keypoints: List[List[float]] = []

    def __len__(self):
        return len(self.classIDs)

    @classmethod
    def FromRopose(cls, dataset: 'Dataset'):
        ret = cls()
        ret.classIDs.append(config.yolo_RoposeClassNum)
        ret.boundingBoxes.append(dataset.rgbFrame.boundingBox)
        ret.keypoints.append(dataset.rgbFrame.resizedReprojectedPoints)
        return ret

    def Extend(self, other: 'YoloData'):
        self.boundingBoxes.extend(other.boundingBoxes)
        self.resizedBoundingBoxes.extend(other.resizedBoundingBoxes)
        self.classIDs.extend(other.classIDs)
        self.keypoints.extend(other.keypoints)

    @classmethod
    def FromCoco(cls, annotations):
        ret = cls()
        ret.ExtractClassIDs(annotations)
        ret.CreateBoundingBoxesFromCoco(annotations)
        ret.ExtractKeypointsFromAnotations(annotations)
        return ret

    def ExtractClassIDs(self, annotations: Dict):
        for annotation in annotations:
            if "category_id" in annotation:
                self.classIDs.append(config.yolo_cocoClassMap[annotation["category_id"]])

    def ExtractKeypointsFromAnotations(self, annotations: Dict):
        for annotation in annotations:
            if "keypoints" in annotation:
                rawKeypoints = annotation["keypoints"]
                subKeypoints = []
                for i in range(0, rawKeypoints.__len__(), 3):
                    keypoint = Pose2D.fromData(rawKeypoints[i], rawKeypoints[i + 1], angle=0.0, visible=True)

                    if keypoint[0] == 0 and keypoint[1] == 0:
                        keypoint.visible = False

                    subKeypoints.append(keypoint)

                self.keypoints.append(subKeypoints)

    def CreateBoundingBoxesFromCoco(self, annotations: Dict):
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            self.boundingBoxes.append(BoundingBox.FromList([x, y, x+w, y+h]))

    def ToYoloDetections(self) -> List[YoloDetection]:
        yoloDetections = []
        for i in range(0, self.classIDs.__len__()):
            yoloDetections.append(YoloDetection(boundingBox=self.boundingBoxes[i], predictedClass=self.classIDs[i],
                                                confidence=1.0))

        return  yoloDetections



