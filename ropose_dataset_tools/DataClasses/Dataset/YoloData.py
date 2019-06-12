from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from typing import List, Tuple, Dict
import ropose_dataset_tools.config as config

class YoloData(object):
    boundingBoxes: List[BoundingBox] = None
    resizedBoundingBoxes:  List[BoundingBox] = None
    classIDs: List[int] = None
    keypoints: List[List[float]] = None


    def __init__(self):
        self.boundingBoxes = []
        self.classIDs = []
        self.resizedBoundingBoxes = []
        self.keypoints = []


    @classmethod
    def FromRopose(cls, dataset: 'Dataset'):
        ret = cls()
        ret.classIDs.append(config.yolo_RoposeClassNum)
        ret.boundingBoxes.append(dataset.rgbFrame.boundingBox)
        ret.keypoints.append(dataset.rgbFrame.resizedReprojectedPoints)
        return ret

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
                    keypoint = [rawKeypoints[i], rawKeypoints[i + 1]]

                    if keypoint[0] == 0 and keypoint[1] == 0:
                        keypoint = [-1, -1]

                    subKeypoints.append(keypoint)

                self.keypoints.append(subKeypoints)

    def CreateBoundingBoxesFromCoco(self, annotations: Dict):
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            self.boundingBoxes.append(BoundingBox.FromList([x, y, x+w, y+h]))




