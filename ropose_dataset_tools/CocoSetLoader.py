import os
from typing import List, Optional
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Vector2D import Vector2D
from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
from pycocotools.coco import COCO
from copy import deepcopy
import ropose_dataset_tools.config as config

def GetDataSets(path: str):
    dataDirs = []
    for x in os.listdir(path):
        if x != "examples":
            dirName = path + x + "/"
            if os.path.isdir(dirName):
                dataDirs.append(dirName)

    return dataDirs

def LoadCocoSets(cocoPath = config.cocoPath, cocoDataset=config.cocoDatasetTrain, mixWithZeroHumans=False,
                 mixWithZeroHuamnsFactor=0.1, amount: Optional[int]=None, setSize: Optional[int]=None) \
    -> List[type(Dataset)]:
    ret = []

    for dataset in cocoDataset:
        ret.extend(_LoadCocoSets(cocoPath, dataset, mixWithZeroHumans, mixWithZeroHuamnsFactor, amount, setSize))

    return ret

def _LoadCocoSets(cocoPath = config.cocoPath, cocoDataset="train2017", mixWithZeroHumans=False,
                 mixWithZeroHuamnsFactor=0.1, amount: Optional[int]=None, setSize: Optional[int]=None) \
        -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    personIds = coco.getCatIds(catNms=['person'])
    tempIds = deepcopy(config.coco_catIDs)
    tempIds.remove('person')
    otherIds = coco.getCatIds(catNms=tempIds)

    personImageIDs = coco.getImgIds(catIds=personIds)
    otherImageIDs = coco.getImgIds(catIds=otherIds)

    if amount is not None:
        personImageIDs = personImageIDs[:amount]

    datasets: List[type(Dataset)] = []
    zeroDatasets: List[type(Dataset)] = []

    metadata = Metadata(False, False, "Unknown")
    #hack coco to RoPose datasets and use our backend afterwards
    for id in ProgressBar(personImageIDs):
        annIds = coco.getAnnIds(imgIds=id, catIds=personIds, iscrowd=None)

        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(annIds)

        for ann in anns:
            dataset = Dataset()
            dataset.metadata = metadata
            imagePath = os.path.join(dataDir, "images", dataType, img['file_name'])
            rgbFrame = Image(filePath=imagePath)
            #generate keypoints
            rawKeypoints = ann["keypoints"]
            keypoints = []
            unvalidCounter = 0
            for i in range(0, rawKeypoints.__len__(), 3):
                keypoint = Pose2D.fromData(rawKeypoints[i], rawKeypoints[i+1], 0.0, visible=True)

                if keypoint[0] == 0 and keypoint[1] == 0:
                    keypoint.visible = False
                    unvalidCounter += 1

                keypoints.append(keypoint)

            # generate Bounding Box
            rawBB = ann["bbox"]

            rgbFrame.boundingBox = BoundingBox(Vector2D(int(rawBB[0]), int(rawBB[1])),
                                               Vector2D(int(rawBB[0] + rawBB[2]), int(rawBB[1] + rawBB[3])))


            dataset.annotations = ann
            dataset.yoloData = YoloData.FromCoco(anns)
            dataset.rgbFrame = rgbFrame
            rgbFrame.projectedJoints = keypoints

            if unvalidCounter != keypoints.__len__():
                datasets.append(dataset)
            else:
                zeroDatasets.append(dataset)

        #break if setsize reached
        if setSize is not None and datasets.__len__() >= setSize:
            break

    if mixWithZeroHumans:
        amount = int(mixWithZeroHuamnsFactor * len(datasets))
        # clip to max len of image ids
        amount = max(amount, len(otherImageIDs))

        for i in range(0, amount):
            id = otherImageIDs[i]
            img = coco.loadImgs(id)[0]
            annIds = coco.getAnnIds(imgIds=id, catIds=otherImageIDs, iscrowd=None)
            anns = coco.loadAnns(annIds)

            dataset = Dataset()
            dataset.metadata = metadata
            imagePath = os.path.join(dataDir, "images", dataType, img['file_name'])
            dataset.rgbFrame = Image(filePath=imagePath)
            dataset.annotations = anns
            datasets.extend(zeroDatasets[0:amount])

    return datasets

def LoadCocoSetsOnlyImages(cocoPath = config.cocoPath, cocoDataset=["train2017"], mixWithZeroHumans=False,
                 mixWithZeroHuamnsFactor=0.1, amount: Optional[int]=None, setSize: Optional[int]=None) \
    -> List[type(Dataset)]:
    ret = []

    for dataset in cocoDataset:
        ret.extend(_LoadCocoSetsOnlyImages(cocoPath, dataset, mixWithZeroHumans, mixWithZeroHuamnsFactor, amount,
                                           setSize))

    return ret

def _LoadCocoSetsOnlyImages(cocoPath = config.cocoPath, cocoDataset="train2017", mixWithZeroHumans=False,
                 mixWithZeroHuamnsFactor=0.1, amount: Optional[int]=None, setSize: Optional[int]=None) \
        -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annoFile)
    imageIDs = coco.getImgIds()
    datasets: List[type(Dataset)] = []
    metadata = Metadata(False, False, "Unknown")
    # hack coco to RoPose datasets and use our backend afterwards
    #hack coco to RoPose datasets and use our backend afterwards
    for id in ProgressBar(imageIDs):
        img = coco.loadImgs(id)[0]
        dataset = Dataset()
        dataset.metadata = metadata
        imagePath = os.path.join(dataDir, "images", dataType, img['file_name'])
        rgbFrame = Image(filePath=imagePath)
        dataset.rgbFrame = rgbFrame
        #generate keypoints

        datasets.append(dataset)
        #break if setsize reached
        if setSize is not None and datasets.__len__() >= setSize:
            break

    return datasets

def LoadCocoSetYolo(cocoPath = config.cocoPath, cocoDataset="train2017",  setSize: Optional[int]=None) \
        -> List[type(Dataset)]:
    ret = []

    for dataset in cocoDataset:
        ret.extend(_LoadCocoSetYolo(cocoPath, dataset, setSize))

    return ret

def _LoadCocoSetYolo(cocoPath = config.cocoPath, cocoDataset="train2017",  setSize: Optional[int]=None)  \
        -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    imageIDs = coco.getImgIds()

    datasets: List[type(Dataset)] = []

    metadata = Metadata(False, False, "Unknown")
    #hack coco to RoPose datasets and use our backend afterwards


    for id in  ProgressBar(imageIDs):
        annIds = coco.getAnnIds(imgIds=id, iscrowd=None)

        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(annIds)

        if anns.__len__() <= 0:
            continue

        dataset = Dataset()
        dataset.metadata = metadata
        imagePath = os.path.join(dataDir, "images", dataType, img['file_name'])
        rgbFrame = Image(filePath=imagePath)
        dataset.yoloData = YoloData.FromCoco(anns)

        dataset.rgbFrame = rgbFrame
        datasets.append(dataset)

        # break if setsize reached
        if setSize is not None and datasets.__len__() >= setSize:
            break

    return datasets

def LoadCocoSetHumansYolo(cocoPath = config.cocoPath, cocoDataset="train2017",  setSize: Optional[int]=None) \
        -> List[type(Dataset)]:
    ret = []

    for dataset in cocoDataset:
        ret.extend(_LoadCocoSetHumansYolo(cocoPath, dataset, setSize))

    return ret

def _LoadCocoSetHumansYolo(cocoPath = config.cocoPath, cocoDataset="train2017",  setSize: Optional[int]=None) \
        -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    imageIDs = coco.getImgIds()

    datasets: List[type(Dataset)] = []

    metadata = Metadata(False, False, "Unknown")
    #hack coco to RoPose datasets and use our backend afterwards
    for id in ProgressBar(imageIDs):
        annIds = coco.getAnnIds(imgIds=id, iscrowd=None)

        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(annIds)

        if anns.__len__() <= 0:
            continue

        dataset = Dataset()
        dataset.metadata = metadata
        imagePath = os.path.join(dataDir, "images", dataType, img['file_name'])
        rgbFrame = Image(filePath=imagePath)
        dataset.yoloData = YoloData.FromCoco(anns)

        dataset.rgbFrame = rgbFrame
        datasets.append(dataset)

        # break if setsize reached
        if setSize is not None and datasets.__len__() >= setSize:
            break

    return datasets