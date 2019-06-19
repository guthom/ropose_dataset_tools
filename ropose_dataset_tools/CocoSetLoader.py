import os
from typing import List
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata

from pycocotools.coco import COCO
import ropose_dataset_tools.config as config

def GetDataSets(path: str):
    dataDirs = []
    for x in os.listdir(path):
        if x != "examples":
            dirName = path + x + "/"
            if os.path.isdir(dirName):
                dataDirs.append(dirName)

    return dataDirs

def LoadCocoSets(cocoPath = config.cocoPath, cocoDataset="train2017", mixWithZeroHumans=False,
                 mixWithZeroHuamnsFactor=0.1) -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    catIds = coco.getCatIds(catNms=['person'])
    imageIDs = coco.getImgIds(catIds=catIds)

    datasets: List[type(Dataset)] = []
    zeroDatasets: List[type(Dataset)] = []

    metadata = Metadata(False, False)
    #hack coco to RoPose datasets and use our backend afterwards
    for id in imageIDs:
        annIds = coco.getAnnIds(imgIds=id, catIds=catIds, iscrowd=None)

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
                keypoint = [rawKeypoints[i], rawKeypoints[i+1]]

                if keypoint[0] == 0 and keypoint[1] == 0:
                    keypoint = [-1, -1]
                    unvalidCounter += 1

                keypoints.append(keypoint)

            # generate Bounding Box
            rawBB = ann["bbox"]

            rgbFrame.boundingBox = BoundingBox.FromList([int(rawBB[0]), int(rawBB[1]), int(rawBB[0] + rawBB[2]),
                                                         int(rawBB[1] + rawBB[3])])


            dataset.annotations = ann
            dataset.yoloData = YoloData.FromCoco(anns)
            dataset.rgbFrame = rgbFrame
            rgbFrame.projectedJoints = keypoints

            if unvalidCounter != keypoints.__len__():
                datasets.append(dataset)
            else:
                zeroDatasets.append(dataset)

    if mixWithZeroHumans:
        amount = int(mixWithZeroHuamnsFactor * len(datasets))
        datasets.extend(zeroDatasets[0:amount])

    return datasets

def LoadCocoSetYolo(cocoPath = config.cocoPath, cocoDataset="train2017")  -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    imageIDs = coco.getImgIds()

    datasets: List[type(Dataset)] = []

    metadata = Metadata(False, False)
    #hack coco to RoPose datasets and use our backend afterwards


    for id in imageIDs:
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


    return datasets

def LoadCocoSetHumansYolo(cocoPath, cocoDataset="train2017") -> List[type(Dataset)]:
    dataDir = cocoPath
    dataType = cocoDataset
    annoFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    coco = COCO(annoFile)

    imageIDs = coco.getImgIds()

    datasets: List[type(Dataset)] = []

    metadata = Metadata(False, False)
    #hack coco to RoPose datasets and use our backend afterwards
    for id in imageIDs:
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

    return datasets