import os
import simplejson as json
from typing import List, Dict
from guthoms_helpers.base_types.Pose3D import Pose3D
from ropose_dataset_tools.DataClasses.Dataset.SensorBase import SensorBase
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose_dataset_tools.DataClasses.Dataset.YoloData import YoloData
from ropose_dataset_tools.DataClasses.Dataset.Image import Image
from ropose_dataset_tools.DataClasses.Dataset.DepthData import DepthData
from ropose_dataset_tools.DataClasses.Dataset.CameraInfo import CameraInfo
from ropose_dataset_tools.DataClasses.Dataset.Metadata import Metadata

import ropose_dataset_tools.config as config

class DataOrganizer:

    def __init__(self, dataPath: str) -> None:
        self.dataPath: str = dataPath
        self.worldTransformPath: str = "world_transforms"
        self.metadataPath: str = "metadata.json"
        self.dirList: List[str] = []
        self.datasets: List[type(Dataset)] = None

        self.datasets = self.LoadData(dataPath)
        pass

    def CheckData(self, dataDirs: [str]):

        tempDirs = dataDirs.copy()

        if not tempDirs.__contains__(self.worldTransformPath):
            raise Exception("Dataset does not contain the directory with world_transforms, this is not a valid RoPose "
                            "dataset!")

        if not os.path.isfile(os.path.join(self.dataPath, self.metadataPath)):
            raise Exception("Dataset does not contain the metadata.json file, this is not a valid RoPose "
                            "dataset!")

        tempDirs.remove(self.worldTransformPath)
        if not tempDirs.__len__() > 0:
            raise Exception("Dataset does not contain a directory with camera data, this is not a valid RoPose "
                            "dataset!")

        print("Dataset in " + self.dataPath + " is valid!")

    @staticmethod
    def GetDirs(dataPath: str) -> List[str]:
        dataDirs = []
        for dirName in os.listdir(dataPath):
            if os.path.isdir(os.path.join(dataPath, dirName)):
                dataDirs.append(dirName)

        return dataDirs

    @staticmethod
    def GetFiles(dataPath: str, endsWith: str) -> List[str]:
        files = []
        for file in os.listdir(dataPath):
            if file.endswith(endsWith):
                files.append(os.path.join(dataPath, file))

        return files

    def LoadData(self, dataPath: str) -> List[type(Dataset)]:
        print("Load raw datasets...")

        #get all dirs
        dataDirs = self.GetDirs(dataPath)

        self.CheckData(dataDirs)

        return self.LoadDataset(dataPath)

    def LoadMetadata(self, dataPath: str) -> type(Metadata):

        filePath = os.path.join(self.dataPath, self.metadataPath)

        rawData = self.LoadJson(filePath)

        metaData = Metadata(rawData["simulated"], rawData["greenscreened"])

        return metaData


    def LoadDataset(self, dataPath: str) -> List[type(Dataset)]:
        datasets: List[Dataset] = []

        metaData = self.LoadMetadata(dataPath)

        dataDirs = self.GetDirs(dataPath)

        worldTransforms = self.LoadTransforms(os.path.join(dataPath, self.worldTransformPath))
        #remobe the world transform path, rest should be cameras
        dataDirs.remove(self.worldTransformPath)

        #rest of dirs are sensor data directories
        sensorData: Dict[str, List[SensorBase]] = dict()

        for sensorDir in dataDirs:
            sensorData[sensorDir] = self.LoadSensorData(dataPath, sensorDir, worldTransforms)

        #fill final Datasets
        for i in range(0, worldTransforms.__len__()):
            for sensor in sensorData:
                dataset = Dataset()
                try:
                    #if exception will happen, just ignore the dataset
                    dataset.worldTransforms = DataOrganizer.FilterTransforms(worldTransforms[i])
                    for frame in sensorData[sensor]:
                        singleData = sensorData[sensor][frame][i]
                        if singleData.frameType == FrameTypes.Image:
                            dataset.rgbFrame = singleData
                        elif singleData.frameType == FrameTypes.Depth:
                            dataset.depthFrame = singleData

                    dataset.metadata = metaData
                    dataset.yoloData = YoloData.FromRopose(dataset)
                    datasets.append(dataset)
                except:
                    pass

        return datasets

    @staticmethod
    def LoadSensorData(dataPath: str, cameraName: str, worldTransforms: List[Dict[str, type(Pose3D)]]) \
            -> Dict[str, List[type(SensorBase)]]:
        dataDir = os.path.join(dataPath, cameraName)
        dirs = DataOrganizer.GetDirs(dataDir)

        ret = dict()

        if dirs.__len__() == 0:
            raise Exception("Sensor directory is emtpy!. Seems like you try to load a not yet "
                            "supported RoPose datastructure!")

        if not dirs.__contains__("transforms"):
            raise Exception("Sensor does not contain a transform directory. Seems like you try to load a not yet "
                            "supported RoPose datastructure!")


        transforms = DataOrganizer.LoadTransforms(os.path.join(dataDir, "transforms"))

        #remove the transform dir because we don't need that anymore
        dirs.remove("transforms")

        for dir in dirs:
            frameType = DataOrganizer.PredictDirContent(os.path.join(dataDir, dir))

            if frameType is FrameTypes.Image:
                frames = DataOrganizer.LoadImageData(os.path.join(dataDir, dir), cameraName,
                                                     worldTransforms=worldTransforms, transforms=transforms)
            elif frameType is FrameTypes.Depth:
                frames = DataOrganizer.LoadDepthData(os.path.join(dataDir, dir), cameraName,
                                                     worldTransforms=worldTransforms, transforms=transforms)
            else:
                raise Exception("Can't predict frame type. Seems like you try to load a not yet supported RoPose "
                                "datastructure!")

            ret[dir] = frames

        return ret

    @staticmethod
    def PredictDirContent(dataDir: str) -> type(FrameTypes):
        files = os.listdir(dataDir)

        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                return FrameTypes.Image
            elif file.endswith(".xml"):
                return FrameTypes.Depth
            else:
                return None

    @staticmethod
    def LoadCameraInfo(dataDir: str) -> type(CameraInfo):

        infoPath = os.path.join(dataDir, "CameraInfo.json")
        if not os.path.isfile(infoPath):
            raise Exception("Can't find CameraInfo.json. Seems like you try to load a bad RoPose datastructure!")

        dataRaw = DataOrganizer.LoadJson(infoPath)

        data = dict()

        for dataPart in dataRaw:
            if dataPart == "CameraName":
                data[dataPart] = dataRaw[dataPart]
            else:
                data[dataPart] = json.loads(dataRaw[dataPart])

        data["filePath"] = infoPath

        return CameraInfo.fromJson(data)

    @staticmethod
    def LoadImageData(dataDir: str, cameraName: str, worldTransforms:  List[Dict[str, type(Pose3D)]],
                      transforms: List[Dict[str, type(Pose3D)]] = None) -> List[type(Image)]:

        cameraInfo = DataOrganizer.LoadCameraInfo(dataDir)

        files = DataOrganizer.GetFiles(dataDir, ".png")
        if files.__len__() == 0:
            files = DataOrganizer.GetFiles(dataDir, ".jpg")

        imageData = [Image] * (files.__len__())

        for file in files:
            index = DataOrganizer.GetFrameIndex(file)
            imageData[index] = DataOrganizer.LoadSingleImageSet(file, cameraInfo,
                                                                worldTransforms[index][cameraName + "_link"],
                                                                transforms[index])

        return imageData

    @staticmethod
    def LoadSingleImageSet(dataPath: str, cameraInfo: type(CameraInfo), sensorPose: type(Pose3D),
                           transforms:  Dict[str, type(Pose3D)] = None) -> type(Image):
        try:
            transforms = DataOrganizer.FilterTransforms(transforms)
        except:
            return None

        return Image(dataPath, cameraInfo, sensorPose, transforms)

    @staticmethod
    def LoadDepthData(dataDir: str, cameraName: str, worldTransforms:  List[Dict[str, type(Pose3D)]],
                      transforms: List[Dict[str, type(Pose3D)]] = None) -> List[type(DepthData)]:

        cameraInfo = DataOrganizer.LoadCameraInfo(dataDir)

        files = DataOrganizer.GetFiles(dataDir, ".xml")
        depthData = [DepthData] * (files.__len__())

        for file in files:
            index = DataOrganizer.GetFrameIndex(file)
            depthData[index] = DataOrganizer.LoadSingleDepthSet(file, cameraInfo,
                                                                worldTransforms[index][cameraName + "_link"],
                                                                transforms[index])

        return depthData

    @staticmethod
    def LoadSingleDepthSet(dataPath: str, cameraInfo: type(CameraInfo), sensorPose: type(Pose3D),
                           transforms: List[type(Pose3D)] = None) -> type(DepthData):
        #sensorPose = transforms[cameraInfo.cameraName + "_link"]
        #transforms = DataOrganizer.FilterTransforms(transforms)
        return DepthData(sensorPose)

    @staticmethod
    def LoadTransforms(dataDir: str) -> List[Dict[str, type(Pose3D)]]:
        files = DataOrganizer.GetFiles(dataPath=dataDir, endsWith=".json")

        transforms = [Dict[str, type(Pose3D)]] * files.__len__()

        for file in files:
            index = DataOrganizer.GetDatasetIndex(file)
            jsonData = DataOrganizer.LoadJson(file)

            poses: Dict[str, type(Pose3D)] = dict()
            for data in jsonData:
                if data != "ros_time" and data != "redundant":
                    poses[data] = Pose3D.fromDict(jsonData[data])

            transforms[index] = poses

        return transforms

    @staticmethod
    def FilterTransforms(transforms: Dict[str, type(Pose3D)]) -> List[type(Pose3D)]:
        ret = []

        for joint in config.linkOrder:
            ret.append(transforms[joint])

        return ret

    @staticmethod
    def LoadJson(filePath: str) -> Dict:
        ret = dict()
        with open(filePath) as data_file:
            data = json.load(data_file)

            for d in data:
                    ret[d] = data[d]

        return ret

    @staticmethod
    def GetDatasetIndex(filename: str) -> int:
        return int(filename.split("/")[-1].split(".")[0].split("_")[-1])

    @staticmethod
    def GetFrameIndex(filename: str) -> int:
        return int(filename.split("/")[-1].split(".")[0])

