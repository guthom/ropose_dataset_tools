import os
from typing import List
from ropose_dataset_tools.DataOrganizer import DataOrganizer
from ropose_dataset_tools.DataClasses import Dataset

def GetDataSets(path: str):
    dataDirs = []
    for x in os.listdir(path):
        if x != "examples":
            dirName = path + x + "/"
            if os.path.isdir(dirName):
                dataDirs.append(dirName)

    return dataDirs

def LoadDir(path: str) -> List[type(Dataset)]:
    dirs = GetDataSets(path)
    datasets = []
    for datasetDir in dirs:
        organizer = DataOrganizer(dataPath=datasetDir)
        datasets.extend(organizer.datasets)

    return datasets

def LoadDataSets(realSetPath, simulatedPath = None, mixRealWithSimulation: bool = True,
                 mixSimulationFactor: int = 0.5) -> List[type(Dataset)]:

    dataSets_Real = GetDataSets(realSetPath)

    roposData: List[type(DataOrganizer)] = []
    datasets: List[type(Dataset)] = []

    for datasetDir in dataSets_Real:
        roposData.append(DataOrganizer(dataPath=datasetDir))
        datasets.extend(roposData[-1].datasets)

    if simulatedPath is not None:
        dataSets_Sim = GetDataSets(simulatedPath)

        if mixRealWithSimulation:
            simDatasets: List[type(Dataset)] = []
            for datasetDir in dataSets_Sim:
                roposData.append(DataOrganizer(dataPath=datasetDir))
                simDatasets.extend(roposData[-1].datasets)

            simulationCount = int(datasets.__len__() / mixSimulationFactor)
            datasets.extend(simDatasets[:simulationCount])

    return datasets


def LoadDataSet(path: str) -> List[type(Dataset)]:
    return DataOrganizer(dataPath=path).datasets

def LoadDataSetsForFinetuning(paths: List[str], includeNones:bool = False) -> List[type(Dataset)]:
    ret = []

    for dir in paths:
        #append none to reset tracker
        if includeNones:
            ret.append(None)
        ret.extend(LoadDataSet(dir))

    return ret


def FindImages(path: str)-> List[str]:
    images = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            images.append(os.path.join(path, file))
    return images
