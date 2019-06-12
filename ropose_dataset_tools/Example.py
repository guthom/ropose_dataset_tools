import ropose_dataset_tools.DataSetLoader as roposeLoader
import ropose_dataset_tools.CocoSetLoader as cocoLoader

#load single ropose dataset
roposeSet = roposeLoader.LoadDataSet(path="path/to/single/set")

#load all ropose datasets in dir
roposeSets = roposeLoader.LoadDataSets(path="path/to/dir/with/sets")

#load coco in ropose dataset structure
cocoSet = cocoLoader.LoadCocoSets("/path/to/coco/")

#load coco in ropose dataset yolo structure
cocoSetYolo = cocoLoader.LoadCocoSetYolo("/path/to/coco/")

#load only human tagged coco-samples in ropose dataset yolo structure
cocoSetHuman = cocoLoader.LoadCocoSetHumansYolo("/path/to/coco/")