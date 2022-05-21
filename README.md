# RoPose-Dataset-Tools
This repository contains the Dataset-Tools we developed for the RoPose-System. 

This includes:
* Loading and processing of the datasets generated with the RoPose-Datagrabber (https://github.com/guthom/ropose_datagrabber)

* Augmentation of the Datasets for training purposes

## Prerequisites
* Python 3.6 (for Typing etc.)
* Numpy
* Matplotlib
* OpenCV
* Pycocotools
* scikit-image
* simplejson
* sklearn
* imgaug

## Installing (Source)
Clone the needed repositories to your virtual environment and install requirements:

git clone https://github.com/guthom/ropose_dataset_tools
cd ropose_dataset_tools
pip install -r requirements.txt


## Usage
``` python
import ropose_dataset_tools.DataSetLoader as roposeLoader
import ropose_dataset_tools.CocoSetLoader as cocoLoader

#load single ropose dataset
roposeSet = roposeLoader.LoadDataSet(path="path/to/single/set/")

#load all ropose datasets in dir
roposeSets = roposeLoader.LoadDataSets(config.dataPathpath="path/to/dir/with/sets/")

#load coco in ropose dataset structure
cocoSet = cocoLoader.LoadCocoSets("/path/to/coco/")

#load coco in ropose dataset yolo structure
cocoSetYolo = cocoLoader.LoadCocoSetYolo(config.cocoPath"/path/to/coco/")

#load only human tagged coco-samples in ropose dataset yolo structure
cocoSetHuman = cocoLoader.LoadCocoSetHumansYolo("/path/to/coco/")
```

## Open Source Acknowledgments
This work uses parts from:
* **simplejson** https://simplejson.readthedocs.io/en/
* **numpy** https://www.numpy.org/
* **OpenCV** https://opencv.org/
* **MatplotLib** https://matplotlib.org/
* **pycocotools** http://cocodataset.org
* **skimage** https://scikit-image.org/
* **sklearn** https://scikit-learn.org/stable/
* **imgaug** https://github.com/aleju/imgaug
* **
**Thanks to ALL the people who contributed to the projects!**

## Authors

* **Thomas Gulde**

Cognitive Systems Research Group, Reutlingen-University:
https://cogsys.reutlingen-university.de/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation
Please cite the following papers if this code is helpful in your research. 

```bib
@inproceedings{gulde2019roposeReal,
  title={RoPose-Real: Real World Dataset Acquisition for Data-Driven Industrial Robot Arm Pose Estimation},
  author={Gulde, Thomas and Ludl, Dennis and Andrejtschik, Johann and Thalji, Salma and Curio, Crist{\'o}bal},
  booktitle={2019 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019},
  organization={IEEE}
}

@inproceedings{gulde2018ropose,
  title={RoPose: CNN-based 2D Pose Estimation of industrial Robots},
  author={Gulde, Thomas and Ludl, Dennis and Curio, Crist{\'o}bal},
  booktitle={2018 IEEE 14th International Conference on Automation Science and Engineering (CASE)},
  pages={463--470},
  year={2018},
  organization={IEEE}
}
```

