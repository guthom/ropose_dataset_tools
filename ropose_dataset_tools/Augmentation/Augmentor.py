from typing import Optional, Tuple, List
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import ropose_dataset_tools.config as config

class Augmentor(object):

    def __init__(self):
        self.pipeline: Optional[iaa.Sequential] = None

        self.DefineSeq()

    def DefineSeq(self):
        self.pipeline = iaa.Sequential()
        padmode = 'constant'
        cval = config.augmentationCval
        #flipping
        self.pipeline.append(iaa.Fliplr(0.5))
        self.pipeline.append(iaa.Flipud(0.5))

        self.pipeline.append(iaa.Sometimes(0.95, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=padmode,
                                                                pad_cval=cval)))

        self.pipeline.append(
            iaa.Sometimes(0.95,
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-180, 180),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=cval,  # if mode is constant, use a cval between 0 and 255
            mode=padmode  # use any of scikit-image's warping modes (see 2nd image from the top for examples
        )))


    def AugmentImagesAndHeatmaps(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        x, y = self.pipeline(images=x, heatmaps=y)
        return x, y

    def AugmentImagesAndBBs(self, x: np.array, y: List[ia.BoundingBox]) -> Tuple[np.array, np.array]:
        x, y = self.pipeline(images=x, bounding_boxes=y)
        return x, y

    def ShowExample(self, image: np.array):
        self.pipeline.show_grid(image, cols=8, rows=8)