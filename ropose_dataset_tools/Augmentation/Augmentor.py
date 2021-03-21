from typing import Optional, Tuple, List
from typing import Callable, Optional, List
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import ropose_dataset_tools.config as config

class Augmentor(object):

    def __init__(self):
        self.pipeline: Optional[iaa.Sequential] = None

        self.DefineSeq()

    def Sometimes(self, func: Callable, prob: float = 0.75) -> Callable:
        return iaa.Sometimes(prob, func)

    def DefineSeq(self):
        self.pipeline = iaa.Sequential()

        padmode = 'constant'
        cval = config.augmentationCval

        # Flipping
        self.pipeline.append(iaa.Fliplr(0.5))
        self.pipeline.append(iaa.Flipud(0.5))

        # Affine transformation
        self.pipeline.append(
            self.Sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180), shear=(-16, 16), order=1, cval=cval, mode=padmode)
            )
        )

        # Noise Augmentation
        self.pipeline.append(
            self.Sometimes(
                iaa.SomeOf((1, 2), [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11))
                    ]),
                    iaa.OneOf([
                        iaa.SaltAndPepper(0.1, per_channel=True),
                        iaa.imgcorruptlike.Spatter(severity=2)
                    ]),
                ]))
        )

        # Color Channel Augmentation
        self.pipeline.append(
            self.Sometimes(
                iaa.OneOf([
                    iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                    iaa.ChangeColorTemperature((1100, 10000))
                ]),)
        )


        self.pipeline.append(self.Sometimes(iaa.CropAndPad(percent=(-0.15, 0.15), pad_mode=padmode, pad_cval=cval)))


    def AugmentImagesAndHeatmaps(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        x, y = self.pipeline(images=x, heatmaps=y)
        return x, y

    def AugmentImagesAndBBs(self, x: np.array, y: List[ia.BoundingBox]) -> Tuple[np.array, np.array]:
        x, y = self.pipeline(images=x, bounding_boxes=y)
        return x, y

    def ShowExample(self, image: np.array):
        self.pipeline.show_grid(image, cols=8, rows=8)