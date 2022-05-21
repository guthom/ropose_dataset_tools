from typing import Optional, Tuple, List
from typing import Callable, Optional, List
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import ropose_dataset_tools.config as config


from imgaug.augmentables.batches import UnnormalizedBatch, Batch


class Augmentor(object):

    def __init__(self):
        self.pipeline: Optional[iaa.Sequential] = None
        self.pipeline100: Optional[iaa.Sequential] = None

        self.DefineSeq()

    def Sometimes(self, func: Callable, prob: float = 0.50) -> Callable:
        return iaa.Sometimes(prob, func)

    def DefineSeq(self):
        self.pipeline = iaa.Sequential()

        padmode = 'constant'
        cval = config.augmentationCval

        pipe = []
        # Flipping
        pipe.append(iaa.Fliplr(0.5))
        pipe.append(iaa.Flipud(0.5))

        # Affine transformation
        pipe.append(
            self.Sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(-180, 180),
                #translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, shear={"x": (-2, 2), "y": (-2, 2)},
                order=1, cval=cval, mode=padmode)
            )
        )
        self.pipeline.extend(pipe)

        pipe100 = []
        self.pipeline100 = iaa.Sequential()
        # Flipping
        pipe100.append(iaa.Fliplr(0.5))
        pipe100.append(iaa.Flipud(0.5))

        # Affine transformation
        pipe100.append(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(-180, 180),
                #translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, shear={"x": (-2, 2), "y": (-2, 2)},
                order=1, cval=cval, mode=padmode)
            )
        self.pipeline100.extend(pipe100)



    def AugmentImagesAndHeatmaps(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        expanded = False
        if len(x.shape) == 3:
            expanded = True
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            # transpose to satisfy imgaugs expectations
            y = y.transpose((0, 2, 3, 1))

        #batch = UnnormalizedBatch(images=x, heatmaps=y)
        #augmented = self.pipeline.augment_batches(batch)

        #for part in augmented:
        #    x = part.images_aug
        #    y = part.heatmaps_aug

        x, y = self.pipeline.augment(images=x, heatmaps=y)

        if expanded:
            x = np.squeeze(x, axis=0)
            # transpose back to satisfy our expectations
            y = y.transpose((0, 3, 1, 2))
            y = np.squeeze(y, axis=0)

            return x, y

    def AugmentImagesAndBBs(self, x: np.array, y: List[ia.BoundingBox], forceAugmentation: bool = False) -> Tuple[np.array, np.array]:
        expanded = False
        if len(x.shape) == 3:
            expanded = True
            x = np.expand_dims(x, axis=0)
            #y = np.expand_dims(y, axis=0)

        if forceAugmentation:
            x, y = self.pipeline.augment(images=x, bounding_boxes=y)
        else:
            x, y = self.pipeline100.augment(images=x, bounding_boxes=y)

        if expanded:
            x = np.squeeze(x, axis=0)
            #y = np.squeeze(y, axis=0)

        return x, y

    def ShowExample(self, image: np.array):
        self.pipeline.show_grid(image, cols=8, rows=8)