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

        self.DefineSeq()

    def Sometimes(self, func: Callable, prob: float = 0.50) -> Callable:
        return iaa.Sometimes(prob, func)

    def DefineSeq(self):
        self.pipeline = iaa.Sequential()

        padmode = 'constant'
        cval = config.augmentationCval

        pipe = []
        # Flipping
        #pipe.append(iaa.Fliplr(0.5))
        #pipe.append(iaa.Flipud(0.5))

        # Affine transformation
        pipe.append(
            self.Sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-180, 180), shear={"x": (-2, 2), "y": (-2, 2)}, order=1, cval=cval, mode=padmode)
            )
        )

        # Noise Augmentation
        '''
        pipe.append(
            self.Sometimes(
                iaa.SomeOf((1, 2), [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.MedianBlur(k=(3, 5))
                    ]),
                    iaa.OneOf([
                        iaa.SaltAndPepper(0.1, per_channel=True),
                        iaa.imgcorruptlike.Spatter(severity=2)
                    ]),
                ]))
        )
        '''
        # Color Channel Augmentation
        #pipe.append(
        #    self.Sometimes(
        #        iaa.OneOf([
        #            iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True),
        #            iaa.ChangeColorTemperature((1100, 10000))
        #        ]), prob=0.25)
        #)


        #pipe.append(self.Sometimes(iaa.CropAndPad(percent=(-0.15, 0.15), pad_mode=padmode, pad_cval=cval)))
        self.pipeline.append(iaa.Sometimes(0.75, pipe))


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

    def AugmentImagesAndBBs(self, x: np.array, y: List[ia.BoundingBox]) -> Tuple[np.array, np.array]:
        expanded = False
        if len(x.shape) == 3:
            expanded = True
            x = np.expand_dims(x, axis=0)
            #y = np.expand_dims(y, axis=0)

        x, y = self.pipeline.augment(images=x, bounding_boxes=y)

        if expanded:
            x = np.squeeze(x, axis=0)
            #y = np.squeeze(y, axis=0)

        return x, y

    def ShowExample(self, image: np.array):
        self.pipeline.show_grid(image, cols=8, rows=8)