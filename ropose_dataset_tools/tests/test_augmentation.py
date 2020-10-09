from unittest import TestCase
import os
import numpy as np
import cv2
from guthoms_helpers.filesystem.FileHelper import FileHelper
from ropose_dataset_tools.Augmentation.Augmentor import Augmenter

import copy
import copy

class AugmentationTests(TestCase):

    def setUp(self):
        self.augmentor = Augmenter()
        imagePath = FileHelper.GetFilePath(__file__)
        imagePath = os.path.join(imagePath, 'test_data', 'ropose_test_dataset', 'depthcam1', 'depthcam1', '0.png')
        self.image = cv2.imread(imagePath)

    def tearDown(self):
        pass

    def test_VisualizeAugmentationPipeline(self):
        self.augmentor.ShowExample(self.image)

