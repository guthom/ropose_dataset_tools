from __future__ import print_function, division
from typing import Tuple
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset

from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Vector2D import Vector2D
from math import cos, sin, pi

import copy
import random
import numpy as np
import ropose_dataset_tools.config as config
from skimage.transform import warp
from skimage.util.noise import random_noise
from skimage.draw import rectangle, ellipse, line, random_shapes
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

class Augmenter:

    counter = 0
    def __init__(self, flip=True, rotate=True, tranform=True, scale=True):
        self.augmentMethods = []

        self.flip = flip
        self.rotate = rotate  # rotate
        self.tranform = tranform  # shift actually
        self.scale = scale

        if (rotate and tranform and scale and flip):
            #Affine will take care of everithing together
            self.augmentMethods.append(self.Affine)
            return

        if (flip):
            self.augmentMethods.append(self.flip)
        if (rotate):
            self.augmentMethods.append(self.Rotate)
        if (tranform):
            self.augmentMethods.append(self.Transform)
        if (scale):
            self.augmentMethods.append(self.scale)

    @staticmethod
    def DecideByProb(prob: float = 0.5):
        return random.random() <= prob

    @staticmethod
    def EraseMask(img: np.array, bb: BoundingBox, randomNoise: bool = True):

        objectType = random.randint(0, 1)

        if objectType == 0:
            #rectangles
            rr, cc = rectangle(start=(int(bb.P1[0]), int(bb.P1[1])), end=(int(bb.P2[0]), int(bb.P2[1])))

            if randomNoise:
                # create random numpy array
                img[rr, cc] = np.random.rand(rr.shape[0], rr.shape[1], 3)
            else:
                img[rr, cc] = config.augmentationCval

        elif objectType == 1:
            #circles/ellipses
            rr, cc = ellipse(r=int(bb.midX), c=int(bb.midY), r_radius=max(1, int(bb.width/2)),
                             c_radius=max(1, int(bb.height/2)), shape=img.shape,
                             rotation=np.deg2rad(random.randint(0, 180)))

            if randomNoise:
                # create random numpy array
                mask = np.random.rand(rr.shape[0], 3)
                img[rr, cc] = mask
            else:
                img[rr, cc] = config.augmentationCval
        else:
            raise Exception("Not Implemented!")


        return img



    @staticmethod
    def AddRandomErasing(img: np.array,  probabilty: float = 0.5, maxObjectCount: int = 10, coverRange: float = 0.3):

        raise Exception("Not yet tested!")
        #inspired by the idea of
        #https://arxiv.org/abs/1708.04896
        #https://github.com/zhunzhong07/Random-Erasing

        if Augmenter.DecideByProb(probabilty):
            objCount = int(random.uniform(1, maxObjectCount))

            #create objects
            for i in range(0, objCount):
                area = random.uniform(0.1, coverRange)

                #take care for a min height/width of 2 pixels
                heigth = max(1, int(img.shape[0] * area))
                width = max(1, int(img.shape[1] * area))

                midX = int(random.uniform(0, img.shape[0]))
                midY = int(random.uniform(0, img.shape[1]))

                #create fake bb to work with
                bb = BoundingBox.FromMidAndRange(Vector2D(midX, midY), Vector2D(heigth, width))
                bb = bb.ClipToShape(img.shape)

                img = Augmenter.EraseMask(img, bb)

        return img

    @staticmethod
    def GetNoneManipulativeValues(inputRes: Tuple[int, int], outputRes: Tuple[int, int] = None):
        AugCollection = dict()

        if outputRes is None:
            outputRes = inputRes

        AugCollection["flip"] = False
        AugCollection["randomErasing"] = False
        AugCollection["scale"] = 1.0
        AugCollection["rotation"] = 0.0
        AugCollection["shear"] = [0.0, 0.0]
        placing_inp = [0.0,  0.0]  # y direction
        AugCollection["placing_inp"] = placing_inp
        AugCollection["inputRes"] = inputRes

        gtFactor = [inputRes[0] / outputRes[0], inputRes[1] / outputRes[1]]
        AugCollection["placing_gt"] = [placing_inp[0] / gtFactor[0], placing_inp[1] / gtFactor[1]]
        AugCollection["outputRes"] = outputRes

        return AugCollection

    @staticmethod
    def GetRandomValues(inputRes: Tuple[int, int], outputRes: Tuple[int, int] = None):
        AugCollection = dict()

        if outputRes is None:
            outputRes = inputRes

        if not Augmenter.DecideByProb(config.onTheFlyAugmentationProbability):
            #augmentation will not be applied every time
            return Augmenter.GetNoneManipulativeValues(inputRes, outputRes)

        AugCollection["flip"] = bool(random.getrandbits(1))
        AugCollection["randomErasing"] = True
        AugCollection["scale"] = random.uniform(0.85, 1.15)
        AugCollection["rotation"] = random.uniform(-180.0, 180.0)
        AugCollection["shear"] = [random.uniform(-0.1, -0.1), random.uniform(0.1, 0.1)]
        placing_inp = [random.randint(-25, 25), #x direction
                       random.randint(-25, 25)] #y direction
        AugCollection["placing_inp"] = placing_inp
        AugCollection["inputRes"] = inputRes

        gtFactor = [inputRes[0]/outputRes[0], inputRes[1]/outputRes[1]]
        AugCollection["placing_gt"] = [placing_inp[0] / gtFactor[0], placing_inp[1] / gtFactor[1]]
        AugCollection["outputRes"] = outputRes


        return AugCollection

    def Flip(self, dataset, augCollection):
        raise Exception("Not Implemented! Use Affine with GetRandomValues!")

    def Rotate(self,  dataset, augCollection):
        raise Exception("Not Implemented! Use Affine with GetRandomValues!")

    def Transform(self, dataset, augCollection):
        raise Exception("Not Implemented! Use Affine with GetRandomValues!")

    @staticmethod
    def Affine(augCollection, inputRes=None, outputRes=None):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        if inputRes is None:
            inputRes = augCollection["inputRes"]

        if outputRes is None:
            outputRes = augCollection["outputRes"]


        center2zero_inp = np.array([[1., 0., inputRes[1]/2],
                                [0., 1., inputRes[0]/2],
                                [0., 0., 1.]])

        center2zero_gt = np.array([[1., 0., outputRes[1]/2],
                                [0., 1., outputRes[0]/2],
                                [0., 0., 1.]])

        translation_inp = np.array([[1., 0., augCollection["placing_inp"][0]],
                                    [0., 1., augCollection["placing_inp"][1]],
                                    [0., 0., 1.]])

        translation_gt = np.array([[1., 0., augCollection["placing_gt"][0]],
                                    [0., 1., augCollection["placing_gt"][1]],
                                    [0., 0., 1.]])


        A = cos(augCollection["rotation"] / 180. * pi)
        B = sin(augCollection["rotation"] / 180. * pi)

        rotate = np.array([
            [A, -B, 0.],
            [B, A, 0.],
            [0., 0., 1.],
        ])

        scale = np.array([
            [augCollection["scale"], 0., 0.],
            [0., augCollection["scale"], 0.0],
            [0., 0., 1.]
        ])

        flip = np.array([
            [-1. if augCollection["flip"] else 1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        shear = np.array([
            [1., augCollection["shear"][1], 0.],
            [augCollection["shear"][0], 1., 0.],
            [0., 0., 1.]
        ])

        zero2center_inp = np.array([[1., 0., -inputRes[1]/2],
                                    [0., 1., -inputRes[0]/2],
                                    [0., 0., 1.]])


        zero2center_gt = np.array([[1., 0., -outputRes[1]/2],
                                   [0., 1., -outputRes[0]/2],
                                   [0., 0., 1.]])

        M_img = center2zero_inp.dot(flip).dot(scale).dot(shear).dot(rotate).dot(translation_inp).dot(zero2center_inp)

        if outputRes is not None:
            M_gt = center2zero_gt.dot(flip).dot(scale).dot(shear).dot(rotate).dot(translation_gt).dot(zero2center_gt)
        else:
            M_gt = None

        return M_img, M_gt

    @staticmethod
    def AugmentImg(img, M_img, cval=config.augmentationCval):

        img = warp(image=img, inverse_map=np.linalg.inv(M_img), cval=cval, mode="constant")

        return img

    @staticmethod
    def AugmentYoloData(dataset: Dataset, M_img):

        for i in range(0, dataset.yoloData.resizedBoundingBoxes.__len__()):
            boundingBox = dataset.yoloData.resizedBoundingBoxes[i]
            boundingBox = Augmenter.AugmentBoundingBox(boundingBox, M_img)
            dataset.yoloData.resizedBoundingBoxes[i] = boundingBox

        return dataset

    @staticmethod
    def AugmentBoundingBox(boundingBox: BoundingBox, M_img) -> BoundingBox:

        mat = M_img

        edges = boundingBox.GetEdgePoints()

        transEdgeList = []
        for j in range(0, edges.__len__()):
            edge = np.array([edges[j][0], edges[j][1], 1])
            edge = np.matmul(mat, edge)
            transEdgeList.append(Vector2D(edge[0], edge[1]))

        boundingBox = BoundingBox.CreateBoundingBox(transEdgeList, expandBox=False)
        return boundingBox


    @staticmethod
    def AugmentHeatmaps(heatmaps, M_gt, heatmapMin=0.0, heatmapMax=1.0):

        for j in range(0, heatmaps.shape[0] - 1):
            heatmaps[j] = warp(image=heatmaps[j], inverse_map=np.linalg.inv(M_gt), cval=heatmapMin, mode="constant")

        # Background need constant = 1.0
        heatmaps[-1] = warp(image=heatmaps[-1], inverse_map=np.linalg.inv(M_gt), cval=heatmapMax, mode="constant")

        return heatmaps

    @staticmethod
    def AugmentHeatmap(heatmap, M_gt, cval):

        heatmap = warp(image=heatmap, inverse_map=np.linalg.inv(M_gt), cval=cval, mode="constant")

        return heatmap

    @staticmethod
    def AugmentGT(dataset, M):

        M = np.matrix(np.linalg.inv(M))
        dataset = copy.deepcopy(dataset)
        jointPoses = dataset["joint_pos2D"]

        for i in range(0, jointPoses.__len__()):
            pose = jointPoses[str(i)]
            pose.append(1.0)
            pose = M.dot(np.matrix(pose).transpose())
            #pose = np.matmul(M, np.matrix(pose).transpose())
            jointPoses[str(i)] = [pose[0][0], pose[1][0]]

        dataset["joint_pos2D"] = jointPoses
        return dataset