from __future__ import print_function, division
from typing import Tuple
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from math import cos, sin, pi
import copy
import random
import numpy as np
from skimage.transform import warp

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

class Augmenter:

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
    def GetRandomValues(inputRes: Tuple[int, int], outputRes: Tuple[int, int] = None):
        AugCollection = dict()

        if outputRes is None:
            outputRes = inputRes


        AugCollection["flip"] = bool(random.getrandbits(1))
        AugCollection["scale"] = random.uniform(0.85, 1.15)
        AugCollection["rotation"] = random.uniform(-35.0, 35.0)
        AugCollection["shear"] = [random.uniform(0, 0), random.uniform(0, 0)]
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


        center2zero_inp = np.array([[1., 0., inputRes[0]/2],
                                [0., 1., inputRes[1]/2],
                                [0., 0., 1.]])

        center2zero_gt = np.array([[1., 0., outputRes[0]/2],
                                [0., 1., outputRes[1]/2],
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
            [-1 if augCollection["flip"] else 1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        shear = np.array([
            [1., augCollection["shear"][1], 0.],
            [augCollection["shear"][0], 1., 0.],
            [0., 0., 1.]
        ])

        zero2center_inp = np.array([[1., 0., -inputRes[0] / 2],
                                    [0., 1., -inputRes[1] / 2],
                                    [0., 0., 1.]])

        zero2center_gt = np.array([[1., 0., -outputRes[0] / 2],
                                    [0., 1., -outputRes[1] / 2],
                                    [0., 0., 1.]])

        M_img = center2zero_inp.dot(rotate).dot(flip).dot(translation_inp).dot(zero2center_inp).dot(scale)

        if outputRes is not None:
            M_gt = center2zero_gt.dot(rotate).dot(flip).dot(translation_gt).dot(zero2center_gt).dot(scale)
        else:
            M_gt = None

        return M_img, M_gt

    @staticmethod
    def AugmentImg(img, M_img):

        img = warp(image=img, inverse_map=np.linalg.inv(M_img), cval=0.4980392156862745, mode="constant")

        return img

    @staticmethod
    def AugmentYoloData(dataset: Dataset, M_img):

        mat = M_img

        for i in range(0, dataset.yoloData.resizedBoundingBoxes.__len__()):
            boundingBox = dataset.yoloData.resizedBoundingBoxes[i]
            edges = boundingBox.GetEdgePoints()

            transEdgeList = []
            for j in range(0, edges.__len__()):
                edge = np.array([edges[j][0], edges[j][1], 1])
                edge = np.matmul(mat, edge)
                transEdgeList.append([edge[0], edge[1]])

            boundingBox = BoundingBox.CreateBoundingBox(transEdgeList, expandBox=False)
            dataset.yoloData.resizedBoundingBoxes[i] = boundingBox

        return dataset

    @staticmethod
    def AugmentHeatmaps(heatmaps, M_gt):

        for j in range(0, heatmaps.shape[0] - 1):
            heatmaps[j] = warp(image=heatmaps[j], inverse_map=np.linalg.inv(M_gt), cval=0.0, mode="constant")

        # Background need constant = 1.0
        heatmaps[-1] = warp(image=heatmaps[-1], inverse_map=np.linalg.inv(M_gt), cval=1.0, mode="constant")

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