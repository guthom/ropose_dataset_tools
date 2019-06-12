import unittest
from unittest import TestCase
import numpy as np
from typing import List, Tuple
from data_framework.DataClasses.BaseTypes.Rotation import Rotation
from data_framework.DataClasses.BaseTypes.Quaternion import Quaternion
from data_framework.DataClasses.BaseTypes.RotationMatrix import RotationMatrix
from data_framework.DataClasses.BaseTypes.EulerAngles import EulerAngles
from data_framework.DataClasses.BaseTypes.Vector3D import Vector3D
from data_framework.DataClasses.BaseTypes.AxisAngles import AxisAngles
from math import pow, sqrt
class TestCollection(object):

    def __init__(self, order: str="xyz"):
        self.order = None
        self.quats = []
        self.rpys = []
        self.rotMs = []
        self.axisAngles = []

        if order is "xyz":
            self.XYZ()
        else:
            raise Exception("No Golden model for this order!")

    def Normalize(self, quat: List[float]) -> List[float]:
        d = sqrt(pow(quat[0], 2) + pow(quat[1], 2) + pow(quat[2], 2) + pow(quat[3], 2))
        x = quat[0] / d
        y = quat[1] / d
        z = quat[2] / d
        w = quat[3] / d
        return [x, y, z, w]

    def XYZ(self):
        self.order = "xyz"
        self.quats = []
        # [x, y, z, w]
        self.quats.append(self.Normalize([0.500, -0.500, 0.5, 0.500]))
        self.quats.append(self.Normalize([0.500, 0.500, 0.500, 0.500]))
        self.quats.append(self.Normalize([0.707, 0, 0, 0.707]))
        self.quats.append(self.Normalize([0.270, 0.653, 0.270, 0.653]))
        self.quats.append(self.Normalize([0.2579252, 0.2579252, 0.0716607, 0.9283393]))

        self.rpys = []
        self.rpys.append([1.571, 0.0, 1.571])
        self.rpys.append([1.571, 1.571, 0.0])
        self.rpys.append([1.571, 0.0, 0.0])
        self.rpys.append([0.7841392, 1.5707963, 0.0])
        self.rpys.append([0.542, 0.542, 0.0])

        self.rotMs = []
        self.rotMs.append(np.array([
            [0.0000000, -1.0000000,  0.0000000],
            [0.0000000,  0.0000000, -1.0000000],
            [1.0000000,  0.0000000,  0.0000000]
        ]))
        self.rotMs.append(np.array([
            [0.0007963,  0.0000000,  0.9999997],
            [0.9999993,  0.0007963, -0.0007963],
            [-0.0007963,  0.9999997,  0.0000006]
        ]))
        self.rotMs.append(np.array([
            [1.0000000,  0.0000000,  0.0000000],
            [0.0000000,  0.0007963, -0.9999997],
            [0.0000000,  0.9999997,  0.0007963]
        ]))
        self.rotMs.append(np.array([
            [0.0000000,  0.0000000,  1.0000000],
            [0.7062160,  0.7079964,  0.0000000],
            [-0.7079964,  0.7062160,  0.0000000]
        ]))
        self.rotMs.append(np.array([
            [0.8566787,  0.0000000,  0.5158504],
            [0.2661016,  0.8566787, -0.4419180],
            [-0.4419180,  0.5158504,  0.7338984]
        ]))

        self.axisAngles = []
        self.axisAngles.append([[0.5773111, -0.5774287, 0.5773111], 2.0946303])
        self.axisAngles.append([[0.5773111, 0.5773111, 0.5774287], 2.0946303])
        self.axisAngles.append([[1, 0, 0], 1.571])
        self.axisAngles.append([[0.3569328, 0.8632485, 0.3569328], 1.7173218])
        self.axisAngles.append([[0.6938437, 0.6938437, 0.1927741], 0.761752])


class QuaternionTests(TestCase):

    def setUp(self):
        self.testCollections = []
        self.testCollections.append(TestCollection(order="xyz"))



    def test_RotationEuqal(self):
        for testCollection in self.testCollections:

            for i in range(0, testCollection.quats.__len__()):
                rotations = []

                rotations.append(Rotation.fromQuat(Quaternion.fromList(testCollection.quats[i]),
                                               order=testCollection.order))
                rotations.append(Rotation.fromEuler(EulerAngles.fromList(testCollection.rpys[i],
                                                                     order=testCollection.order)))

                rotations.append(Rotation.fromRotationMatrix(RotationMatrix(testCollection.rotMs[i],
                                                                        order=testCollection.order)))

                rotations.append(Rotation.fromAxisAngles(AxisAngles.fromList(testCollection.axisAngles[i])))

                results = []
                for rotation in rotations:
                    results.append(np.round(np.array(rotation.rotM.rotM), 3))

                for res1 in results:
                    for res2 in results:
                        equal = np.allclose(res1, res2, atol=0.01)
                        self.assertTrue(equal)
                print("All RotMs are equal!")

                results = []
                for rotation in rotations:
                    results.append(np.array(rotation.quat.toList()))

                for res1 in results:
                    for res2 in results:
                        equal = np.allclose(res1, res2, atol=0.01) or np.allclose(-res1, res2, atol=0.01)
                        self.assertTrue(equal)
                print("All Quats are equal!")

                results = []
                for rotation in rotations:
                    results.append(np.array(rotation.euler.GetRPY()))

                for res1 in results:
                    for res2 in results:
                        equal = np.allclose(res1, res2, atol=0.001)
                        self.assertTrue(equal)
                print("All RPYs are equal!")

                results = []
                for rotation in rotations:
                    results.append(np.array(rotation.axiAngles.toList()))

                for res1 in results:
                    for res2 in results:
                        equal = np.allclose(res1[0], res2[0], atol=0.01) and np.allclose(res1[1], res2[1], atol=0.01)
                        self.assertTrue(equal)
                print("All AxisAngles are equal!")





