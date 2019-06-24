from typing import List
import sys
import numpy as np
import cv2
import matplotlib.patches as patches

class BoundingBox(object):

    x1: float = None
    y1: float = None
    x2: float = None
    y2: float = None

    width: float = None
    height: float = None

    midX: float = None
    midY: float = None

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.SetWidth()
        self.SetHeight()
        self.SetMidX()
        self.SetMidY()

    def __getitem__(self, i):
        if i == 0:
            return self.x1
        if i == 1:
            return self.y1
        if i == 2:
            return self.x2
        if i == 3:
            return self.y2

        raise Exception("Index not supported!")
    @classmethod
    def FromTwoPoints(cls, p1: List[float], p2: List[float]):

        if p1[0] < p2[0]:
            x1 = p1[0]
            x2 = p2[0]
        else:
            x1 = p2[0]
            x2 = p1[0]

        if p1[1] < p2[1]:
            y1 = p1[1]
            y2 = p2[1]
        else:
            y1 = p2[1]
            y2 = p1[1]

        return cls(x1, y1, x2, y2)


    @classmethod
    def CreateBoundingBox(cls, keyPoints: List[List[float]], expandBox= True, max_x_val: int = sys.maxsize, max_y_val: int = sys.maxsize):
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        for joint in keyPoints:
            x = joint[0]
            y = joint[1]
            min_x = min_x if x > min_x else x
            min_y = min_y if y > min_y else y
            max_x = max_x if x < max_x else x
            max_y = max_y if y < max_y else y

        bb_width = max_x - min_x
        bb_height = max_y - min_y

        if expandBox:
            bb_expand_width = 0.35 * bb_width
            bb_expand_height = 0.35 * bb_height

            min_x -= bb_expand_width
            min_y -= bb_expand_height
            max_x += bb_expand_width
            max_y += bb_expand_height

        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x = max_x if max_x < max_x_val else max_x_val
        max_y = max_y if max_y < max_y_val else max_y_val
        return BoundingBox.FromList([int(min_x), int(min_y), int(max_x), int(max_y)])

    def  GetEdgePoints(self) -> List[List[float]]:

        #1 - 2
        #|   |
        #3 - 4

        p1 = [self.midX - self.width/2, self.midY - self.height/2]
        p2 = [self.midX + self.width/2, self.midY - self.height/2]
        p3 = [self.midX - self.width/2, self.midY + self.height/2]
        p4 = [self.midX + self.width/2, self.midY + self.height/2]

        return [p1, p2, p3, p4]


    @classmethod
    def FromList(cls, list: List[float]):
        return cls(list[0], list[1], list[2], list[3])

    def SetWidth(self, width: float = None):
        if width is None:
            width = self.x2 - self.x1

        self.width = width

    def SetHeight(self, height: float = None):
        if height is None:
            height = self.y2 - self.y1

        self.height = height

    def SetMidX(self, midX: float = None):
        if midX is None:
            midX = self.x1 + self.width/2

        self.midX = midX

    def SetMidY(self, midY: float = None):
        if midY is None:
            midY = self.y1 + self.height/2

        self.midY = midY

    def NormWithImage(self, image: np.array):
        w, h, _ = image.shape
        ret = BoundingBox(self.x1 / w, self.y1 / h,  self.x2 / w, self.y2 / h)
        return ret

    def Clip(self, clipVal=1.0):
        ret = BoundingBox(self._clip(self.x1, clipVal), self._clip(self.y1, clipVal),
                          self._clip(self.x2, clipVal), self._clip(self.y2, clipVal))
        return ret

    def _clip(self, value, clipVal):
        return max(0, min(value, clipVal))


    def AddPadding(self, padX = 0.0, padY = 0.0):
        return BoundingBox(self.x1 + padX, self.y1 + padY,  self.x2 + padX, self.y2 + padY)

    def SubstractPadding(self,  padX = 0.0, padY = 0.0):
        return BoundingBox(self.x1 - padX, self.y1 - padY,  self.x2 - padX, self.y2 - padY)

    def ScaleBB(self, scaleX, scaleY):
        return BoundingBox(self.x1 * scaleX, self.y1 * scaleY, self.x2 * scaleX, self.y2 * scaleY)

    def ExtendBB(self, scale):
        return BoundingBox(self.x1 - (self.x1 / scale), self.y1 - (self.y1 / scale),
                           self.x2 + (self.x2 / scale), self.y2 + (self.y2 / scale))

    def CropImage(self, image):
        return np.array(image[int(self.y1):int(self.y2), int(self.x1):int(self.x2)])

    def Draw(self, image, description=None, color=list([0.0, 0.0, 0.0])):
        p1 = (int(self.x1), int(self.y1))
        p2 = (int(self.x2), int(self.y2))
        cv2.rectangle(image, p1, p2, color, thickness=2)

        if description is not None:
            cv2.putText(image, description, p1, fontScale=1, thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color)
        return image

    def AddPatch(self, plt, ax, description=None, color=list([0.0, 0.0, 0.0, 1.0])):
        # Create a Rectangle patch
        bbox = patches.Rectangle((self.x1, self.y1), self.width, self.height, linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')

        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        if description is not None:
            plt.text(self.x1, self.y1, s=description, color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

        return plt

    def Area(self):
        return self.height * self.width

    def CalculateOverlapp(self, target: "BoundingBox") -> float:
        x1 = max(self.x1, target.x1)
        y1 = max(self.y1, target.y1)
        x2 = min(self.x2, target.x2)
        y2 = min(self.y2, target.y2)

        interArea = (x1-x2)*(y1-y2)

        return interArea

