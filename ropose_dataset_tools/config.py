import os

#path stuff
homedir = os.environ['HOME']
basePath = os.path.realpath(__file__ + "/../")

cocoPath = cocoPath = homedir + "/coco/"

augmentationCval = 0.4980392156862745

# RoposePose model stuff
linkOrder = ["base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link"]

coco_classes = {0: {'supercategory': 'unknown', 'id': 0, 'name': 'unknown'},
                1: {'supercategory': 'person', 'id': 1, 'name': 'person'},
                2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
                6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
                7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
                8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
                9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
                10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
                11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
                12: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
                13: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
                14: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
                15: {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
                16: {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
                17: {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
                18: {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
                19: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
                20: {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
                21: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
                22: {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
                23: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
                24: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
                25: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
                26: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
                27: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
                28: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
                29: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
                30: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
                31: {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
                32: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
                33: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
                34: {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
                35: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
                36: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
                37: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
                38: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
                39: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
                40: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                41: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
                42: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
                43: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
                44: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
                45: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
                46: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
                47: {'supercategory': 'food', 'id': 52, 'name': 'banana'},
                48: {'supercategory': 'food', 'id': 53, 'name': 'apple'},
                49: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
                50: {'supercategory': 'food', 'id': 55, 'name': 'orange'},
                51: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
                52: {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
                53: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
                54: {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
                55: {'supercategory': 'food', 'id': 60, 'name': 'donut'},
                56: {'supercategory': 'food', 'id': 61, 'name': 'cake'},
                57: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                58: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
                59: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                60: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
                61: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
                62: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
                63: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
                64: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
                65: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
                66: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
                67: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
                68: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
                69: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
                70: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
                71: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
                72: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
                73: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
                74: {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
                75: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
                76: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
                77: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
                78: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
                79: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
                80: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'},
                81: {'supercategory': 'robot', 'id': 100, 'name': 'ropose robot'}}

yolo_cocoClassMap = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 8,
                9: 9,
                10: 10,
                11: 11,
                13: 12,
                14: 13,
                15: 14,
                16: 15,
                17: 16,
                18: 17,
                19: 18,
                20: 19,
                21: 20,
                22: 21,
                23: 22,
                24: 23,
                25: 24,
                27: 25,
                28: 26,
                31: 27,
                32: 28,
                33: 29,
                34: 30,
                35: 31,
                36: 32,
                37: 33,
                38: 34,
                39: 35,
                40: 36,
                41: 37,
                42: 38,
                43: 39,
                44: 40,
                46: 41,
                47: 42,
                48: 43,
                49: 44,
                50: 45,
                51: 46,
                52: 47,
                53: 48,
                54: 49,
                55: 50,
                56: 51,
                57: 52,
                58: 53,
                59: 54,
                60: 55,
                61: 56,
                62: 57,
                63: 58,
                64: 59,
                65: 60,
                67: 61,
                70: 62,
                72: 63,
                73: 64,
                74: 65,
                75: 66,
                76: 67,
                77: 68,
                78: 69,
                79: 70,
                80: 71,
                81: 72,
                82: 73,
                84: 74,
                85: 75,
                86: 76,
                87: 77,
                88: 78,
                89: 79,
                90: 80,
                100: 81,
}

yolo_Classes = coco_classes
yolo_HumanClassNum = 1
yolo_RoposeClassNum = 81