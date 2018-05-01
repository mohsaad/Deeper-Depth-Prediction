# Mohammad Saad
# 2/19/2018
# preprocess.py
# Preprocesses the dataset into train and test images

import os
import h5py
import numpy as np
from PIL import Image
import random
import collections
import sys


class DataProcess:

    def __init__(self, filepath, outFilePath):
        self.filename = filepath
        self.file_handle = h5py.File(filepath, 'r')
        self.outFilePath = outFilePath
        self.rgb_train = ""
        self.rgb_test = ""
        self.depth_train = ""
        self.depth_test = ""

        print self.file_handle.keys()

    def make_folders(self):
        if not os.path.exists(self.outFilePath):
            os.makedirs(self.outFilePath)


        if not os.path.exists(self.outFilePath + "/rgb/"):
            os.makedirs(self.outFilePath + "/rgb/")

        if not os.path.exists(self.outFilePath + "/depth/"):
            os.makedirs(self.outFilePath + "/depth/")



    def separate_images(self):
        for i, (image, depth) in enumerate(zip(self.file_handle["images"], self.file_handle["depths"])):
            image_tp = image.transpose(2, 1, 0)
            depth_tp = depth.transpose(1, 0)

            # normalize depth
            depth_norm = (depth_tp - np.min(depth_tp)) * 255.0 / (np.max(depth_tp) - np.min(depth_tp))

            image_PIL = Image.fromarray(np.uint8(image_tp))
            depth_PIL = Image.fromarray(np.uint8(depth_norm))

            image_file = os.path.join(self.outFilePath, "rgb", "%05d.png" % i)
            depth_file = os.path.join(self.outFilePath, "depth", "%05d.png" % i)

            image_PIL.save(image_file)
            depth_PIL.save(depth_file)


    def process(self):
        self.make_folders()
        self.separate_images()


if __name__ == '__main__':
    d = DataProcess(sys.argv[1], sys.argv[2])
    d.process()
