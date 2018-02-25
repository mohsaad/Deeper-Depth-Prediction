# Mohammad Saad
# 2/24/2018
# predict.py
# Loads a model and outputs a depth map

import numpy as np
import random
import os
from PIL import Image
from scipy.ndimage import imread
from skimage import img_as_float


from model import *
from weights import *
from utils import *

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import sys

class DepthPrediction:

	def __init__(self, weight_file, batch_size):
		self.weight_file = weight_file

		self.model = Model(batch_size)
		self.dtype = torch.cuda.FloatTensor

		self.model.load_state_dict(load_weights(self.model, self.weight_file, self.dtype))

	def print_model(self):
		print(self.model)

	def predict(self, img):
		cropped_img = center_crop(img, 304, 228)
		cropped_img = np.reshape(cropped_img, [3, 304, 228])
		pytorch_img = torch.from_numpy(cropped_img).unsqueeze(0).float()
		print(type(pytorch_img))
		print(list(pytorch_img.size()))
		pytorch_input = Variable(pytorch_img)
		out_img = self.model(pytorch_input)
		print(type(out_img))
		#save_image(pytorch_img, "output_image.png")


if __name__ == '__main__':
	prediction = DepthPrediction('NYU_ResNet-UpProj.npy', 1)
	img = img_as_float(imread(sys.argv[1]))
	print img.dtype
	prediction.predict(img)
