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
import scipy.misc
import time


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
		print("Model on cuda? {0}".format(next(self.model.parameters()).is_cuda))

	def print_model(self):
		print(self.model)

	def predict(self, img):
		cropped_img = center_crop(img, 304, 228)
		scipy.misc.toimage(cropped_img, cmin = 0.0, cmax = 1.0).save('cropped_img.jpg')


		pytorch_img = torch.from_numpy(cropped_img).permute(2,0,1).unsqueeze(0).float()
		save_image(pytorch_img, "input_image.jpg")
		pytorch_input = Variable(pytorch_img)
		t = time.time()
		out_img = self.model(pytorch_input)
		save_image(out_img.data, "output_image.jpg", normalize = True)
		print("Finished image in {0} s".format(time.time() - t))

if __name__ == '__main__':
	prediction = DepthPrediction('NYU_ResNet-UpProj.npy', 1)
	img = img_as_float(imread(sys.argv[1]))
	prediction.predict(img)
