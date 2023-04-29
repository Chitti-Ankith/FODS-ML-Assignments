import numpy as np
import sys
import datetime
import imageio
import os
from PIL import Image


path = "D:\\Studies\\ML\\ML - Assignment 3\\dogs-vs-cats-redux-kernels-edition\\train\\train\\"
files = os.listdir(path)

filenames = [path + i for i in files]

images = []
for filename in filenames:
	img = Image.open(filename).convert('L')
	#img.ioff()
	img.save(filename,bbox_inches='tight')
