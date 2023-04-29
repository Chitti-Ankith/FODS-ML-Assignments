import numpy as np
import sys
import datetime
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

	
	
	
path = "D:\\Studies\\ML\\ML - Assignment 3\\dogs-vs-cats-redux-kernels-edition\\test\\test\\"
files = os.listdir(path)

filenames = [path + i for i in files]
print(filenames)
images = []
for filename in filenames:
	img = mpimg.imread(filename)     
	gray = rgb2gray(img)    
	plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
	plt.show()
