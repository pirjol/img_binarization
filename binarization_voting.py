import sys
import os
import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im

from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from nick.nick import nick
from pythreshold.local_th.bernsen import bernsen_threshold

'''
Image processing voting algorithm
Algorithms used:
	- threshold otsu
	- niblack
	- sauvola
	- nick method
	- mahotas bernsen
	- isodata
	- li

'''

FG_OTSU_WEIGHT	 =	1
FG_NIBLACK_WEIGHT=	1
FG_SAUVOLA_WEIGHT=	1
FG_BERNSEN_WEIGHT=	1
FG_NICK_WEIGHT 	 =	1

BG_OTSU_WEIGHT	 = 1
BG_NIBLACK_WEIGHT= 1
BG_SAUVOLA_WEIGHT= 1
BG_BERNSEN_WEIGHT= 1
BG_NICK_WEIGHT   = 1

from numpy.fft import fft2, ifft2
from scipy import ndimage
from scipy.signal import gaussian
from scipy.signal.signaltools import wiener

def ratio(image):
	white_pixels = np.sum(image) / 255
	black_pixels = image.size - white_pixels
	ratio = black_pixels / white_pixels

	return ratio

def vote(image):
	ws = 25
	height, width = image.shape
	acc_matrix = np.zeros(shape = (height, width), dtype=np.uint8)

	binary_global 	= image > threshold_otsu(image)
	binary_niblack 	= image > threshold_niblack(image, window_size=ws, k=0.9)
	binary_sauvola 	= image > threshold_sauvola(image, window_size=ws)
	binary_bernsen 	= image > bernsen_threshold(image, c_thr=40)
	binary_nick 	= image < nick(image, window=(ws, ws), k=-0.2, padding='edge')
	for x in range(0, height):
		for y in range(0, width):
			white = 0
			black = 0
			if binary_nick[x,y] == False and binary_sauvola[x,y] == False:
				white = white + (BG_NICK_WEIGHT + BG_SAUVOLA_WEIGHT) * 2
			elif binary_nick[x,y] != False and binary_sauvola[x,y] != False:
				black = black + (FG_NICK_WEIGHT + FG_SAUVOLA_WEIGHT) * 2
			else:
				if binary_nick[x,y] == False:
					white = white + BG_NICK_WEIGHT
				else:
					black = black + FG_NICK_WEIGHT
				if binary_sauvola[x,y] == False:
					white = white + BG_SAUVOLA_WEIGHT
				else:
					black = black + FG_SAUVOLA_WEIGHT
			if binary_global[x,y] == False and binary_niblack[x,y] == False:
				white = white + (BG_OTSU_WEIGHT + BG_NIBLACK_WEIGHT) * 3
			else:
				if binary_global[x,y] == False:
					white = white + BG_OTSU_WEIGHT
				else:
					black = black + FG_OTSU_WEIGHT
				if binary_niblack[x,y] == False:
					white = white + BG_NIBLACK_WEIGHT
				else:
					black = black + FG_NIBLACK_WEIGHT
			if binary_bernsen[x,y] == False:
				white = white + BG_BERNSEN_WEIGHT
			else:
				black = black + FG_BERNSEN_WEIGHT
			if white < black:
				acc_matrix[x, y] = 255
			else:
				acc_matrix[x, y] = 0

	return acc_matrix

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def preprocess(image):
	kernel 	= np.ones((3,3),np.float32)/9
	image 	= ndimage.median_filter(image, size=5)
	image 	= cv2.medianBlur(image,3)
	# image 	= wiener_filter(image, 50, gaussian_kernel(7))

	return image

def binarize(filename):

	
	image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	original_image = copy.deepcopy(image) 

	## Preprocess
	image = preprocess(image)
	
	## Voting algorithm
	binary_image = vote(image)
	
	## Post-processing; to add

	plt.figure(figsize=(15, 14))
	plt.subplot(1, 2, 1)
	plt.imshow(original_image, cmap=plt.cm.gray)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.imshow(binary_image, cmap=plt.cm.gray)
	plt.title('Voting')
	plt.axis('off')

	plt.show()
	
	return binary_image


if __name__ == '__main__':
	binary_image = binarize(filename=sys.argv[1])
	img = im.fromarray(binary_image)
	img.save('result_' + os.path.splitext(sys.argv[1])[0] + '.png')