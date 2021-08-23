import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
from model2 import EASTER
import os
from dataset import get_rotate_mat
import numpy as np

test_images = ['test_image1', 'apple_tc_full1', 'adobe_tc_full2']

for t in test_images:
	print ('Testing')
	print (t)

	img_path = 'test_images/' + t + '.jpg'
	img = Image.open(img_path)
	w, h = img.size

	print ('Orig Width')
	print (w)
	print ('Orig Height')
	print (h)

	#try scaling by 2
	resize_w = w*2
	resize_h = h*2

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	print ('Resize Width')
	print (resize_w)
	print ('Resize Height')
	print (resize_h)

	res_img = 'r_images/' + t + '.jpg'
	img.save(res_img)

print ('Done')