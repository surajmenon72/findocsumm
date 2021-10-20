from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import lanms

def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

#TODO: These three functions should be moved to some more abstract file, they generally belong in detect.py
def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img

def plot_boxes_labels(img, boxes, labels):

	if boxes is None:
		return img

	draw = ImageDraw.Draw(img)

	for i, box in enumerate(boxes):
		#if (labels[i] == 1):
		if 1:
			draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))

	return img
def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False

def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index

def get_boxes(score, geo, score_thresh=0, nms_thresh=0.2, scale=4):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape, scale=scale) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes

def cal_distance(x1, y1, x2, y2):
	'''calculate the Euclidean distance'''
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
	'''move the two points to shrink edge
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
		index1  : offset of point1
		index2  : offset of point2
		r       : [r1, r2, r3, r4] in paper
		coef    : shrink ratio in paper
	Output:
		vertices: vertices where one edge has been shinked
	'''
	index1 = index1 % 4
	index2 = index2 % 4
	x1_index = index1 * 2 + 0
	y1_index = index1 * 2 + 1
	x2_index = index2 * 2 + 0
	y2_index = index2 * 2 + 1
	
	r1 = r[index1]
	r2 = r[index2]
	length_x = vertices[x1_index] - vertices[x2_index]
	length_y = vertices[y1_index] - vertices[y2_index]
	length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
	if length > 1:	
		ratio = (r1 * coef) / length
		vertices[x1_index] += ratio * (-length_x) 
		vertices[y1_index] += ratio * (-length_y) 
		ratio = (r2 * coef) / length
		vertices[x2_index] += ratio * length_x 
		vertices[y2_index] += ratio * length_y
	return vertices	


def shrink_poly(vertices, coef=0.3):
	'''shrink the text region
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
		coef    : shrink ratio in paper
	Output:
		v       : vertices of shrinked text region <numpy.ndarray, (8,)>
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
	r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
	r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
	r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
	r = [r1, r2, r3, r4]

	# obtain offset to perform move_points() automatically
	if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
		offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
	else:
		offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

	v = vertices.copy()
	v = move_points(v, 0 + offset, 1 + offset, r, coef)
	v = move_points(v, 2 + offset, 3 + offset, r, coef)
	v = move_points(v, 1 + offset, 2 + offset, r, coef)
	v = move_points(v, 3 + offset, 4 + offset, r, coef)
	return v


def rotate_vertices(vertices, theta, anchor=None):
	'''rotate vertices around anchor
	Input:	
		vertices: vertices of text region <numpy.ndarray, (8,)>
		theta   : angle in radian measure
		anchor  : fixed position during rotation
	Output:
		rotated vertices <numpy.ndarray, (8,)>
	'''
	v = vertices.reshape((4,2)).T
	if anchor is None:
		anchor = v[:,:1]
	rotate_mat = get_rotate_mat(theta)
	res = np.dot(rotate_mat, v - anchor)
	return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
	'''get the tight boundary around given vertices
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		the boundary
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	x_min = min(x1, x2, x3, x4)
	x_max = max(x1, x2, x3, x4)
	y_min = min(y1, y2, y3, y4)
	y_max = max(y1, y2, y3, y4)
	return x_min, x_max, y_min, y_max


def cal_error(vertices):
	'''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
	calculate the difference between the vertices orientation and default orientation
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		err     : difference measure
	'''
	x_min, x_max, y_min, y_max = get_boundary(vertices)
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
	return err	


def find_min_rect_angle(vertices):
	'''find the best angle to rotate poly and obtain min rectangle
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		the best angle <radian measure>
	'''
	angle_interval = 1
	angle_list = list(range(-90, 90, angle_interval))
	area_list = []
	for theta in angle_list: 
		rotated = rotate_vertices(vertices, theta / 180 * math.pi)
		x1, y1, x2, y2, x3, y3, x4, y4 = rotated
		temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
		area_list.append(temp_area)
	
	sorted_area_index = sorted(list(range(len(area_list))), key=lambda k : area_list[k])
	min_error = float('inf')
	best_index = -1
	rank_num = 10
	# find the best angle with correct orientation
	for index in sorted_area_index[:rank_num]:
		rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
		temp_error = cal_error(rotated)
		if temp_error < min_error:
			min_error = temp_error
			best_index = index
	return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
	'''check if the crop image crosses text regions
	Input:
		start_loc: left-top position
		length   : length of crop image
		vertices : vertices of text regions <numpy.ndarray, (n,8)>
	Output:
		True if crop image crosses text region
	'''
	if vertices.size == 0:
		return False
	start_w, start_h = start_loc
	a = np.array([start_w, start_h, start_w + length, start_h, \
          start_w + length, start_h + length, start_w, start_h + length]).reshape((4,2))
	p1 = Polygon(a).convex_hull
	for vertice in vertices:
		p2 = Polygon(vertice.reshape((4,2))).convex_hull
		inter = p1.intersection(p2).area
		if 0.01 <= inter / p2.area <= 0.99: 
			return True
	return False
		

def crop_img(img, vertices, labels, length):
	'''crop img patches to obtain batch and augment
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		length      : length of cropped image region
	Output:
		region      : cropped image region
		new_vertices: new vertices in cropped region
	'''
	h, w = img.height, img.width
	# confirm the shortest side of image >= length
	if h >= w and w < length:
		img = img.resize((length, int(h * length / w)), Image.BILINEAR)
	elif h < w and h < length:
		img = img.resize((int(w * length / h), length), Image.BILINEAR)
	ratio_w = img.width / w
	ratio_h = img.height / h
	assert(ratio_w >= 1 and ratio_h >= 1)

	new_vertices = np.zeros(vertices.shape)
	if vertices.size > 0:
		new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

	# find random position
	remain_h = img.height - length
	remain_w = img.width - length
	flag = True
	cnt = 0
	while flag and cnt < 1000:
		cnt += 1
		start_w = int(np.random.rand() * remain_w)
		start_h = int(np.random.rand() * remain_h)
		flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
	box = (start_w, start_h, start_w + length, start_h + length)
	region = img.crop(box)
	if new_vertices.size == 0:
		return region, new_vertices	
	
	new_vertices[:,[0,2,4,6]] -= start_w
	new_vertices[:,[1,3,5,7]] -= start_h
	return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
	'''get rotated locations of all pixels for next stages
	Input:
		rotate_mat: rotatation matrix
		anchor_x  : fixed x position
		anchor_y  : fixed y position
		length    : length of image
	Output:
		rotated_x : rotated x positions <numpy.ndarray, (length,length)>
		rotated_y : rotated y positions <numpy.ndarray, (length,length)>
	'''
	x = np.arange(length)
	y = np.arange(length)
	x, y = np.meshgrid(x, y)
	x_lin = x.reshape((1, x.size))
	y_lin = y.reshape((1, x.size))
	coord_mat = np.concatenate((x_lin, y_lin), 0)
	rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
	rotated_x = rotated_coord[0, :].reshape(x.shape)
	rotated_y = rotated_coord[1, :].reshape(y.shape)
	return rotated_x, rotated_y

def resize_with_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    # target_ratio = target_height / target_width
    # im_ratio = im.height / im.width
    # if target_ratio > im_ratio:
    #     # It must be fixed by width
    #     resize_width = target_width
    #     resize_height = round(resize_width * im_ratio)
    # else:
    #     # Fixed by height
    #     resize_height = target_height
    #     resize_width = round(resize_height / im_ratio)

    resize_width = im.width
    resize_height = im.height

    #image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    image_resize = im

    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    x_offset = round((target_width - resize_width) / 2)
    y_offset = round((target_height - resize_height) / 2)

    offset = (x_offset, y_offset)
    background.paste(image_resize, offset)
    return background.convert('RGB'), (x_offset, y_offset)

def adjust_height(img, vertices, ratio=0.2):
	'''adjust height of image to aug data
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		ratio       : height changes in [0.8, 1.2]
	Output:
		img         : adjusted PIL Image
		new_vertices: adjusted vertices
	'''
	ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
	old_h = img.height
	new_h = int(np.around(old_h * ratio_h))
	img = img.resize((img.width, new_h), Image.BILINEAR)
	new_vertices = vertices.copy()
	if vertices.size > 0:
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
	return img, new_vertices

def scale_img(img, vertices, low=0.2, high=1.0, scale_prob=1):
	'''adjust scale of image to aug data
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		ratio       : height/width changes in [0.5, 1.5]
	Output:
		img         : adjusted PIL Image
		new_vertices: adjusted vertices
	'''

	#ratio_hw = 1 + ratio * (np.random.rand() * 2 - 1)
	do_scale = False

	num =  np.random.uniform(low=0, high=1)
	if (num < scale_prob):
		do_scale = True
	else:
		do_scale = False

	if (do_scale == True):
		ratio_hw = np.random.uniform(low=low, high=high)
		old_h = img.height
		old_w = img.width
		new_h = int(np.around(old_h * ratio_hw))
		new_w = int(np.around(old_w * ratio_hw))
		img = img.resize((new_w, new_h), Image.BILINEAR)

		new_vertices = vertices.copy()
		if (vertices.size) > 0:
			new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
			new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * (new_w / old_w)

		scaled_image, offsets = resize_with_pad(img, 512, 512)

		if (vertices.size) > 0:
			new_vertices[:,[1,3,5,7]] = new_vertices[:,[1,3,5,7]] + offsets[1]
			new_vertices[:,[0,2,4,6]] = new_vertices[:,[0,2,4,6]] + offsets[0]
	else:
		scaled_image, new_vertices = adjust_height(img, vertices)

	return scaled_image, new_vertices

def full_scale_img(img, vertices, full_scale_factor=1):
	scale_factor = full_scale_factor

	old_h = img.height
	old_w = img.width
	new_h = int(np.around(old_h * scale_factor))
	new_w = int(np.around(old_w * scale_factor))
	resize_h = new_h if new_h % 32 == 0 else int(new_h / 32) * 32
	resize_w = new_w if new_w % 32 == 0 else int(new_w / 32) * 32

	scaled_img = img.resize((resize_w, resize_h), Image.BILINEAR)

	new_vertices = vertices.copy()
	if (vertices.size) > 0:
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (resize_h / old_h)
		new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * (resize_w / old_w)

	return scaled_img, new_vertices

def rotate_img(img, vertices, angle_range=10):
	'''rotate image [-10, 10] degree to aug data
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		angle_range : rotate range
	Output:
		img         : rotated PIL Image
		new_vertices: rotated vertices
	'''
	center_x = (img.width - 1) / 2
	center_y = (img.height - 1) / 2
	angle = angle_range * (np.random.rand() * 2 - 1)
	img = img.rotate(angle, Image.BILINEAR)
	new_vertices = np.zeros(vertices.shape)
	for i, vertice in enumerate(vertices):
		new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	return img, new_vertices


def get_score_geo(img, vertices, labels, scale, length, ignore=True):
	'''generate score gt and geometry gt
	Input:
		img     : PIL Image
		vertices: vertices of text regions <numpy.ndarray, (n,8)>
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		scale   : feature map / image
		length  : image length
	Output:
		score gt, geo gt, ignored
	'''
	score_map   = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
	geo_map     = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
	ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
	
	index = np.arange(0, length, int(1/scale))
	index_x, index_y = np.meshgrid(index, index)
	ignored_polys = []
	polys = []
	
	for i, vertice in enumerate(vertices):
		if (ignore == True):
			if labels[i] == 0:
				ignored_polys.append(np.around(scale * vertice.reshape((4,2))).astype(np.int32))
				continue		
		
		poly = np.around(scale * shrink_poly(vertice).reshape((4,2))).astype(np.int32) # scaled & shrinked
		polys.append(poly)
		temp_mask = np.zeros(score_map.shape[:-1], np.float32)
		cv2.fillPoly(temp_mask, [poly], 1)
		
		theta = find_min_rect_angle(vertice)
		rotate_mat = get_rotate_mat(theta)
		
		rotated_vertices = rotate_vertices(vertice, theta)
		x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
		rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
	
		d1 = rotated_y - y_min
		d1[d1<0] = 0
		d2 = y_max - rotated_y
		d2[d2<0] = 0
		d3 = rotated_x - x_min
		d3[d3<0] = 0
		d4 = x_max - rotated_x
		d4[d4<0] = 0
		geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
		geo_map[:,:,1] += d2[index_y, index_x] * temp_mask
		geo_map[:,:,2] += d3[index_y, index_x] * temp_mask
		geo_map[:,:,3] += d4[index_y, index_x] * temp_mask
		geo_map[:,:,4] += theta * temp_mask
	
	cv2.fillPoly(ignored_map, ignored_polys, 1)
	cv2.fillPoly(score_map, polys, 1)
	return torch.Tensor(score_map).permute(2,0,1), torch.Tensor(geo_map).permute(2,0,1), torch.Tensor(ignored_map).permute(2,0,1)


def extract_vertices(lines):
	'''extract vertices info from txt lines
	Input:
		lines   : list of string info
	Output:
		vertices: vertices of text regions <numpy.ndarray, (n,8)>
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
	'''
	labels = []
	vertices = []
	for line in lines:
		vertices.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
		label = 0 if '###' in line else 1
		labels.append(label)
	return np.array(vertices), np.array(labels)

	
class custom_dataset(data.Dataset):
	def __init__(self, img_path, gt_path, scale=0.25, length=512, scale_aug=False, ignore=True, full_scale=False, batch_size=16):
		super(custom_dataset, self).__init__()
		self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
		self.gt_files  = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
		self.scale = scale
		self.length = length
		self.scale_aug = scale_aug
		self.scale_len = int(self.length*self.scale)
		self.ignore = ignore
		self.full_scale = full_scale
		self.batch_size = batch_size
		self.full_scale_factor = 1
		self.full_scale_count = 0

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, index):
		with open(self.gt_files[index], 'r') as f:
			lines = f.readlines()
		vertices, labels = extract_vertices(lines)
		
		img = Image.open(self.img_files[index])

		#check that things match
		img_addr = self.img_files[index]
		gt_addr = self.gt_files[index]

		img_split = img_addr.split('_')
		img_split2 = img_split[-1].split('.')
		img_num = img_split2[0]

		gt_split = gt_addr.split('_')
		gt_split2 = gt_split[-1].split('.')
		gt_num = gt_split2[0]

		if (img_num != gt_num):
			print ('ERROR: IMG AND GT NOT MATCHING!!')

		# res_img = plot_boxes_labels(img, vertices, labels)
		# res_img.save('./pre_test.bmp')

		if (self.full_scale == True):
			if (self.full_scale_count == 0):
				random_num = np.random.uniform(low=0, high=1.0)
				self.full_scale_factor = 0.5 + random_num
				#self.full_scale_factor = 1

			img, vertices = full_scale_img(img, vertices, self.full_scale_factor)
			img, vertices = rotate_img(img, vertices)

			self.length = int(512*self.full_scale_factor)
			self.length = self.length if self.length % 32 == 0 else int(self.length / 32) * 32

			h = img.height
			w = img.width

			if (h > w):
				long_side = h
			else:
				long_side = w

			if (self.length > long_side):
				self.length = long_side

			# print (self.full_scale_factor)
			# print (self.length)

			img, vertices = crop_img(img, vertices, labels, self.length)

			if (self.full_scale_count == self.batch_size-1):
				self.full_scale_count = 0
			else:
				self.full_scale_count += 1
		else:
			if (self.scale_aug == True):
				img, vertices = scale_img(img, vertices)
			else:
				img, vertices = adjust_height(img, vertices) 

			img, vertices = rotate_img(img, vertices)
			img, vertices = crop_img(img, vertices, labels, self.length)

		# res_img = plot_boxes(img, vertices)
		# res_img.save('./pre_test2.bmp')
		# exit()

		transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
		
		score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length, ignore=self.ignore)
		#score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length, ignore=False)

		# r_score_map = torch.reshape(score_map, (256, 256))
		# score_sum = torch.sum(torch.sum(score_map, axis=1))
		# print ('Score sum')
		# print (score_sum)

		# score_map_r = torch.reshape(score_map, (1, 1, self.scale_len, self.scale_len))
		# geo_map_r = torch.reshape(geo_map, (1, 5, self.scale_len, self.scale_len))

		# boxes = get_boxes(score_map_r.squeeze(0).cpu().numpy(), geo_map_r.squeeze(0).cpu().numpy(), scale=int(1/self.scale))
	
		# res_img = plot_boxes(img, boxes)
		# res_img.save('./scale_test.bmp')

		return transform(img), score_map, geo_map, ignored_map

