import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
from model2 import EASTER
from model3 import EAST_STRETCH
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
from loss import Loss


def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size

	#rescale factor
	#w_scale = 2.75
	#h_scale = 2.75

	#w_scale = 1.6
	#h_scale = 1.6

	#w_scale = 1.875
	#h_scale = 1.875

	w_scale = 1.5
	h_scale = 1.5

	#w_scale = 1.25
	#h_scale = 1.25

	resize_w = int(w * w_scale)
	resize_h = int(h * h_scale)

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


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


def get_boxes(score, geo, score_thresh=0.99, nms_thresh=0.2, scale=4):
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

# def get_boxes_from_geo(geo):
# 	img.save("img_test_before.jpeg")
# 	test_img = transform(img)
# 	test_img = test_img.cpu().detach().numpy()
# 	img_test = test_img.reshape(512, 512, 3)
# 	img_test = img_test * 255
# 	img_test = img_test.astype(np.uint8)
# 	im = Image.fromarray(img_test)
# 	im.save("img_test_after.jpeg")

def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device, scale=4):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
		#score, geo, _ = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy(), scale=scale)
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def detect_dataset(model, device, test_img_path, submit_path, scale=4, limit_images=False):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	images_to_check = 100

	for i, img_file in enumerate(img_files):
		if (limit_images == True):
			if (i == images_to_check):
				break
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect(Image.open(img_file), model, device, scale=scale)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)

def do_detection(img_path, model_path, res_img, scale=4, model='EAST'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#model = EAST_STRETCH(False).to(device)
	if (model == 'EAST'):
		model = EAST(False).to(device)
	else:
		model = EASTER(False).to(device)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()
	img = Image.open(img_path)
	
	boxes = detect(img, model, device, scale=scale)
	plot_img = plot_boxes(img, boxes)	
	plot_img.save(res_img)

test_images = ['test_img2', 'apple_tc_full1', 'adobe_tc_full2', 'mcds_tc_full1', 'cat_tc_full2']
#test_images = ['test_img2', 'mcds_tc_full1']
#test_images = ['mcds_tc_full1']

if __name__ == '__main__':
	#model_path = './pths/east_vgg16.pth'
	#model_path = './pths/EAST-aug3-100.pth'
	model_path  = './pths2/EAST-llr-fs-0.5-1.5-490.pth'
	model = 'EAST'
	scale = 4
	for t in test_images:
		img_path = 'test_img/' + t + '.jpg'
		segs = t.split('_')
		company = segs[0]
		res_img = './' + company + '.bmp'

		do_detection(img_path, model_path, res_img, scale=scale, model=model)
		print ('Done with:')
		print (t)

	print ('Done')


