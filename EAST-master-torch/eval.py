import time
import torch
import subprocess
import os
from model import EAST
from model2 import EASTER
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_path, test_img_path, submit_path='./submit', save_flag=True, set_scale=4, model='EAST', limit=False):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print ('Picked Device')
	print (device)
	if (model == 'EAST'):
		model = EAST(False).to(device)
	else:
		model = EASTER(False).to(device)
	scale = set_scale
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path, scale=scale, limit_images=limit)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)

	return res


if __name__ == '__main__': 
	#model_name = './pths/east_vgg16.pth'
	model_name  = './pths/EASTER-sm1-aug3-no_ignore-375.pth'
	test_img_path = os.path.abspath('/home/surajm72/data/ICDAR_2015/test_img')
	submit_path = './submit'
	scale = 2
	model = 'EASTER'
	eval_model(model_name, test_img_path, submit_path, set_scale=scale, model=model)
