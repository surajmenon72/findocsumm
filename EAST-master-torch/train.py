import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from model2 import EASTER
from model3 import EAST_STRETCH
from loss import Loss
import os
import time
import numpy as np
from PIL import Image, ImageDraw


def train(train_img_path, train_gt_path, test_img_path, test_gt_path, pths_path, batch_size, test_batch_size, lr, num_workers, epoch_iter, interval, eval_interval):
	file_num = len(os.listdir(train_img_path))
	
	#trainset = custom_dataset(train_img_path, train_gt_path, scale=0.25, scale_aug=True)
	trainset = custom_dataset(train_img_path, train_gt_path, scale=0.5, scale_aug=True)

	#testset = custom_dataset(test_img_path, test_gt_path, scale=0.25, scale_aug=True)
	testset = custom_dataset(test_img_path, test_gt_path, scale=0.5, scale_aug=True)

	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

	test_loader = data.DataLoader(testset, batch_size=test_batch_size, \
                               	   shuffle=True, num_workers=num_workers, drop_last=True)
	

	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	print ('Picked Device')
	print (device)
	torch.cuda.empty_cache()
	print ('Emptied Cache')
	#model = EAST()
	model = EASTER()
	#model = EAST_STRETCH()
	#model_name = './pths/east_vgg16.pth'
	model_name = './pths/EASTER-sm2-415.pth'
	model.load_state_dict(torch.load(model_name))
	epoch_start = 0
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=.1)

	use_scheduler = True
	do_eval = True

	eval_epochs = []
	eval_losses = []
	eval_vars = []

	if (use_scheduler == True):
		print ('Catching up Scheduler')
		for epoch in range(epoch_start):
			print (epoch)
			model.train()
			scheduler.step()

	print ('Starting Training')
	for epoch in range(epoch_start, epoch_iter):
		#EVAL code
		if (do_eval == True):
			if (epoch + 1) % eval_interval == 0:
				print ('Doing Eval')
				model.eval()
				full_test_loss = 0.0
				full_test_var = 0.0

				torch.cuda.empty_cache()
				for k, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
					img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
					with torch.no_grad():
						#pred_score, pred_geo = model(img)
						pred_score, pred_geo, feat_var = model(img)
						test_loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
						full_test_loss += test_loss.item()
						full_test_var += feat_var
					torch.cuda.empty_cache()
					avg_test_var = full_test_var/(k+1)


				eval_epochs.append(epoch)
				eval_losses.append(full_test_loss)
				eval_vars.append(avg_test_var)

				print ('EVAL: TEST LOSS: {:.8f}'.format(full_test_loss))
				print ('EVAL: TEST VAR: {:.8f}'.format(avg_test_var))
		exit()	
		#TRAIN code	
		model.train()
		if (use_scheduler == True):
			scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
	train_img_path = os.path.abspath('/home/surajm72/data/ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('/home/surajm72/data/ICDAR_2015/train_gt')
	test_img_path = os.path.abspath('/home/surajm72/data/ICDAR_2015/test_img')
	test_gt_path  = os.path.abspath('/home/surajm72/data/ICDAR_2015/test_gt')
	# train_img_path = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/train_img')
	# train_gt_path  = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/train_gt')
	# test_img_path = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/test_img')
	# test_gt_path  = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/test_gt')
	pths_path      = './pths'
	#batch_size     = 24
	train_batch_size = 16
	test_batch_size = 16
	lr             = 1e-3
	num_workers    = 0
	epoch_iter     = 900
	save_interval  = 5
	eval_interval  = 1
	train(train_img_path, train_gt_path, test_img_path, test_gt_path, pths_path, train_batch_size, test_batch_size, lr, num_workers, epoch_iter, save_interval, eval_interval)	
	
