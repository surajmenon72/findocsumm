import torch
import torch.onnx
from torch.utils import data
import torchvision
import torchvision.models as models
from dataset import custom_dataset
import sys
import os
from model import EAST
from matplotlib import pyplot as plt

test_img_path = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/test_img')
test_gt_path  = os.path.abspath('/Users/surajmenon/Desktop/findocsumm/data/ICDAR_2015/test_gt')

file_num = len(os.listdir(test_img_path))
testset = custom_dataset(test_img_path, test_gt_path)
test_loader = data.DataLoader(testset, batch_size=1, \
                               shuffle=True, num_workers=0, drop_last=True)

model_name = './pths/sm1-60.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EAST(False).to(device)
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

# set the model to inference mode
model.eval()

for i, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
	img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
	test_img = img[0]
	test_img = test_img.reshape(512, 512, 3)
	pred_score, pred_geo = model(img)

	# plt.imshow(test_img)
	# plt.title('Image')
	# plt.show()

	print (pred_score.shape)
	print (pred_geo.shape)
	pred_score = pred_score[0].reshape(128, 128, 1)
	pred_geo = pred_geo[0].reshape(128, 128, 5)
	print (pred_score[0, 1])
	print (pred_geo[0, 1])
	exit()

