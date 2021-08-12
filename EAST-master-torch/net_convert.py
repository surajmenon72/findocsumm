import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
from model import EAST
 
onnx_model_path = "./pths/sm1-60-onnx"
 
# https://pytorch.org/hub/pytorch_vision_densenet/
#model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

model_name = './pths/sm1-60.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EAST(False).to(device)
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

# set the model to inference mode
model.eval()
 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(12, 3, 512, 512, device='cpu')
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

print ('Done')

