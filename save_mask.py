import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet
from resnet import ResNet18
from wideresnet import WideResNet
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

from dropblock.dropblock import DropBlock2D
from dropblock.scheduler import LinearScheduler


from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import json

from cutout import Cutout
from PIL import Image
import os
import cv2

def pil_loader(img_str):
    
    with Image.open(img_str) as img:
        img = img.convert('RGB')
    return img

model_name = os.sys.argv[1]
file_listn = os.sys.argv[2]

with open(file_listn) as fr:
	file_list = fr.readlines()

normalize = transforms.Normalize(mean=[x / 255.0 for x in [123.675, 116.28, 103.53]],
                                     std=[x / 255.0 for x in [58.395, 57.12, 57.375]])
test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define model
model = ResNet18(num_classes=num_classes, drop_prob=0.1)
model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage.cuda(0)))
model = model.cuda()
model.train()

masks = []

for fn in file_list:
	img = pil_loader(fn.strip())
	w, h = img.size
	img = img.resize((32, 32), Image.ANTIALIAS)
	img_var = test_transform(img).unsqueeze(0).cuda()
	# img_var = img_var.cuda()
	out, blk1, blk2 = model.get_mask(img_var, 1.0)
	for i in range(blk2.shape[1]):
		heatmap = cv2.applyColorMap(cv2.resize(blk2[0,i].cpu().data.numpy() * 255.0, (w, h)).astype(np.uint8), cv2.COLORMAP_JET)
		# masks.append(blk1.data.numpy().tolist())
		masks.append(heatmap.tolist())

with open('masks_out.json', 'w') as fw:
	fw.write(json.dumps(masks))
