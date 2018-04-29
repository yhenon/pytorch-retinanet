import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import model
from anchors import Anchors
import losses
import pdb
import time
from dataloader import CocoDataset, collater, ToTensor, Resizer
from torch.utils.data import Dataset, DataLoader
import cv2 
assert torch.__version__.split('.')[1] == '4'
import requests
import coco_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))

model = model.resnet18(pretrained=True)

dataset_train = CocoDataset('../coco/', set_name='train2017', transform=transforms.Compose([Resizer(), ToTensor()]))
dataset_val = CocoDataset('../coco/', set_name='val2017', transform=transforms.Compose([Resizer(), ToTensor()]))

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4, collate_fn=collater)
dataloader_val   = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4, collate_fn=collater)

use_gpu = True

if use_gpu:
    model = model.cuda()

model.training = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)

total_loss = losses.loss

running_loss = 0.0

model.train()
model.freeze_bn()

for i in range(10):

	model.train()

	for idx, data in enumerate(dataloader_train):
		
		optimizer.zero_grad()


		classification, regression, anchors = model(data['img'].cuda().float())
		if data['annot'][0, 0, 4] == -1:
			continue

		classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'].cuda().float())

		loss = classification_loss + regression_loss

		loss.backward()

		torch.nn.utils.clip_grad_value_(model.parameters(), 0.001)

		running_loss = running_loss * 0.99 + 0.01 * loss

		optimizer.step()

		print(i, idx, classification_loss, regression_loss, running_loss)

		if idx > 10000:
			break

	print('Evaluating dataset')
	coco_eval.evaluate_coco(dataset_val, model)

#torch.save(model, 'model.pt')
#model.save_state_dict('mytraining.pt')

#model = torch.load('model.pt')
#model.load_state_dict('mytraining.pt')

model.eval()

for i in range(100):

	for idx, data in enumerate(dataloader):
	
		scores, classification, transformed_anchors = model(data['img'].cuda().float())

		idxs = np.where(scores>0.8)
		img = np.transpose(np.array(data['img'])[0, :, :, :], (1,2,0)).astype(np.uint8)
		
		print(idxs[0].shape[0])

		for j in range(idxs[0].shape[0]):
			bbox = transformed_anchors[idxs[0][j], :]
			x1 = int(bbox[0])
			y1 = int(bbox[1])
			x2 = int(bbox[2])
			y2 = int(bbox[3])

			print(classification[idxs[0][j]])

			cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
		cv2.imshow('img', img)
		cv2.waitKey(0)

pdb.set_trace()

# Observe that all parameters are being optimized

