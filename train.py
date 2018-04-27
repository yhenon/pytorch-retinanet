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
print('CUDA available: {}'.format(torch.cuda.is_available()))

model = model.resnet18(pretrained=True)

dataset = CocoDataset('../data/', transform=transforms.Compose([Resizer(), ToTensor()]))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collater)

'''
img = np.zeros((2, 3, 256, 256)).astype(np.float32)

img[0, 0, 30:130, 30:130] = 1
img[0, 1, 130:190, 130:150] = 1

img[1, 0, 30:230, 30:230] = 1

annotations = np.zeros((2, 3, 5)).astype(np.float32)

annotations[0, 0, :] = [0, 30, 30, 130, 130]
annotations[0, 1, :] = [11, 130, 130, 190, 150]

annotations[1, 0, :] = [1, 30, 30, 230, 230]

X = torch.from_numpy(img).cuda()
annotations = torch.from_numpy(annotations).cuda()
print(type(X))
'''


use_gpu = True

if use_gpu:
    model = model.cuda()

model.training = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# regression_loss = losses.regressionLoss()
total_loss = losses.loss

running_loss = 0.0

for i in range(100):

	for idx, data in enumerate(dataloader):
		
		optimizer.zero_grad()


		classification, regression, anchors, transformed_anchors = model(data['img'].cuda().float())
		if data['annot'][0, 0, 4] == -1:
			continue

		classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'].cuda().float())

		loss = classification_loss + regression_loss

		#pdb.set_trace()

		loss.backward()

		torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

		running_loss = running_loss * 0.99 + 0.01 * loss
		optimizer.step()

		print(idx, classification_loss, regression_loss, running_loss)


#pdb.set_trace()

for i in range(100):

	for idx, data in enumerate(dataloader):
	
		classification, regression, anchors, transformed_anchors = model(data['img'].cuda().float())
		
		idxs = np.where(classification>0.5)
		img = np.transpose(np.array(data['img'])[0, :, :, :], (1,2,0)).astype(np.uint8)
		print(idxs[0].shape[0])
		for j in range(idxs[0].shape[0]):
			bbox = transformed_anchors[0, idxs[1][j], :]
			x1 = int(bbox[0])
			y1 = int(bbox[1])
			x2 = int(bbox[2])
			y2 = int(bbox[3])

			cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
		cv2.imshow('img', img)
		cv2.waitKey(0)

pdb.set_trace()

# Observe that all parameters are being optimized

