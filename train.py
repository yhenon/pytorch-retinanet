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
from dataloader import CocoDataset, collater, Resizer, Normalizer, UnNormalizer, AspectRatioBasedSampler
from torch.utils.data import Dataset, DataLoader
import cv2 
assert torch.__version__.split('.')[1] == '4'
import requests
import coco_eval
import collections

print('CUDA available: {}'.format(torch.cuda.is_available()))

model = model.resnet50(pretrained=True)


dataset_train = CocoDataset('../coco/', set_name='train2017', transform=transforms.Compose([Resizer(), Normalizer()]))
dataset_val = CocoDataset('../coco/', set_name='val2017', transform=transforms.Compose([Resizer(), Normalizer()]))

sampler = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)

dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
# dataloader_val   = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler)

use_gpu = True

if use_gpu:
	model = model.cuda()

model.training = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)

total_loss = losses.loss

loss_hist = collections.deque(maxlen=500)

model.train()
model.freeze_bn()

for i in range(1000):

	model.train()
	model.freeze_bn()
	
	for idx, data in enumerate(dataloader_train):
	
		optimizer.zero_grad()

		classification, regression, anchors = model(data['img'].cuda().float())
		
		classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

		loss = classification_loss + regression_loss

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

		optimizer.step()

		loss_hist.append(float(loss))

		print(i, idx, float(classification_loss), float(regression_loss), np.mean(loss_hist))

	print('Evaluating dataset')
	coco_eval.evaluate_coco(dataset_val, model)

	torch.save(model, 'model_{}.pt'.format(i))

#model = torch.load('model.pt')
#model.load_state_dict('mytraining.pt')

model.eval()

unnormalize = UnNormalizer()


def draw_caption(image, box, caption):
	""" Draws a caption above the box in an image.
	# Arguments
		image   : The image to draw on.
		box     : A list of 4 elements (x1, y1, x2, y2).
		caption : String containing the text to draw.
	"""
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

for idx, data in enumerate(dataloader_val):

	scores, classification, transformed_anchors = model(data['img'].cuda().float())

	idxs = np.where(scores>0.5)
	img = np.array(unnormalize(data['img']))[0, :, :, :]

	img[img<0] = 0
	img[img>255] = 255

	img = np.transpose(img, (1,2,0)).astype(np.uint8)
	print(idxs[0].shape[0])
	print(scores.max())

	for j in range(idxs[0].shape[0]):
		bbox = transformed_anchors[idxs[0][j], :]
		x1 = int(bbox[0])
		y1 = int(bbox[1])
		x2 = int(bbox[2])
		y2 = int(bbox[3])
		label_name = dataset_val.labels[int(classification[idxs[0][j]])]
		draw_caption(img, (x1, y1, x2, y2), label_name)

		cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
		print(label_name)

	cv2.imshow('img', img)
	cv2.waitKey(0)

	pdb.set_trace()