import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import model
from anchors import Anchors
import losses
import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

X = Variable(torch.cuda.FloatTensor(np.random.rand(4,3,256,256)))

model = model.resnet18(pretrained=False)
anchor_gen = Anchors()

use_gpu = True

if use_gpu:
    model = model.cuda()

S = model(X)
'''
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
# regression_loss = losses.regressionLoss()
loss = losses.loss

for i in range(100):
	S = model(X)

	classification, regression, anchors = loss(S[0], S[1], S[2])
	loss.backward()
	optimizer.step()
	print(loss)
'''
pdb.set_trace()

# Observe that all parameters are being optimized

