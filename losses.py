import numpy as np
import torch
import torch.nn as nn

def loss(classification, regression, anchors):

	return torch.norm(classification[0, :, :4]/100000.- 0.1 * regression[0, :, :4]/100000.)
