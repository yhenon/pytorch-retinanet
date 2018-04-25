import numpy as np
import torch
import torch.nn as nn

def loss(classification, regression, anchors):
	import pdb
	pdb.set_trace()
	#return torch.norm(true_x[0, :, :4]/100000.-pred_x[0, :, :4]/100000.)
	return torch.norm(true_x[0, :, :4]/100000.- 0.1 * pred_x[0, :, :4]/100000.)
