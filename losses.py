import numpy as np
import torch
import torch.nn as nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def loss(classifications, regression, anchors, annotations):
    alpha = 0.25
    gamma = 2.0
    batch_size = classifications.shape[0]

    losses = []

    for j in range(batch_size):

        classification = classifications[j, :, :]
        annotation = annotations[j, :, :]

        classification = torch.clamp(classification, 1e-6, 1.0 - 1e-6)

        IoU = calc_iou(anchors[0, :, :], annotation[:, :4]) # num_anchors x num_annotations

        IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

        # compute the loss for classification
        targets = torch.ones(classification.shape) * -1
        targets = targets.cuda()
        
        targets[torch.lt(IoU_max, 0.4), :] = 0

        positive_indices = torch.ge(IoU_max, 0.5)

        num_positive_anchors = positive_indices.sum()

        assigned_annotations = annotation[IoU_argmax, :]

        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

        alpha_factor = torch.ones(targets.shape) * alpha
        alpha_factor = alpha_factor.cuda()

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

        cls_loss = focal_weight * torch.pow(bce, gamma)

        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        # compute the loss for regression
        # TODO

    return torch.stack(losses).mean()
