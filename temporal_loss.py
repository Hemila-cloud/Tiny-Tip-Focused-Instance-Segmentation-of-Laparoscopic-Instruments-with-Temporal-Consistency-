import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def focal_loss(pred, target, alpha=0.8, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def combined_temporal_loss(pred, target):
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)

    return 0.5 * d_loss + 0.5 * f_loss
