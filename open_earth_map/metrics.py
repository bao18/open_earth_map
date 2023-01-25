import torch
import torch.nn as nn


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def fscore(pr, gt, beta=1, eps=1e-7, threshold=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )
    return score


class Fscore(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "Fscore"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            scores.append(fscore(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)


class IoU(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "IoU"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            scores.append(iou(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)
