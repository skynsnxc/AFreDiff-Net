import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix


def iou_score1(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = (output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output >= 0.5
    target_ = target >= 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return iou, dice


def iou_score(output, target):
    smooth = 1e-5
    # _, output = output  # 运行train文件需要注释
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output >= 0.5
    target_ = target >= 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return iou, dice
"""def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output >= 0.5
    target_ = target >= 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection) / (union + smooth)  # 修正 Dice 系数的计算
    return iou, dice
"""
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
def hausdorff_distance(output, target):
    def binary_edge_points(binary_mask):
        edges = np.zeros_like(binary_mask)
        edges[1:-1, 1:-1] = (binary_mask[1:-1, 1:-1] != np.maximum(
            binary_mask[0:-2, 1:-1],
            binary_mask[2:, 1:-1])) | (binary_mask[1:-1, 1:-1] != np.maximum(
            binary_mask[1:-1, 0:-2],
            binary_mask[1:-1, 2:]))
        return np.argwhere(edges)

    if torch.is_tensor(output):
        output = (output >= 0.5).data.cpu().numpy().astype(np.uint8)
    if torch.is_tensor(target):
        target = (target >= 0.5).data.cpu().numpy().astype(np.uint8)

    output_edges = binary_edge_points(output)
    target_edges = binary_edge_points(target)

    if len(output_edges) == 0 or len(target_edges) == 0:
        return np.nan

    return max(directed_hausdorff(output_edges, target_edges)[0],
               directed_hausdorff(target_edges, output_edges)[0])


def average_boundary_distance(output, target):
    def binary_edge_points(binary_mask):
        edges = np.zeros_like(binary_mask)
        edges[1:-1, 1:-1] = (binary_mask[1:-1, 1:-1] != np.maximum(
            binary_mask[0:-2, 1:-1],
            binary_mask[2:, 1:-1])) | (binary_mask[1:-1, 1:-1] != np.maximum(
            binary_mask[1:-1, 0:-2],
            binary_mask[1:-1, 2:]))
        return np.argwhere(edges)

    def average_distance(points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return np.nan
        distances = [np.min(np.sqrt(np.sum((points1 - p2) ** 2, axis=1))) for p2 in points2]
        return np.mean(distances)

    if torch.is_tensor(output):
        output = (output >= 0.5).data.cpu().numpy().astype(np.uint8)
    if torch.is_tensor(target):
        target = (target >= 0.5).data.cpu().numpy().astype(np.uint8)

    output_edges = binary_edge_points(output)
    target_edges = binary_edge_points(target)

    avg_dist_out_to_tar = average_distance(output_edges, target_edges)
    avg_dist_tar_to_out = average_distance(target_edges, output_edges)

    return (avg_dist_out_to_tar + avg_dist_tar_to_out) / 2