import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys


def load_mask(filename):

    mask = cv2.imread(filename, cv2.IMREAD_COLOR)
    # grayscale to binary -> 1 if pixel value > 1, 0 otherwise
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 1, 1, cv2.THRESH_BINARY)
    return binary_mask


def load_masks(directory):
    filenames = sorted(os.listdir(directory))
    masks = []
    for filename in filenames:
        masks.append(load_mask(os.path.join(directory, filename)))
    return masks


def evaluate_mask(gt_mask, pred_mask):
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    accuracy = accuracy_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)

    return accuracy, precision, recall, f1


def evaluate_masks(gt_masks, pred_masks):

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        accuracy, precision, recall, f1_score = evaluate_mask(gt_mask, pred_mask)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return accuracies, precisions, recalls, f1_scores
