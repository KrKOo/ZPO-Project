import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

def load_masks(directory):

    filenames = sorted(os.listdir(directory))
    masks = []
    for filename in filenames:
       
        mask = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_COLOR)
        #grayscale to binary -> 1 if pixel value > 1, 0 otherwise
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 1, cv2.THRESH_BINARY)  
        masks.append(binary_mask)
    return masks

def evaluate_masks(gt_masks, pred_masks):

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
   
        gt_flat = gt_mask.flatten()
        pred_flat = pred_mask.flatten()
        
        accuracies.append(accuracy_score(gt_flat, pred_flat))
        precisions.append(precision_score(gt_flat, pred_flat, zero_division=0))
        recalls.append(recall_score(gt_flat, pred_flat, zero_division=0))
        f1_scores.append(f1_score(gt_flat, pred_flat, zero_division=0))
    
    return accuracies, precisions, recalls, f1_scores

def main(video_name):

    base_path = "output"
    gt_path = f"data/gt/{video_name}"
    gt_masks = load_masks(gt_path)[1:]
    
    methods = ["MOG", "MOG2", "CB"]
    results = {}

    for method in methods:
        method_path = f"{base_path}/{method}/{video_name}"
        pred_masks = load_masks(method_path)
        metrics = evaluate_masks(gt_masks, pred_masks)
        results[method] = {
            'Accuracy': np.mean(metrics[0]),
            'Precision': np.mean(metrics[1]),
            'Recall': np.mean(metrics[2]),
            'F1 Score': np.mean(metrics[3])
        }

    for method, metrics in results.items():
        print(f"Results for {method}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        main(video_name)
    else:
        print("Usage: python script_name.py video_name")
