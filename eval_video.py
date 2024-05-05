# Description: Evaluate the masks of a video using the ground truth masks.
# Input: GT mask sequence, prediction mask sequence
# Output: Accuracy, Precision, Recall, F1 Score

import numpy as np
import sys
from eval_utils import load_masks, evaluate_masks


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
            "Accuracy": np.mean(metrics[0]),
            "Precision": np.mean(metrics[1]),
            "Recall": np.mean(metrics[2]),
            "F1 Score": np.mean(metrics[3]),
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
