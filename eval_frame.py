import sys
import os
import cv2
from eval_utils import load_mask, evaluate_mask

GT_DIR = "data/gt"
SINGLE_FRAME_EVAL = ["trees", "camouflage", "bootstrap", "lightswitch", "timeofday"]
ALGORITHMS = ["CB", "MOG", "MOG2"]


def get_file_in_dir(directory):
    return os.listdir(directory)[0]


def get_image_id_from_filename(filename):
    return int(filename.split(".")[0].split("_")[-1])


def get_image_by_id(directory, image_id):
    for file in os.listdir(directory):
        if get_image_id_from_filename(file) == image_id:
            return os.path.join(directory, file)


def get_gt_image(directory):
    return os.path.join(directory, get_file_in_dir(directory))


def get_all_gt_images():
    gt_images = {}
    for directory in SINGLE_FRAME_EVAL:
        path = os.path.join(GT_DIR, directory)
        gt_images[directory] = get_gt_image(path)

    return gt_images


def get_all_prec_images():
    pred_images = {}
    for directory in SINGLE_FRAME_EVAL:
        gt_name = get_file_in_dir(os.path.join(GT_DIR, directory))
        image_id = get_image_id_from_filename(gt_name)

        pred_images[directory] = {}
        for algorithm in ALGORITHMS:
            path = os.path.join("output", algorithm, directory)
            pred_images[directory][algorithm] = get_image_by_id(path, image_id)

    return pred_images


def eval_results(gt_images, pred_images):

    for directory in SINGLE_FRAME_EVAL:
        gt_mask = load_mask(gt_images[directory])
        for algorithm in ALGORITHMS:
            pred_mask = load_mask(pred_images[directory][algorithm])
            metrics = evaluate_mask(gt_mask, pred_mask)
            print(f"Results for {algorithm} on {directory}:")
            print(f"Accuracy: {metrics[0]}")
            print(f"Precision: {metrics[1]}")
            print(f"Recall: {metrics[2]}")
            print(f"F1 Score: {metrics[3]}")
            print(f"-----------------------------------")


def main():
    gt_images = get_all_gt_images()
    pred_images = get_all_prec_images()
    eval_results(gt_images, pred_images)


if __name__ == "__main__":
    main()
