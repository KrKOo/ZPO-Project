import numpy as np
import cv2
import os
import glob
import argparse

def process_frame(img, frame_count, output_dir):

    fgmask1 = fgbg1.apply(img)
    fgmask2 = fgbg2.apply(img)

    cv2.imshow('Original', img)
    cv2.imshow('MOG', fgmask1)
    cv2.imshow('MOG2', fgmask2)


    if frame_count is not None and output_dir is not None:
        cv2.imwrite(f"{output_dir['MOG']}/frame_{frame_count}.jpg", fgmask1)
        cv2.imwrite(f"{output_dir['MOG2']}/frame_{frame_count}.jpg", fgmask2)


parser = argparse.ArgumentParser(description='Process video input from camera or folder.')
parser.add_argument('--source', choices=['camera', 'folder'], required=True,
                    help='select the source of the input, "camera" for live feed or "folder" for images in a directory')
parser.add_argument('--path', type=str, help='the path to the folder containing images, required if source is "folder"')

args = parser.parse_args()

fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.createBackgroundSubtractorMOG2()

if args.source == 'camera':
    cap = cv2.VideoCapture(0) 
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break
        process_frame(img, None, None) 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()

elif args.source == 'folder':
    if not args.path:
        raise ValueError("Path is required when source is 'folder'")


    last_part = args.path.strip('/').split('/')[-1]
    output_dirs = {
        "MOG": f"output/MOG/{last_part}",
        "MOG2": f"output/MOG2/{last_part}"
      
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(args.path, '*.jpg')))
    frame_count = 0
    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        if img is None:
            continue
        process_frame(img, frame_count, output_dirs)
        frame_count += 1
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
