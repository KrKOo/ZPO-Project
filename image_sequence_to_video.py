import os
import cv2

SINGLE_FRAME_EVAL = ["trees", "camouflage", "bootstrap", "lightswitch", "timeofday"]
ALGORITHMS = ["CB", "MOG", "MOG2"]


def image_sequence_to_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder)]

    if "_" in images[0]:
        images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    else:
        images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def main():
    os.makedirs("videos", exist_ok=True)
    for directory in SINGLE_FRAME_EVAL:
        image_sequence_to_video(
            "data/frames/" + directory, f"videos/{directory}_original.avi"
        )
        for algorithm in ALGORITHMS:
            image_sequence_to_video(
                f"output/{algorithm}/{directory}", f"videos/{directory}_{algorithm}.avi"
            )


if __name__ == "__main__":
    main()
