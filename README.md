
## Getting Started

To compile and run the c++ implementation of CodeBooks, follow these steps:

1. Create a build directory and navigate into it:
    ```
    mkdir build && cd build
    ```
2. Run CMake to generate the Makefile:
    ```
    cmake ..
    ```
3. Compile the code:
    ```
    make
    ```
4. Run the program:
    ```
    ./VideoBackgroundSubtraction [-c] [-v video_path] [-o output_path] [-n new video path] [-i new background image path]
    ```

## Command Line Arguments

- `-c` or `-v`: Choose one of these options. `-c` will use the camera as the video source, `-v` followed by the path to a sequence of frames will use that video as the source.
- `-n`: Use this option followed by the path to a new video if you want to change the background of a video.
- `-i`: Use this option followed by the path to an image file if you want to use a new background image.
- `-o`: Use this option followed by the path where you want to store the resulting masks. Use output_path = output/CB/{video_name} if you  want to run the evaluation.


## Running Python implementation of CodeBooks
```
python3 CodeBooks.py
```

- this will run CodeBooks with a video specified in the script

## OpenCV built-in methods
```
python3 OpenCV_BS.py --source camera
```
or
```
python3 OpenCV_BS.py --source folder --path /path/to/{video_name}
```

- this will store the output masks in folders output/MOG/{video_name} and output/MOG2/{video_name}

## Evaluation

### eval_video.py

Usage:
```
python3 eval_video.py video_name
```

- will take output masks from each method (CodeBooks, MOG, MOG2) from output/{method}/{video_name} and calculate the accuracy, precision, recall and f1 scores on each of them compared to ground truth masks located in data/gt/{video_name}