#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <getopt.h>
#include <unistd.h>

#define CHANNELS 3
typedef struct ce
{
    uchar learnHigh[CHANNELS]; //High side threshold for learning
    uchar learnLow[CHANNELS]; //Low side threshold for learning
    uchar max[CHANNELS]; //High side of box boundary
    uchar min[CHANNELS]; //Low side of box boundary
    int t_last_update; //Allow us to kill stale entries
    int stale; //max negative run (longest period of inactivity)
} code_element;

typedef struct code_book
{
    code_element **cb;
    int numEntries;
    int t; //count every access
} codeBook;

//function to update the codebook for a single pixel
int update_codebook(uchar *p, codeBook &c, unsigned *cbBounds, int numChannels)
{
    unsigned int high[CHANNELS], low[CHANNELS];
    for (int n = 0; n < numChannels; n++)
    {
        //adjust the pixel values by the specified bounds and clamp the values to valid range
        high[n] = *(p + n) + *(cbBounds + n);
        if (high[n] > 255)
            high[n] = 255;
        low[n] = *(p + n) - *(cbBounds + n);
        if (low[n] < 0)
            low[n] = 0;
    }
    int matchChannel;
    int z = 0;
    for (int i = 0; i < c.numEntries; i++)
    {
        matchChannel = 0;
        for (int n = 0; n < numChannels; n++)
        {
            //check if the pixel matches the codeword within the learn thresholds
            if ((c.cb[i]->learnLow[n] <= *(p + n)) && (*(p + n) <= c.cb[i]->learnHigh[n]))
            {
                matchChannel++;
            }
        }
        if (matchChannel == numChannels)
        {
            //update codeword if there is a match across all channels
            c.cb[i]->t_last_update = c.t;
            for (int n = 0; n < numChannels; n++)
            {
                if (c.cb[i]->max[n] < *(p + n))
                    c.cb[i]->max[n] = *(p + n);
                if (c.cb[i]->min[n] > *(p + n))
                    c.cb[i]->min[n] = *(p + n);
            }
            break;
        }
        z++;
    }

    //create new codeword if no existing codeword matches
    if (z == c.numEntries)
    {
        //allocate memory for new codeword
        code_element **foo = new code_element *[c.numEntries + 1];
        for (int ii = 0; ii < c.numEntries; ii++)
        {
            foo[ii] = c.cb[ii];
        }
        foo[c.numEntries] = new code_element;
        for (int n = 0; n < numChannels; n++)
        {
            foo[c.numEntries]->learnHigh[n] = high[n];
            foo[c.numEntries]->learnLow[n] = low[n];
            foo[c.numEntries]->max[n] = *(p + n);
            foo[c.numEntries]->min[n] = *(p + n);
        }
        foo[c.numEntries]->t_last_update = c.t;
        foo[c.numEntries]->stale = 0;
        if (c.cb)
            delete[] c.cb;
        c.cb = foo;
        c.numEntries++;
    }

    //adjust learning bounds slowly to adapt to changes in the scene
    for (int n = 0; n < numChannels; n++)
    {
        if (z < c.numEntries && c.cb[z]->learnHigh[n] < high[n])
            c.cb[z]->learnHigh[n]++;
        if (z < c.numEntries && c.cb[z]->learnLow[n] > low[n])
            c.cb[z]->learnLow[n]--;
    }
    return z;
}

int clear_stale_entries(codeBook &c)
{
    int staleThresh = c.t >> 1;
    bool *keep = new bool[c.numEntries];
    int keepCnt = 0;
 
    //see which codebook entries are too stale
    for (int i = 0; i < c.numEntries; i++)
    {
        if (c.cb[i]->stale > staleThresh)
        {
            keep[i] = false; //mark for destruction
        }
        else
        {
            keep[i] = true; //mark to keep
            keepCnt++;
        }
    }

    //keep only the good entries
    code_element **newEntries = new code_element *[keepCnt];
    int j = 0;
    for (int i = 0; i < c.numEntries; i++)
    {
        if (keep[i])
        {
            newEntries[j++] = c.cb[i];
        }
        else
        {
            delete c.cb[i];
        }
    }

    //clean up
    delete[] c.cb;
    c.cb = newEntries;
    c.numEntries = keepCnt;
    c.t = 0;

    delete[] keep;
    return (c.numEntries - keepCnt);
}

uchar background_diff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod)
{
    int ii = 0;
    //see if this pixel is within range of an existing codeword
    for (int i = 0; i < c.numEntries; i++)
    {
        int matchChannel = 0;
        for (int n = 0; n < numChannels; n++)
        {
            if ((c.cb[i]->min[n] - minMod[n] <= p[n]) && (p[n] <= c.cb[i]->max[n] + maxMod[n]))
            {
                matchChannel++; //found an entry for this channel
            }
            else
            {
                break;
            }
        }
        if (matchChannel == numChannels)
        {
            break; //found an entry that matches all channels
        }
        ii++;
    }
    if (ii >= c.numEntries)
        return (255);
    return (0);
}


void processFrame(cv::Mat &frame, cv::Mat &newFrame, std::vector<std::vector<codeBook>> &codebooks, int *minMod, int *maxMod, unsigned *cbBounds)
{
    newFrame.create(frame.size(), CV_8UC3);

    //loop through each pixel in the frame
    for (int x = 0; x < frame.rows; x++)
    {
        for (int y = 0; y < frame.cols; y++)
        {
            uchar *p = frame.ptr(x, y);
            uchar *p2 = newFrame.ptr(x, y);
            //apply background difference check
            p2[0] = background_diff(p, codebooks[x][y], CHANNELS, minMod, maxMod);
            p2[1] = p2[0];
            p2[2] = p2[0];
        }
    }

    //update the codebook for each pixel in the frame
    for (int x = 0; x < frame.rows; x++)
    {
        for (int y = 0; y < frame.cols; y++)
        {
            uchar *p = frame.ptr(x, y);
            update_codebook(p, codebooks[x][y], cbBounds, CHANNELS);
        }
    }
}

int main(int argc, char *argv[])
{

    int option;
    bool useCamera = false;
    std::string outputPath;
    std::string videoPath;
    std::string newVideoPath;
    std::string newImagePath;
    cv::Mat frame, newFrame;
    int width, height;
    unsigned cbBounds[3] = {10, 10, 10};
    int minMod[3] = {10, 10, 10};
    int maxMod[3] = {10, 10, 10};

    //parse command line arguments
    while ((option = getopt(argc, argv, "cv:o:n:i:")) != -1)
    {
        switch (option)
        {
        case 'c':
            useCamera = true;
            break;
        case 'v':
            videoPath = optarg;
            break;
        case 'o':
            outputPath = optarg;
            break;
        case 'n':
            newVideoPath = optarg;
            break;
        case 'i':
            newImagePath = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " [-c] [-v video_path] [-o output_path] [-n new video path] [-i new background image path]\n";
            return -1;
        }
    }
    std::cout << "Video path:" << videoPath << std::endl;
    std::cout << "Output path:" << outputPath << std::endl;
    std::cout << "New vide path:" << newVideoPath << std::endl;

    //handle output directories
    if (!outputPath.empty())
    {
        std::filesystem::path dir(outputPath);
        if (!std::filesystem::exists(dir))
        {
            if (!std::filesystem::create_directories(dir))
            {
                std::cerr << "Failed to create directory: " << outputPath << std::endl;
            }
        }
    }
    if (!newVideoPath.empty())
    {
        std::filesystem::path dir(newVideoPath);
        if (!std::filesystem::exists(dir))
        {
            if (!std::filesystem::create_directories(dir))
            {
                std::cerr << "Failed to create directory: " << newVideoPath << std::endl;
            }
        }
    }

    //camera or video input handling
    if (useCamera)
    {

        cv::VideoCapture cap(0);
        cap.open(0);

        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera\n";
            return -1;
        }

        width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::vector<std::vector<codeBook>> codebooks(height, std::vector<codeBook>(width));

        //camera loop
        while (true)
        {
            cap >> frame;
            if (frame.empty())
                break;

            processFrame(frame, newFrame, codebooks, minMod, maxMod, cbBounds);
            cv::imshow("Frame", newFrame);
            /*
            if (!outputPath.empty()){
                std::string filename = outputPath + "/frame_" + std::to_string(cv::getTickCount()) + ".png";
                if (!cv::imwrite(filename, newFrame))
                {
                    std::cerr << "Failed to write image: " << filename << std::endl;
                }
            }*/

            if (cv::waitKey(1) == 'q')
                break;
        }

        cap.release();
        cv::destroyAllWindows();
    }
    else if (!videoPath.empty())
    {

        //read all images in the video directory
        std::vector<std::string> files;
        for (const auto &entry : std::filesystem::directory_iterator(videoPath))
        {
            files.push_back(entry.path());
        }

        std::sort(files.begin(), files.end());

        std::vector<cv::Mat> images;
        for (const auto &file : files)
        {
            images.push_back(cv::imread(file));
        }

        width = images[0].cols;
        height = images[0].rows;
        std::vector<std::vector<codeBook>> codebooks(height, std::vector<codeBook>(width));
        int iter = 0;
        //video processing loop
        for (auto &frame : images)
        {
            processFrame(frame, newFrame, codebooks, minMod, maxMod, cbBounds);
            cv::imshow("Frame", newFrame);

            if (!outputPath.empty() && iter != 0)
            {
                std::string filename = outputPath + "/frame_" + std::to_string(iter) + ".png";
                if (!cv::imwrite(filename, newFrame))
                {
                    std::cerr << "Failed to write image: " << filename << std::endl;
                }
            }

            if (!newVideoPath.empty() && iter != 0)
            {
                cv::Mat newImage = cv::imread(newImagePath);
                cv::Mat outputFrame = cv::Mat::zeros(newFrame.size(), CV_8UC3);
                newFrame.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int pos[]) -> void
                                            {
                    int x = pos[0];
                    int y = pos[1];
                    if (pixel[0] == 0) {
                        outputFrame.at<cv::Vec3b>(x, y) = newImage.at<cv::Vec3b>(x, y);
                    }
                    else{
                        outputFrame.at<cv::Vec3b>(x, y) = frame.at<cv::Vec3b>(x, y);
                    } });
                std::string newFilename = newVideoPath + "/frame_" + std::to_string(cv::getTickCount()) + ".png";
                if (!cv::imwrite(newFilename, outputFrame))
                {
                    std::cerr << "Failed to write image: " << newFilename << std::endl;
                }
            }
            iter++;
            if (cv::waitKey(1) == 'q')
                break;
        }

        cv::destroyAllWindows();
    }
    else
    {
        std::cerr << "Error: No input source specified\n";
        return -1;
    }

    return 0;
}