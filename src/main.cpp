#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <getopt.h>
#include <unistd.h>

#define CHANNELS 3
typedef struct ce {
    uchar learnHigh[CHANNELS];
    uchar learnLow[CHANNELS];
    uchar max[CHANNELS];
    uchar min[CHANNELS];
    int t_last_update;
    int stale;
} code_element;

typedef struct code_book {
    code_element **cb;
    int numEntries;
    int t;
} codeBook;

int update_codebook(uchar* p, codeBook& c, unsigned* cbBounds, int numChannels) {
    unsigned int high[CHANNELS], low[CHANNELS];
    for (int n = 0; n < numChannels; n++) {
        high[n] = *(p+n) + *(cbBounds+n);
        if (high[n] > 255) high[n] = 255;
        low[n] = *(p+n) - *(cbBounds+n);
        if (low[n] < 0) low[n] = 0;
    }
    int matchChannel;
    int z = 0;
    for (int i = 0; i < c.numEntries; i++) {
        matchChannel = 0;
        for (int n = 0; n < numChannels; n++) {
            if ((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n])) {
                matchChannel++;
            }
        }
        if (matchChannel == numChannels) {
            c.cb[i]->t_last_update = c.t;
            for (int n = 0; n < numChannels; n++) {
                if (c.cb[i]->max[n] < *(p+n)) c.cb[i]->max[n] = *(p+n);
                if (c.cb[i]->min[n] > *(p+n)) c.cb[i]->min[n] = *(p+n);
            }
            break;
        }
        z++;
    }

    if (z == c.numEntries) {
        code_element** foo = new code_element*[c.numEntries + 1];
        for (int ii = 0; ii < c.numEntries; ii++) {
            foo[ii] = c.cb[ii];
        }
        foo[c.numEntries] = new code_element;
        for (int n = 0; n < numChannels; n++) {
            foo[c.numEntries]->learnHigh[n] = high[n];
            foo[c.numEntries]->learnLow[n] = low[n];
            foo[c.numEntries]->max[n] = *(p+n);
            foo[c.numEntries]->min[n] = *(p+n);
        }
        foo[c.numEntries]->t_last_update = c.t;
        foo[c.numEntries]->stale = 0;
        if (c.cb) delete[] c.cb;
        c.cb = foo;
        c.numEntries++;
    }

    for (int n = 0; n < numChannels; n++) {
        if (z < c.numEntries && c.cb[z]->learnHigh[n] < high[n]) c.cb[z]->learnHigh[n]++;
        if (z < c.numEntries && c.cb[z]->learnLow[n] > low[n]) c.cb[z]->learnLow[n]--;
    }
    return z;
}


int clear_stale_entries(codeBook &c) {
    int staleThresh = c.t >> 1; 
    bool *keep = new bool[c.numEntries];
    int keepCnt = 0;

    for (int i = 0; i < c.numEntries; i++) {
        if (c.cb[i]->stale > staleThresh) {
            keep[i] = false; 
        } else {
            keep[i] = true; 
            keepCnt++;
        }
    }


    code_element **newEntries = new code_element*[keepCnt];
    int j = 0;
    for (int i = 0; i < c.numEntries; i++) {
        if (keep[i]) {
            newEntries[j++] = c.cb[i];
        } else {
            delete c.cb[i];
        }
    }

    delete [] c.cb;
    c.cb = newEntries;
    c.numEntries = keepCnt;
    c.t = 0;

    delete [] keep; 
    return (c.numEntries - keepCnt); 
}


uchar background_diff(uchar* p, codeBook& c, int numChannels, int* minMod, int* maxMod) {
    int ii = 0;
    for (int i = 0; i < c.numEntries; i++) {
        int matchChannel = 0;
        for (int n = 0; n < numChannels; n++) {
            if ((c.cb[i]->min[n] - minMod[n] <= p[n]) && (p[n] <= c.cb[i]->max[n] + maxMod[n])) {
                matchChannel++; 
            } else {
                break;
            }
        }
        if (matchChannel == numChannels) {
            break;
        }
        ii++;
    }
    if(ii >= c.numEntries) return(255);
    return(0);
}

void processFrame(cv::Mat& frame, cv::Mat& newFrame, std::vector<std::vector<codeBook>>& codebooks, int* minMod, int* maxMod, unsigned* cbBounds) {
    newFrame.create(frame.size(), CV_8UC1);

    for (int x = 0; x < frame.rows; x++) {
        for (int y = 0; y < frame.cols; y++) {
            uchar* p = frame.ptr(x, y);
            uchar* p2 = newFrame.ptr(x, y);
            p2[0] = background_diff(p, codebooks[x][y], CHANNELS, minMod, maxMod);
        }
    }

    for (int x = 0; x < frame.rows; x++) {
        for (int y = 0; y < frame.cols; y++) {
            uchar* p = frame.ptr(x, y);
            update_codebook(p, codebooks[x][y], cbBounds, CHANNELS);
        }
    }

}

int main(int argc, char* argv[]) {

    int option;
    bool useCamera = false;
    std::string outputPath;
    std::string videoPath;
    cv::Mat frame, newFrame;
    int width, height;
    unsigned cbBounds[3] = {10, 10, 10}; 
    int minMod[3] = {10, 10, 10}; 
    int maxMod[3] = {10, 10, 10}; 

    while ((option = getopt(argc, argv, "cv:")) != -1) {
        switch (option) {
            case 'c':
                useCamera = true;
                break;
            case 'v':
                videoPath = optarg;
                break;
            /*case 'o':
                outputPath = optarg;
                break;*/
            default:
                std::cerr << "Usage: " << argv[0] << " [-c] [-v video_path] [-o output_path]\n";
                return -1;
        }
    }

    //std::cout<<outputPath<<std::endl;
    /*if (!outputPath.empty())
    {
        std::filesystem::path dir(outputPath);
        if (!std::filesystem::exists(dir))
        {
            if (!std::filesystem::create_directories(dir))
            {
                std::cerr << "Failed to create directory: " << outputPath << std::endl;
               
            }
        }
    }*/

    if (useCamera) {

        cv::VideoCapture cap(0); 
        cap.open(0);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera\n";
            return -1;
        }

        width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::vector<std::vector<codeBook>> codebooks(height, std::vector<codeBook>(width));

        while(true){
            cap >> frame; 
            if (frame.empty()) break; 
    
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

            if (cv::waitKey(1) == 'q') break;
        
        }

        cap.release();
        cv::destroyAllWindows();

    } else if (!videoPath.empty()) {

        std::vector<std::string> files;
        for (const auto & entry : std::filesystem::directory_iterator(videoPath)) {
            files.push_back(entry.path());
        }

        std::sort(files.begin(), files.end());

        std::vector<cv::Mat> images;
        for (const auto & file : files) {
            images.push_back(cv::imread(file));
        }

        width = images[0].cols;
        height = images[0].rows;
        std::vector<std::vector<codeBook>> codebooks(height, std::vector<codeBook>(width));

        for(auto &frame : images){

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

            if (cv::waitKey(1) == 'q') break;
        }

        cv::destroyAllWindows();

    } else {
        std::cerr << "Error: No input source specified\n";
        return -1;
    }

    return 0;
}