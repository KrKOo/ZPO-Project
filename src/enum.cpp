#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <getopt.h>

int main(int argc, char *argv[])
{
    int option;
    std::string refPath;
    std::string testPath;
    while ((option = getopt(argc, argv, "r:t:")) != -1)
    {
        switch (option)
        {
        case 'r':
            refPath = optarg;
            break;
        case 't':
            testPath = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " [-r ref_images_path] [-t testing_images_path]\n";
            return -1;
        }
    }

    std::filesystem::path refFolder(refPath);
    std::filesystem::path testFolder(testPath);
    std::vector<std::filesystem::path> refPaths, testPaths;

    for (const auto &entry : std::filesystem::directory_iterator(refFolder))
    {
        if (entry.is_regular_file())
        {
            refPaths.push_back(entry.path());
        }
    }
    for (const auto &entry : std::filesystem::directory_iterator(testFolder))
    {
        if (entry.is_regular_file())
        {
            testPaths.push_back(entry.path());
        }
    }

    std::sort(refPaths.begin(), refPaths.end());
    std::sort(testPaths.begin(), testPaths.end());
    int counter;
    int min = 100;
    int max = 0;
    int sum = 0;
    for (size_t i = 0; i < refPaths.size(); i++)
    {
        counter = 0;
        cv::Mat refImg = cv::imread(refPaths[i].string());
        cv::Mat testImg = cv::imread(testPaths[i].string());
        int width = refImg.cols;
        int height = refImg.rows;
        refImg.forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int pos[]) -> void
                                  {
            int y = pos[0];  // row
            int x = pos[1];  // column
            cv::Vec3b &testPixel = testImg.at<cv::Vec3b>(y, x);
            if ((pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && testPixel[0] == 0 && testPixel[1] == 0 && testPixel[2] == 0) ||  // Both pixels are black
                (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0) && (testPixel[0] != 0 || testPixel[1] != 0 || testPixel[2] != 0)) {  // Both pixels are not black
                counter++;
            } });
        int perc = (counter * 100) / (width * height);
        if (perc < min)
            min = perc;
        if (perc > max)
            max = perc;
        sum += perc;
    }
    int avg = sum / refPaths.size();
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    std::cout << "Avg: " << avg << std::endl;
}
