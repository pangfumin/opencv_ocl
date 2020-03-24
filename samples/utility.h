#ifndef __OPENCV_OCL_UTILITY__
#define __OPENCV_OCL_UTILITY__

#include <chrono>
#include <opencv2/core/core.hpp>
class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

/// https://www.mathworks.com/help/matlab/ref/jet.html
const double jet[64][3] = {
        {     0,         0,    0.5625},
        {     0,         0,    0.6250},
        {     0,         0,    0.6875},
        {     0,         0,    0.7500},
        {     0,         0,    0.8125},
        {     0,         0,    0.8750},
        {     0,         0,    0.9375},
        {     0,         0,    1.0000},
        {     0,    0.0625,    1.0000},
        {     0,    0.1250,    1.0000},
        {     0,    0.1875,    1.0000},
        {     0,    0.2500,    1.0000},
        {     0,    0.3125,    1.0000},
        {     0,    0.3750,    1.0000},
        {     0,    0.4375,    1.0000},
        {     0,    0.5000,    1.0000},
        {     0,    0.5625,    1.0000},
        {     0,    0.6250,    1.0000},
        {     0,    0.6875,    1.0000},
        {     0,    0.7500,    1.0000},
        {     0,    0.8125,    1.0000},
        {     0,    0.8750,    1.0000},
        {     0,    0.9375,    1.0000},
        {     0,    1.0000,    1.0000},
        {0.0625,    1.0000,    0.9375},
        {0.1250,    1.0000,    0.8750},
        {0.1875,    1.0000,    0.8125},
        {0.2500,    1.0000,    0.7500},
        {0.3125,    1.0000,    0.6875},
        {0.3750,    1.0000,    0.6250},
        {0.4375,    1.0000,    0.5625},
        {0.5000,    1.0000,    0.5000},
        {0.5625,    1.0000,    0.4375},
        {0.6250,    1.0000,    0.3750},
        {0.6875,    1.0000,    0.3125},
        {0.7500,    1.0000,    0.2500},
        {0.8125,    1.0000,    0.1875},
        {0.8750,    1.0000,    0.1250},
        {0.9375,    1.0000,    0.0625},
        {1.0000,    1.0000,         0},
        {1.0000,    0.9375,         0},
        {1.0000,    0.8750,         0},
        {1.0000,    0.8125,         0},
        {1.0000,    0.7500,         0},
        {1.0000,    0.6875,         0},
        {1.0000,    0.6250,         0},
        {1.0000,    0.5625,         0},
        {1.0000,    0.5000,         0},
        {1.0000,    0.4375,         0},
        {1.0000,    0.3750,         0},
        {1.0000,    0.3125,         0},
        {1.0000,    0.2500,         0},
        {1.0000,    0.1875,         0},
        {1.0000,    0.1250,         0},
        {1.0000,    0.0625,         0},
        {1.0000,         0,         0},
        {0.9375,         0,         0},
        {0.8750,         0,         0},
        {0.8125,         0,         0},
        {0.7500,         0,         0},
        {0.6875,         0,         0},
        {0.6250,         0,         0},
        {0.5625,         0,         0},
        {0.5000,         0,         0},
};


cv::Mat colorDisparityImage(const cv::Mat1b& dis_image) {
    cv::Mat color_image(dis_image.rows, dis_image.cols, CV_8UC3);
    color_image.setTo(cv::Scalar(0,0,0));

    double min, max;
    cv::minMaxLoc(dis_image, &min, &max);
    std::cout << "max: " << max << " " << min << std::endl;

    for (int i = 0; i < dis_image.rows; i++) {
        for (int j = 0; j < dis_image.cols; j++) {
            auto dis = dis_image.at<char>(i,j);

            int index = (int)((dis - min) /(max - min) * 63.0);
//            std::cout << index << " ";
//            int index =   63.0 / (dis - 1);
//            if (index > 63) index = 63;

            const double *color_jet = jet[index];
            color_image.at<cv::Vec3b>(i, j)[0] = 255 * (*color_jet);
            color_image.at<cv::Vec3b>(i, j)[1] = 255 * (*color_jet + 1);
            color_image.at<cv::Vec3b>(i, j)[2] = 255 * (*color_jet + 2);
        }

//        std::cout << std::endl;

    }

    return color_image;

}


#endif
