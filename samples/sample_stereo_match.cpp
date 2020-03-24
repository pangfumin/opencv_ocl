#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace ocl;



int main(int argc, char** argv)
{
    cv::Mat left = cv::imread("/home/pang/software/opencv_ocl/data/tsukuba_l.png", cv::IMREAD_GRAYSCALE );

    cv::Mat right = cv::imread("/home/pang/software/opencv_ocl/data/tsukuba_r.png",  cv::IMREAD_GRAYSCALE );

    oclMat d_left, d_right;
    d_left.upload(left);
    d_right.upload(right);

    StereoBM_OCL bm;
    StereoBeliefPropagation bp;
    StereoConstantSpaceBP csbp;

    // Set common parameters
    int ndisp = 30;
    bm.ndisp = ndisp;
    bp.ndisp = ndisp;
    csbp.ndisp = ndisp;

    Mat disp;
    oclMat d_disp;
    enum {BM, BP, CSBP} method;
    method = CSBP;
    switch (method)
    {
        case BM:
            bm(d_left, d_right, d_disp);
            break;
        case BP:
            bp(d_left, d_right, d_disp);
            break;
        case CSBP:
            csbp(d_left, d_right, d_disp);
            break;
    }
    // Show results
    d_disp.download(disp);
    if (method != BM)
    {
        disp.convertTo(disp, 0);
    }

    imshow("left", left);
    imshow("right", right);
    imshow("disparity", disp);

    cv::waitKey();
    return 0;
}
