#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "utility.h"

using namespace cv;
using namespace std;
using namespace ocl;



void pipeline_sgbm(const cv::Mat image_left, const cv::Mat image_right,
                   cv::Mat& image_disp) {

    cv::Mat I_l = image_left;
    cv::Mat I_r = image_right;


    Mat g1, g2;
    //Mat disp, disp8;


    // Parameter values according to:
    // http://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&result=3ae300a3a3b3ed3e48a63ecb665dffcc127cf8ab
//	StereoSGBM sgbm;

    int SgbmSADWindowSize = 3;
    cv::Ptr<cv::StereoBM> sgbm = StereoBM::create(0, 21 );
    sgbm->setNumDisparities(128);
    sgbm->setPreFilterCap(63);
    sgbm->setMinDisparity(0);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
//	sgbm.fullDP = 1;
//    sgbm->setP1(SgbmSADWindowSize*SgbmSADWindowSize*4);
//    sgbm->setP2(SgbmSADWindowSize*SgbmSADWindowSize*32);

    sgbm->compute(image_left,image_right, image_disp);

}



int main(int argc, char** argv)
{
    cv::Mat left = cv::imread("/home/pang/software/opencv_ocl/data/000000_l.png", cv::IMREAD_GRAYSCALE );

    cv::Mat right = cv::imread("/home/pang/software/opencv_ocl/data/000000_r.png",  cv::IMREAD_GRAYSCALE );

    oclMat d_left, d_right;
    d_left.upload(left);
    d_right.upload(right);

    StereoBM_OCL bm;
    StereoBeliefPropagation bp;
    StereoConstantSpaceBP csbp;

    // Set common parameters
    int ndisp = 128;
    bm.ndisp = ndisp;
    bm.preset = 6;
    bm.winSize = 6;
    bp.ndisp = ndisp;
    csbp.ndisp = ndisp;

//    int preset;
//    int ndisp;
//    int winSize;

    Mat cpu_disp;
    Mat gpu_disp;
    oclMat d_disp;
    enum {BM, BP, CSBP} method;
    method = BM;
    TicToc gpu_timer;
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
    std::cout << "gpu_timing: " << gpu_timer.toc() << std::endl;
    // Show results
    d_disp.download(gpu_disp);
    if (method != BM)
    {
        gpu_disp.convertTo(gpu_disp, 0);
    }

//    imshow("left", left);
//    imshow("right", right);
//    imshow("disparity", disp);

    gpu_disp.convertTo(gpu_disp,CV_8UC1);

    TicToc cpu_timer;
    pipeline_sgbm(left, right,cpu_disp);
    std::cout << "cpu_timing: " << cpu_timer.toc() << std::endl;

    cpu_disp.convertTo(cpu_disp, CV_32F, 1.0/16);
    cpu_disp.convertTo(cpu_disp,CV_8UC1);

    imshow("gpu_disp", gpu_disp);
    imshow("cpu_disp", cpu_disp);
    cv::waitKey();
    return 0;
}
