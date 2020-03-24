#include <iostream>
#include <vector>
#include <iomanip>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

typedef unsigned char uchar;
#define LOOP_NUM 10
int64 work_begin = 0;
int64 work_end = 0;

static void workBegin()
{
    work_begin = getTickCount();
}
static void workEnd()
{
    work_end += (getTickCount() - work_begin);
}
static double getTime()
{
    return work_end * 1000. / getTickFrequency();
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

static void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}


int main(int argc, const char* argv[])
{


    string fname0 = "/home/pang/software/opencv_ocl/data/tsukuba_l.png";
    string fname1 = "/home/pang/software/opencv_ocl/data/tsukuba_r.png";


    bool useCPU = false;


    Mat frame0 = imread(fname0, cv::IMREAD_GRAYSCALE);
    Mat frame1 = imread(fname1, cv::IMREAD_GRAYSCALE);
    cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();
    cv::ocl::OpticalFlowDual_TVL1_OCL d_alg;

    Mat flow, show_flow;
    Mat flow_vec[2];

    oclMat d_flowx, d_flowy;
    for(int i = 0; i <= LOOP_NUM; i ++)
    {
        cout << "loop" << i << endl;

        if (i > 0) workBegin();
        if (useCPU)
        {
            alg->calc(frame0, frame1, flow);
            split(flow, flow_vec);
        }
        else
        {
            d_alg(oclMat(frame0), oclMat(frame1), d_flowx, d_flowy);
            d_flowx.download(flow_vec[0]);
            d_flowy.download(flow_vec[1]);
        }
        if (i > 0 && i <= LOOP_NUM)
            workEnd();

        if (i == LOOP_NUM)
        {
            if (useCPU)
                cout << "average CPU time (noCamera) : ";
            else
                cout << "average GPU time (noCamera) : ";
            cout << getTime() / LOOP_NUM << " ms" << endl;

            getFlowField(flow_vec[0], flow_vec[1], show_flow);
            imshow("PyrLK [Sparse]", show_flow);
//            imwrite(outpath, show_flow);
        }
    }


    waitKey();

    return 0;
}