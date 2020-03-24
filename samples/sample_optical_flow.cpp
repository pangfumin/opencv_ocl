 #include <opencv2/core/core.hpp>
 #include "opencv2/ocl/ocl.hpp"
 #include <opencv2/highgui/highgui.hpp>
 #include <opencv2/features2d/features2d.hpp>
 #include <opencv2/video/video.hpp>


#include <ctime>
#include <cstdlib>
#include <chrono>

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

using namespace cv;
using namespace cv::ocl;

 int main() {
     bool useGray = true; 
    cv::Mat frame0 = cv::imread("/home/pang/dataset/euroc/MH_01_easy/mav0/cam0/data/1403636696463555584.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    // ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = cv::imread("/home/pang/dataset/euroc/MH_01_easy/mav0/cam1/data/1403636696463555584.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    // ASSERT_FALSE(frame1.empty());

    cv::resize(frame0, frame0, cv::Size(2 * frame0.cols, 2 * frame0.rows));
    cv::resize(frame1, frame1, cv::Size(2 * frame1.cols, 2 * frame1.rows));
    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

   

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

    cv::ocl::oclMat d_pts;
    cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void *)&pts[0]);
    d_pts.upload(pts_mat);

    cv::ocl::PyrLKOpticalFlow pyrLK;

    cv::ocl::oclMat oclFrame0;
    cv::ocl::oclMat oclFrame1;
    cv::ocl::oclMat d_nextPts;
    cv::ocl::oclMat d_status;
    cv::ocl::oclMat d_err;

    oclFrame0 = frame0;
    oclFrame1 = frame1;

    TicToc gpu_timer;
    pyrLK.sparse(oclFrame0, oclFrame1, d_pts, d_nextPts, d_status, &d_err);
    std::cout << "gpu_timer: " << gpu_timer.toc() << std::endl;

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void *)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void *)&status[0]);
    d_status.download(status_mat);

    std::vector<float> err(d_err.cols);
    cv::Mat err_mat(1, d_err.cols, CV_32FC1, (void*)&err[0]);
    d_err.download(err_mat);

    std::cout << "d_nextPts: " << nextPts_mat.cols << std::endl;

    std::vector<cv::Point2f> nextPts_gold;
    std::vector<unsigned char> status_gold;
    std::vector<float> err_gold;
     TicToc cpu_timer;
    
    cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold);
    std::cout << "cpu_timer: " << cpu_timer.toc() << std::endl;

    // ASSERT_EQ(nextPts_gold.size(), nextPts.size());
    // ASSERT_EQ(status_gold.size(), status.size());

    cv::Mat blend_image = 0.5 * frame0 + 0.5 * frame1;
    cv::cvtColor(blend_image, blend_image, CV_GRAY2RGB);

    size_t mistmatch = 0;
    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        if (status[i] != status_gold[i])
        {
            ++mistmatch;
            continue;
        }

        if (status[i])
        {
            cv::Point2i pre_a = pts[i];
            cv::Point2i a = nextPts[i];
            cv::Point2i b = nextPts_gold[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;
            //float errdiff = std::abs(err[i] - err_gold[i]);
            float errdiff = 0.0f;

            if (!eq || errdiff > 1e-1)
                ++mistmatch;


            cv::circle(blend_image, a, 1, cv::Scalar(0,225,0));
            cv::circle(blend_image, pre_a, 1, cv::Scalar(0,0,255));
            cv::line(blend_image, pre_a, a, cv::Scalar(225,0,255));
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / (nextPts.size());
    std::cout << "bad_ratio: " << bad_ratio << std::endl;

    cv::imshow("track", blend_image);
    cv::waitKey(1);

    // ASSERT_LE(bad_ratio, 0.02f);
    return 0;
 }
