#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <math.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/format.hpp>
#include <ctime>
#include <iostream>
#include <memory>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pm_msg/pose.h"
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ceres;
//! stereo camera match
class PuzzleMatch {
 public:
  PuzzleMatch();
  bool method = 1;
  // ~PuzzleMatch();
  Mat scene, background, full, full_;
  vector<double> dist;
  Eigen::Matrix<float, 3, 1> origin_pos;
  Eigen::Matrix<float, 3, 1> camera_pos;
  double height = 0.6, p_width, p_height;
  //! Intrinsic parameters
  Mat M1, D1, M2, D2;
  Mat img_l, img_r;
  //! Extrinsic parameters
  Mat R, T, R1, P1, R2, P2;
  //! disparity map and depth map
  cv::Mat disp, disp8, depth;
  Mat pic_l, pic_r;
  pm_msg::pose poses;
  ros::Publisher pose_pub;
  typedef struct {
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
  } four_corners_t;
  double result_angle;
  four_corners_t corners;
  cv::Mat1f camera_matrix;
  Eigen::Matrix<float, 3, 3> camera_matrix_2;
  Mat distortion_coefficients;
  Mat transH;
  Rect box_target;
  image_transport::Subscriber camera_subscriber;
  vector<KeyPoint> keypoints;
  Mat descriptors;
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void mainloop();
  void Init(Mat scene, Mat background, Mat full);
  vector<Rect> getROI(Mat image, bool show);
  void featurematch(Mat src1, Mat src2, vector<KeyPoint> keypoints2,
                    Mat descriptors2, Rect roi, Rect roi_full);

  void stitchImage(Mat src1, Mat src2);
  void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

  void find_feature_matches(const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector<DMatch>& matches);

  void pose_estimation_2d2d(const std::vector<KeyPoint>& keypoints_1,
                            const std::vector<KeyPoint>& keypoints_2,
                            const std::vector<DMatch>& matches, Mat& R, Mat& t);
  struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double xt, double yt, double xw, double yw)
        : xt_(xt), yt_(yt), xw_(xw), yw_(yw) {}
    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
      residual[0] = T(xw_) * ceres::cos(abc[0]) - T(yw_) * ceres::sin(abc[0]) +
                    abc[1] - T(xt_);
      residual[1] = T(xw_) * ceres::sin(abc[0]) + T(yw_) * ceres::cos(abc[0]) +
                    abc[2] - T(yt_);

      // residual[0] =
      //     T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] *
      //     T(_x) + abc[2]);
      return true;
    }
    const double xt_, yt_, xw_, yw_;
  };
};
