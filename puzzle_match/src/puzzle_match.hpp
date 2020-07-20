#include <ros/ros.h>
#include <stdio.h>

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
using namespace cv;
using namespace std;

//! stereo camera match
class PuzzleMatch {
 public:
  PuzzleMatch();

  // ~PuzzleMatch();
  Mat scene, background, full;
  vector<double> dist;
  int feature_method = 1;
  //! Intrinsic parameters
  Mat M1, D1, M2, D2;
  Mat img_l, img_r;
  //! Extrinsic parameters
  Mat R, T, R1, P1, R2, P2;
  //! disparity map and depth map
  cv::Mat disp, disp8, depth;
  Mat pic_l, pic_r;

  typedef struct {
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
  } four_corners_t;

  four_corners_t corners;
  cv::Mat1f camera_matrix;
  Mat distortion_coefficients;
  Mat transH;
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
};
