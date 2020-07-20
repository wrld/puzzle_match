#include "puzzle_match.hpp"
enum {
  STEREO_BM = 0,
  STEREO_SGBM = 1,
  STEREO_HH = 2,
  STEREO_VAR = 3,
  STEREO_3WAY = 4
};
PuzzleMatch::PuzzleMatch() {}

void PuzzleMatch::Init(Mat scene, Mat background, Mat full) {
  this->scene = scene;
  this->background = background;
  this->full = full;
  float f = 1, w = scene.cols, h = scene.rows;

  camera_matrix = (cv::Mat1f(3, 3) << f, 0, w / 2, 0, f, h / 2, 0, 0, 1);
}

void PuzzleMatch::find_feature_matches(const Mat& img_1, const Mat& img_2,
                                       std::vector<KeyPoint>& keypoints_1,
                                       std::vector<KeyPoint>& keypoints_2,
                                       std::vector<DMatch>& matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  //   Ptr<FeatureDetector> detector = ORB::create();
  //   Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<FeatureDetector> detector = ORB::create();

  Ptr<DescriptorExtractor> descriptor = ORB::create();

  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离,
  //即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void PuzzleMatch::pose_estimation_2d2d(const std::vector<KeyPoint>& keypoints_1,
                                       const std::vector<KeyPoint>& keypoints_2,
                                       const std::vector<DMatch>& matches,
                                       Mat& R, Mat& t) {
  // 相机内参,TUM Freiburg2
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

  //-- 计算本质矩阵
  Point2d principal_point(325.1, 249.7);  //相机主点, TUM dataset标定值
  int focal_length = 521;                 //相机焦距, TUM dataset标定值
  Mat essential_matrix;
  essential_matrix =
      findEssentialMat(points1, points2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  //-- 计算单应矩阵
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3);
  cout << "homography_matrix is " << endl << homography_matrix << endl;
  this->transH = homography_matrix;
  //-- 从本质矩阵中恢复旋转和平移信息.
  recoverPose(essential_matrix, points1, points2, R, t, focal_length,
              principal_point);
  this->R = R;
  this->T = t;
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
}

//优化两图的连接处，使得拼接自然
void PuzzleMatch::OptimizeSeam(Mat& img1, Mat& trans, Mat& dst) {
  int start = MIN(corners.left_top.x,
                  corners.left_bottom.x);  //开始位置，即重叠区域的左边界

  double processWidth = img1.cols - start;  //重叠区域的宽度
  int rows = dst.rows;
  int cols = img1.cols;  //注意，是列数*通道数
  double alpha = 1;      // img1中像素的权重
  for (int i = 0; i < rows; i++) {
    uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
    uchar* t = trans.ptr<uchar>(i);
    uchar* d = dst.ptr<uchar>(i);
    for (int j = start; j < cols; j++) {
      //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
      if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0) {
        alpha = 1;
      } else {
        // img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
        alpha = (processWidth - (j - start)) / processWidth;
      }

      d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
      d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
      d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
    }
  }
}

void PuzzleMatch::featurematch(Mat src1, Mat src2, vector<KeyPoint> keypoints2,
                               Mat descriptors2, Rect roi, Rect roi_full) {
  // src1:right src2:left
  vector<DMatch> matches;
  vector<vector<DMatch>> knn_matches;
  // 1 初始化特征点和描述子,ORB

  Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 1, 0.04, 10, 1.6);
  vector<KeyPoint> keypoints1;
  Mat descriptors1;
  // 2 提取 Oriented FAST 特征点
  sift->detect(src1, keypoints1);
  //   sift->detect(src2, keypoints2);
  //   drawKeypoints(src2, keypoints2, src2);
  //   imshow("draw", src2);
  //   waitKey(0);
  // 3 根据角点位置计算 BRIEF 描述子
  cout << keypoints1.size() << " " << keypoints2.size() << endl;
  sift->compute(src1, keypoints1, descriptors1);
  //   sift->compute(src2, keypoints2, descriptors2);

  // 4 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离

  BFMatcher bfmatcher(NORM_L2);
  //   bfmatcher.match(descriptors1, descriptors2, matches);
  bfmatcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

  // 5 匹配对筛选
  double min_dist = 1000, max_dist = 0;
  vector<DMatch> good_matches;
  // 找出所有匹配之间的最大值和最小值
  for (int i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance < 0.5 * knn_matches[i][1].distance)
      good_matches.push_back(knn_matches[i][0]);
    // if (dist < min_dist) min_dist = dist;
    // if (dist > max_dist) max_dist = dist;
  }
  // 当描述子之间的匹配大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
  // 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
  //   cout << min_dist << endl;

  //   for (int i = 0; i < descriptors1.rows; i++) {
  //     if (matches[i].distance <= max(3 * min_dist, 100.0))
  //       good_matches.push_back(matches[i]);
  //   }
  matches.clear();
  matches = good_matches;
  Mat img;
  drawMatches(src1, keypoints1, src2, keypoints2, matches, img);
  //   resize(img, img, Size(img.cols * 0.2, img.rows * 0.2));
  //   imshow("match1", img);
  //   waitKey(2000);

  vector<Point2f> pic1, pic2;  //滤掉误匹配点
  for (int i = 0; i < matches.size(); i++) {
    pic1.push_back(Point2f(keypoints1[matches[i].queryIdx].pt.x + roi.tl().x,
                           keypoints1[matches[i].queryIdx].pt.y + roi.tl().y));
    pic2.push_back(
        Point2f(keypoints2[matches[i].trainIdx].pt.x + roi_full.tl().x,
                keypoints2[matches[i].trainIdx].pt.y + roi_full.tl().y));
  }
  vector<unsigned char> mark(pic1.size());
  //   for (auto p : pic2) circle(this->full, p, 1, Scalar(0, 255, 0));
  //   for (auto p : pic1) circle(this->scene, p, 1, Scalar(0, 255, 0));
  //   imshow("full", this->full);
  //   imshow("scene", this->scene);
  //   waitKey(0);
  // find homography for right to left
  transH = findHomography(pic1, pic2, CV_RANSAC, 5, mark, 500);
  // find essential mat for right to left
  Mat E = cv::findEssentialMat(pic1, pic2, camera_matrix, CV_RANSAC);
  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(E, R1, R2, t);
  cout << "camera_matrix" << camera_matrix << endl;
  decomposeHomographyMat(transH, camera_matrix, R1, t, noArray());
  this->R = R1.clone();
  this->T = t.clone();
  Mat origin_angle = Mat::zeros(3, 1, CV_32FC1);
  Mat RR = Mat::zeros(3, 3, CV_32FC1);

  origin_angle.at<float>(0, 0) = 1;
  origin_angle.at<float>(1, 0) = 0;
  origin_angle.at<float>(2, 0) = 0;
  RR.at<float>(0, 0) = R.at<Vec3b>(0, 0)[0];
  RR.at<float>(0, 1) = R.at<Vec3b>(0, 1)[0];
  RR.at<float>(0, 2) = R.at<Vec3b>(0, 2)[0];
  RR.at<float>(1, 0) = R.at<Vec3b>(1, 0)[0];
  RR.at<float>(1, 1) = R.at<Vec3b>(1, 1)[0];
  RR.at<float>(1, 2) = R.at<Vec3b>(1, 2)[0];
  RR.at<float>(2, 0) = R.at<Vec3b>(2, 0)[0];
  RR.at<float>(2, 1) = R.at<Vec3b>(2, 1)[0];
  RR.at<float>(2, 2) = R.at<Vec3b>(2, 2)[0];
  Mat target_vector;
  target_vector = RR * origin_angle;
  cout << "origin_angle" << origin_angle << endl;
  cout << "target_vector" << target_vector << endl;
  cout << "transH" << transH << endl;
  cout << "R" << R << endl;
  //   cout << "yaw" << yaw / CV_PI * 180 << endl;
  cout << "T" << T << endl;
  // pose_estimation_2d2d(keypoints1, keypoints2, matches, R1, t);
}

void PuzzleMatch::stitchImage(Mat src1, Mat src2) {
  Mat tempP, dst1, dst2;

  Mat matchP = src2.clone();
  warpPerspective(scene, tempP, transH, src2.size());
  matchP = matchP + tempP;
  resize(matchP, matchP, Size(matchP.cols * 0.6, matchP.rows * 0.6));
  imshow("result", matchP);
  waitKey(2000);
  //   imwrite("/home/gjx/ROS/puzzle_match/src/puzzle_match/result/result.jpg",
  //           matchP);
}

vector<Rect> PuzzleMatch::getROI(Mat image, bool show) {
  Mat backsub_res, gray, threshold;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  vector<Rect> bbox;
  Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
  Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
  subtract(image, background, backsub_res);
  cvtColor(backsub_res, gray, COLOR_BGR2GRAY);
  cv::threshold(gray, threshold, 10, 255, THRESH_BINARY);
  //   adaptiveThreshold(gray, threshold, 10, 255, THRESH_BINARY);
  medianBlur(threshold, threshold, 3);
  morphologyEx(threshold, threshold, MORPH_CLOSE, kernel);
  morphologyEx(threshold, threshold, MORPH_OPEN, kernel2);
  findContours(threshold, contours, hierarchy, RETR_EXTERNAL,
               CHAIN_APPROX_SIMPLE);
  for (auto con : contours) {
    if (contourArea(con) > 1000) {
      bbox.push_back(boundingRect(con));
      //   rectangle(image, bbox.back(), Scalar(0, 255, 0), 1);
    }
  }
  //   if (show) imshow("roi_result", image);
  return bbox;
}