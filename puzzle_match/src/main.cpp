#include "puzzle_match.hpp"
int main(int argc, char** argv) {
  // Mat src = imread("/home/gjx/opencv/open/stereo_camera/2.jpg");
  // Mat distortion = src.clone();
  ros::init(argc, argv, "puzzle_match");
  Mat camera_matrix = Mat(3, 3, CV_32FC1);
  Mat distortion_coefficients;

  clock_t startTime, endTime;

  startTime = clock();
  vector<Mat> srcs;
  // left
  Mat scene = imread(
      "/home/gjx/ROS/puzzle_match/picture-puzzle-matching/images2/"
      "scene_1.jpeg");
  // right
  Mat full = imread(
      "/home/gjx/ROS/puzzle_match/picture-puzzle-matching/images2/full.jpeg");
  Mat background = imread(
      "/home/gjx/ROS/puzzle_match/picture-puzzle-matching/images2/"
      "background.jpeg");

  Mat img_match;
  if (scene.data == NULL || full.data == NULL) {
    cout << "No exist" << endl;
    return -1;
  }
  PuzzleMatch st;
  cout << "param init" << endl;
  st.Init(scene, background, full);
  st.box_target = st.getROI(full, 1)[0];
  full = full(st.box_target);
  Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 1, 0.04, 10, 1.6);

  sift->detect(full, st.keypoints);
  sift->compute(full, st.keypoints, st.descriptors);
  st.mainloop();
}