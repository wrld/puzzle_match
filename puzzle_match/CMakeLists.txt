cmake_minimum_required(VERSION 3.0.2)
project(puzzle_match)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  pm_msg
  cv_bridge
  image_transport
)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp
                   sensor_msgs
                   
 
  )
  if(NOT WIN32)
  ADD_DEFINITIONS("-std=c++11")
endif(NOT WIN32)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${puzzle_match_SOURCE_DIR} include
)
add_executable(puzzle_match
  src/main.cpp
  src/puzzle_match.cpp
 )
target_link_libraries(puzzle_match
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES})
catkin_package()