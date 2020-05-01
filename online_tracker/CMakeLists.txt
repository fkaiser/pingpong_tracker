cmake_minimum_required(VERSION 3.1)
project(PingpongTracker)

message("Building targets for Pingpong tracker")

set(OPENCV_LIB "home/fabian/opencv_build/opencv/build/lib")
# Define executable
add_executable("${PROJECT_NAME}"  track_ball.cpp)

find_library(OPENCV_LIB opencv_lib HINTS "${HOME}/opencv_build/opencv/build/lib")
target_link_libraries("${PROJECT_NAME}"
                     opencv_highgui
                     opencv_imgcodecs
                     opencv_imgproc
                     opencv_core
                     boost_filesystem
                     boost_system)

add_library(opcenv_lib SHARED IMPORTED)