cmake_minimum_required(VERSION 3.1)
project(PingpongTracker)

message("Building targets for Pingpong tracker")
set(OPENCV_LIB "home/fabian/opencv_build/opencv/build/lib")
# Define executable
add_executable("${PROJECT_NAME}"  track_ball.cpp)
find_library(OPENCV_LIB opencv_lib HINTS home/fabian/opencv_build/opencv/build/lib)
target_link_libraries("${PROJECT_NAME}"
                     opencv_highgui
                     opencv_imgcodecs
                     opencv_imgproc
                     opencv_core
                     boost_filesystem
                     boost_system)

add_library(opcenv_lib SHARED IMPORTED)


#g++ track_ball.cpp -L/home/fabian/opencv_build/opencv/build/lib -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lboost_filesystem -lboost_system -o online_tracker 