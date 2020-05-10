#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void image_fetch_callback(const sensor_msgs::CompressedImage& msg) 
{
  cv::Mat src;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  src = cv_ptr->image;
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  cv::medianBlur(gray, gray, 5);
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 20, 100 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    ROS_INFO("Found circles: %d",int(circles.size()));
    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        circle(src, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
    }

  static bool saved = false;
  static int counter = 1;
  if (!saved && counter <= 50) {
    const std::string storing_path = "/home/fabian/Desktop/pingpong/pingpong_" + std::to_string(counter) + ".png";
    imwrite(storing_path, src);

  } else {
    counter = 0;
  }

  counter++;

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/raspicam_node/image/compressed", 1000, image_fetch_callback);

  ros::spin();

  return 0;
}