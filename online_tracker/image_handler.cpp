#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void image_fetch_callback(const sensor_msgs::CompressedImage& msg) 
{
  ROS_INFO("I saw an image with");
  cv::Mat A;
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
  A = cv_ptr->image;

  static bool saved = false;
  
  if (!saved) {
    const std::string storing_path = "/home/fabian/Desktop/pingpong.png";
    imwrite(storing_path, A);
  }

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/raspicam_node/image/compressed", 1000, image_fetch_callback);

  ros::spin();

  return 0;
}