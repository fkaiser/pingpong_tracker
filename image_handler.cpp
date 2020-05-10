#include "ros/ros.h"
#include "sensor_mssgs/CompressedImage.h"

void image_fetch_callback(const sensor_msgs::CompressedImage& img) 
{
  ROS_INFO("I saw an image with format [%s]", img.format);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "listener");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/raspicam_node/image_mouse_left", 1000, image_fetch_callback);

  ros::spin();

  return 0;
}