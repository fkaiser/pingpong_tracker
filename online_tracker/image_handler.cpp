#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
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
                 gray.rows/16,
                 100, 30, 20, 100);
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

// int main(int argc, char **argv)
// {
//   ros::init(argc, argv, "listener");
//   ros::NodeHandle n;
//   ros::Subscriber sub = n.subscribe("/raspicam_node/image/compressed", 1000, image_fetch_callback);
//   image_transport::Publisher image_pub = it_.advertise("/image_converter/output_video", 1);

//   ros::spin();

//   return 0;
// }

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/raspicam_node/image", 1, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
      ROS_INFO("Got raw image");
      return
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

    // Find ball and draw it
    cv::Mat src = cv_ptr->image;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                  gray.rows/16,
                  100, 30, 20, 100);
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

      // Output modified video stream
      image_pub_.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}