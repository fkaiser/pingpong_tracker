#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace std;
namespace fs = ::boost::filesystem;


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret)
{
    if(!fs::exists(root) || !fs::is_directory(root)) return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
        if(fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path());
        //std::cout << it->path() << std::endl;
        ++it;

    }

    std::sort(ret.begin(), ret.end());

}

int circular_hough_extraction(const string& path_name, const string& file_name, const std::string& dir_name_save,
                              const bool store_image, const bool save_images)
{
    // Loads an image
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Mat src = imread(path_name, IMREAD_COLOR );
    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        return -1;
    }
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 20, 100 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(src, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }

    if (store_image) {
        imwrite(dir_name_save + "/" + file_name, src);
    }
    if (save_images) {
        imshow("detected circles", src);
        waitKey();
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "Image load, circular extraction and store took:  " << time_span.count() << " seconds.";
    std::cout << std::endl;
    return 0;

}


static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name <<"\n"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t--save_images: \tSaves processed images\n"
              << "\t--show_images: \t Shows processed images"
              << std::endl;
}


int main(int argc, char** argv)
{
    bool save_images = false;
    bool show_images = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "--save_images") {
            save_images = true;
        } else if (arg == "--show_images") {
            show_images = true;
        }
    }
    const char* home_dir = std::getenv("HOME");
    if(home_dir) {
        std::cout << "Your HOME dir is: " << home_dir << '\n';
    } else {
        std::cout << "Your HOME is not defined.:" << '\n';
        return -1;
    }
    const string file_ending = ".png";
    const std::string dir_name_rel = "/Desktop/pingpong"; //"/Documents/images_ping_pong_tracker";
    const std::string dir_name = std::string(home_dir) + dir_name_rel;
    const std::string dir_name_save = dir_name + "/saved/";
    const char* filename = argc >=2 ? argv[1] : "testimage.png";
    vector<fs::path> image_names;
    get_all(dir_name, file_ending, image_names);
    int counter = 0;
    for (auto it = std::begin(image_names); it != std::end(image_names); ++it) {
        std::cout << it->string() << std::endl;
        circular_hough_extraction(it->string(), it->filename().string(), dir_name_save, save_images, show_images);
        if (counter >= 20) {
            break;
        }
        counter++;
    }
    

    return 0;
}