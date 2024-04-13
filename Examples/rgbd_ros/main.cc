#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <string>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "Osmap.h"
#include "System.h"
#include "ImageDetections.h"
#include "Utils.h"
#include "nlohmann/json.hpp"
namespace fs = std::filesystem;
using json = nlohmann::json;

void try_create_dir(const std::string& dir)
{
    if(!fs::exists(dir))
    {
        std::cout << "Creating directory " << dir << std::endl;
        if(!fs::create_directory(dir))
        {
            std::cout << "Failed to create directory" << std::endl;
            return;
        }
    }
}

std::vector<std::string> rgb_files, depth_files;
std::vector<double> timestamps, timestrack;
std::vector<Eigen::Matrix4d> poses;
class Callback{
private:
    ORB_SLAM2::System *slam;
    ORB_SLAM2::ObjectDetector *detections_manager;
    ORB_SLAM2::Osmap *osmap;
    std::string output_path;
    std::string data_name;
    bool save_images;
public:
    Callback(ORB_SLAM2::System *slam,
                ORB_SLAM2::ObjectDetector *detections_manager,
                ORB_SLAM2::Osmap *osmap,
                const std::string &output_path,
                const std::string &data_name,
                bool save_images
        ) : slam(slam), detections_manager(detections_manager), osmap(osmap), output_path(output_path), data_name(data_name), save_images(save_images) {}
    void grab_rgbd(
        const sensor_msgs::ImageConstPtr &rgb_msg,
        const sensor_msgs::ImageConstPtr &depth_msg
    ){
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        cv_bridge::CvImageConstPtr rgb_ptr = cv_bridge::toCvShare(rgb_msg);
        cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg);
        cv::Mat rgb;
        cv::cvtColor(rgb_ptr->image, rgb, cv::COLOR_BGR2RGB);

        if(save_images){
            std::string path = output_path + "/" + data_name;
            std::string rgb_file = std::to_string(rgb_ptr->header.stamp.toSec())+".png";
            std::string depth_file = std::to_string(depth_ptr->header.stamp.toSec())+".png";
            rgb_files.push_back(rgb_file);
            depth_files.push_back(depth_file);
            cv::imwrite(
                path + "/rgb/" + rgb_file,
                rgb);
            cv::imwrite(
                path + "/depth/" + depth_file,
                depth_ptr->image);
        }
        std::vector<ORB_SLAM2::Detection::Ptr> detections = detections_manager->detect(rgb);
        cv::Mat pose = slam->TrackRGBD(rgb, depth_ptr->image, rgb_msg->header.stamp.toSec(), detections, false);
        timestamps.push_back(rgb_msg->header.stamp.toSec());
        Eigen::Matrix4d pose_eigen = ORB_SLAM2::cvToEigenMatrix<double, double, 4, 4>(pose);
        poses.push_back(pose_eigen);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Tracking " << rgb_msg->header.stamp.toSec() << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        timestrack.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
};
int main(int argc, char** argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    const char *usage = \
    "args:\n"\
    "    1. path to vocabulary (.txt)\n"\
    "    2. path to camera settings (.yaml)\n"\
    "    3. detection file(.json or .onnx)\n"\
    "    4. category to ignore file (.txt)\n"\
    "    5. relocalization mode (string: \"points\", \"objects\" or \"points+objects\")\n"\
    "    6. output path (string)\n"\
    "    7. name of the saved data\n"\
    "    8. save images (yes or no)\n"\
    "    9. use system viewer (yes or no)\n"\
    "    10. runtype ( normal or relocalization map file path)\n";
    if(argc != 11)
    {
        std::cout << usage;
        ros::shutdown();
        return 1;
    }

    std::string voc_path = argv[1];
    std::string settings_path = argv[2];
    std::string detection_path = argv[3];
    std::string ignore_path = argv[4];
    std::string relocalization = argv[5];
    std::string output_path = argv[6];
    std::string data_name = argv[7];
    bool save_images = (std::string(argv[8]) == "yes");
    bool use_viewer = (std::string(argv[9]) == "yes");
    std::string runtype = argv[10];

    try_create_dir(output_path + "/" + data_name);
    if(save_images){
        try_create_dir(output_path + "/" + data_name + "/rgb");
        try_create_dir(output_path + "/" + data_name + "/depth");
    }

    vector<int> ignore_categories;
    if(ignore_path != "null")
    {
        std::ifstream ignore_file(ignore_path);
        if(!ignore_file.is_open())
        {
            std::cout << "Failed to open ignore file" << std::endl;
        }else{
            std::string line;
            while(std::getline(ignore_file, line))
            {
                ignore_categories.push_back(std::stoi(line));
            }
        }
    }
    std::shared_ptr<ORB_SLAM2::ObjectDetector> detections_manager = nullptr;
    std::string extension = get_file_extension(detection_path);
    if(extension == "onnx")
    {
        detections_manager = std::make_shared<ORB_SLAM2::ObjectDetector>(detection_path, ignore_categories);
    }else{
        std::cout << "Invalid detection file: << " << detection_path << std::endl;
        return -1;
    }

    ORB_SLAM2::enumRelocalizationMode relocalization_mode;
    if (relocalization == std::string("points"))
        relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    else if (relocalization == std::string("objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS;
    else if (relocalization == std::string("points+objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS_POINTS;
    else
    {
        std::cerr << "Error: Invalid parameter for relocalization mode. "
                     "It should be 'points', 'objects' or 'points+objects'.\n";
        return 1;
    }

    ORB_SLAM2::System SLAM(voc_path, settings_path, ORB_SLAM2::System::RGBD, use_viewer, use_viewer);
    SLAM.SetRelocalizationMode(relocalization_mode);

    ORB_SLAM2::Osmap osmap(SLAM);
    if (runtype != "normal")
    {
        osmap.mapLoad(runtype);
        SLAM.ActivateLocalizationMode();
    }

    ros::NodeHandle nh;
    Callback callback(&SLAM, detections_manager.get(), &osmap, output_path, data_name, save_images);
    sleep(3);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&Callback::grab_rgbd, &callback, _1, _2));

    while(!SLAM.ShouldQuit())
    {
        ros::spinOnce();
    }
    SLAM.Shutdown();

    std::ofstream trajectory_file(
        output_path + "/" + data_name + "/CameraTrajectory.txt", std::ios::trunc
    );
    std::ofstream trajectory_json_file(
        output_path + "/" + data_name + "/CameraTrajectory.json", std::ios::trunc
    );

    json json_data;
    for (unsigned int i = 0; i < poses.size(); ++i)
    {
        Eigen::Matrix4d m = poses[i];
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = m.block<3, 3>(0, 0).transpose();
        pose.block<3, 1>(0, 3) = -m.block<3, 3>(0, 0).transpose() * m.block<3, 1>(0, 3);

        json R({{m(0, 0), m(0, 1), m(0, 2)},
                {m(1, 0), m(1, 1), m(1, 2)},
                {m(2, 0), m(2, 1), m(2, 2)}});
        json t({m(0, 3), m(1, 3), m(2, 3)});
        json image_data;
        image_data["file_name"] = rgb_files[i];
        image_data["R"] = R;
        image_data["t"] = t;
        json_data.push_back(image_data);

        auto q = Eigen::Quaterniond(pose.block<3, 3>(0, 0));
        auto p = pose.block<3, 1>(0, 3);
        trajectory_file << std::fixed << timestamps[i] << " " << p[0] << " " << p[1] << " " << p[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    trajectory_file.close(); // TUM format
    trajectory_json_file << json_data;
    trajectory_json_file.close();

    SLAM.SaveKeyFrameTrajectoryJSON(output_path + "/" + data_name + "/KeyFrameTrajectory.json", rgb_files);
    SLAM.SaveKeyFrameTrajectoryTUM(output_path + "/" + data_name + "/KeyFrameTrajectory.txt");
    SLAM.SaveMapObjectsOBJ(output_path + "/" + data_name + "/MapObjects.obj");
    ORB_SLAM2::PointCloudMapping::GetSingleton()->saveMesh(output_path + "/" + data_name + "/", "tsdf");

    osmap.mapSave(output_path + "/" + data_name + "/Map.osmap");

    std::sort(timestrack.begin(), timestrack.end());
    std::cout << "Average tracking time: " << std::accumulate(timestrack.begin(), timestrack.end(), 0.0) / timestrack.size() << "ms" << std::endl;
    std::cout << "Fastest tracking time: " << timestrack.front() << "ms" << std::endl;
    std::cout << "Slowest tracking time: " << timestrack.back() << "ms" << std::endl;
    

    if(save_images)
    {
        std::ofstream rgb_outfile;
        rgb_outfile.open(output_path + "/" + data_name + "/rgb.txt", std::ios::trunc);
        if(!rgb_outfile.is_open())
        {
            std::cout << "Failed to open rgb.txt" << std::endl;
            return -1;
        }
        rgb_outfile << "# color images\n" << "# timestamp filename\n";
        for(std::string line : rgb_files)
        {
            rgb_outfile << line << "\n";
        }
        rgb_outfile.close();

        std::ofstream depth_outfile;
        depth_outfile.open(output_path + "/" + data_name + "/depth.txt", std::ios::trunc);
        if(!depth_outfile.is_open())
        {
            std::cout << "Failed to open depth.txt" << std::endl;
            return -1;
        }
        depth_outfile << "# depth images\n" << "# timestamp filename\n";
        for(std::string line : depth_files)
        {
            depth_outfile << line << "\n";
        }
        depth_outfile.close();
    }
    ros::shutdown();
    return 0;
}