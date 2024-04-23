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
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <mesh_msgs/MeshGeometryStamped.h>
#include <map_msgs/PointCloud2Update.h>

#include <opencv2/core/core.hpp>
//cv2eigen
#include <opencv2/core/eigen.hpp>

#include "Osmap.h"
#include "System.h"
#include "ImageDetections.h"
#include "Utils.h"
#include "nlohmann/json.hpp"
namespace fs = std::filesystem;
using json = nlohmann::json;

void try_create_dir(const std::string &dir)
{
    if (!fs::exists(dir))
    {
        std::cout << "Creating directory " << dir << std::endl;
        if (!fs::create_directory(dir))
        {
            std::cout << "Failed to create directory" << std::endl;
            return;
        }
    }
}

std::vector<std::string> rgb_files, depth_files;
std::vector<double> timestamps, timestrack;
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;

// ros::NodeHandle *nh;
// ros::Publisher *pose_pub;
// ros::Publisher *pointcloud_pub;
// ros::Publisher *mesh_pub;
std::shared_ptr<ros::NodeHandle> nh;
std::shared_ptr<ros::Publisher> pose_pub;
std::shared_ptr<ros::Publisher> pointcloud_pub;
std::shared_ptr<ros::Publisher> mesh_pub;
std::string map_frame, camera_frame;
bool reloc_mode = false, stop = false, pauseslam = false;
std::shared_ptr<tf::TransformBroadcaster> br;

class Callback
{
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
             bool save_images) : slam(slam), detections_manager(detections_manager), osmap(osmap), output_path(output_path), data_name(data_name), save_images(save_images) {}
    void grab_rgbd(
        const sensor_msgs::ImageConstPtr &rgb_msg,
        const sensor_msgs::ImageConstPtr &depth_msg)
    {
        if (pauseslam)
        {
            return;
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        cv_bridge::CvImageConstPtr rgb_ptr = cv_bridge::toCvShare(rgb_msg);
        cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg);
        cv::Mat rgb, depth;
        cv::cvtColor(rgb_ptr->image, rgb, cv::COLOR_BGR2RGB);
        cv::Mat depth32f = depth_ptr->image;
        cv::patchNaNs(depth32f, 0);
        depth32f.convertTo(depth, CV_16UC1, 5000.0);

        if (save_images)
        {
            std::string path = output_path + "/" + data_name;
            std::string rgb_file = std::to_string(rgb_ptr->header.stamp.toSec()) + ".png";
            std::string depth_file = std::to_string(depth_ptr->header.stamp.toSec()) + ".png";
            rgb_files.push_back(rgb_file);
            depth_files.push_back(depth_file);
            cv::imwrite(
                path + "/rgb/" + rgb_file,
                rgb);
            cv::imwrite(
                path + "/depth/" + depth_file,
                depth);
        }

        if (reloc_mode)
        {
            slam->ActivateLocalizationMode();
        }

        std::vector<ORB_SLAM2::Detection::Ptr> detections = detections_manager->detect(rgb);
        cv::Mat pose = slam->TrackRGBD(rgb, depth, rgb_msg->header.stamp.toSec(), detections, false);
        if (pose.empty())
        {
            std::cout << "Failed to track frame" << std::endl;
            return;
        }
        timestamps.push_back(rgb_msg->header.stamp.toSec());
        Eigen::Matrix4d pose_eigen;
        cv::cv2eigen(pose, pose_eigen);
        poses.push_back(pose_eigen);
        /*
        know camera_link <-> map,
        assuming know base_link <-> camera_link,
        need to publish camera_link <-> map
        */
        Eigen::Matrix4d pose_inv = pose_eigen.inverse().eval();
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.stamp = rgb_msg->header.stamp;
        pose_msg.header.frame_id = map_frame;
        pose_msg.pose.position.x = pose_inv(0, 3);
        pose_msg.pose.position.y = pose_inv(1, 3);
        pose_msg.pose.position.z = pose_inv(2, 3);
        Eigen::Quaterniond q(pose_inv.block<3, 3>(0, 0));
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        pose_pub->publish(pose_msg);
        tf::Transform transform_tf;
        transform_tf.setOrigin(tf::Vector3(pose_inv(0, 3), pose_inv(1, 3), pose_inv(2, 3)));
        tf::Quaternion q_tf(q.x(), q.y(), q.z(), q.w());
        transform_tf.setRotation(q_tf);
        br->sendTransform(tf::StampedTransform(transform_tf, rgb_msg->header.stamp, camera_frame, map_frame));

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        stringstream ss;
        ss << rgb_msg->header.stamp.toSec();
        std::cout << "Tracking " << ss.str() << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        timestrack.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }
};
int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    const char *usage =
        "args:\n"
        "    1. path to vocabulary (.txt)\n"
        "    2. path to camera settings (.yaml)\n"
        "    3. detection file(.json or .onnx)\n"
        "    4. category to ignore file (.txt)\n"
        "    5. relocalization mode (string: \"points\", \"objects\" or \"points+objects\")\n"
        "    6. output path (string)\n"
        "    7. name of the saved data\n"
        "    8. save images (yes or no)\n"
        "    9. runtype ( normal or relocalization map file path)\n";
    if (argc != 10)
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
    std::string runtype = argv[9];

    try_create_dir(output_path + "/" + data_name);
    if (save_images)
    {
        try_create_dir(output_path + "/" + data_name + "/rgb");
        try_create_dir(output_path + "/" + data_name + "/depth");
    }

    vector<int> ignore_categories;
    if (ignore_path != "null")
    {
        std::ifstream ignore_file(ignore_path);
        if (!ignore_file.is_open())
        {
            std::cout << "Failed to open ignore file" << std::endl;
        }
        else
        {
            int i;
            while (ignore_file >> i)
            {
                ignore_categories.push_back(i);
            }
        }
    }
    std::shared_ptr<ORB_SLAM2::ObjectDetector> detections_manager = nullptr;
    std::string extension = get_file_extension(detection_path);
    if (extension == "onnx")
    {
        detections_manager = std::make_shared<ORB_SLAM2::ObjectDetector>(detection_path, ignore_categories);
    }
    else
    {
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
    int use_viewer;
    cv::FileStorage fsSettings(settings_path, cv::FileStorage::READ);
    use_viewer = fsSettings["visualize"];
    ORB_SLAM2::System SLAM(voc_path, settings_path, ORB_SLAM2::System::RGBD, use_viewer == 1, use_viewer == 1);
    SLAM.SetRelocalizationMode(relocalization_mode);

    ORB_SLAM2::Osmap osmap(SLAM);
    if (runtype != "normal")
    {
        osmap.mapLoad(runtype);
        SLAM.ActivateLocalizationMode();
    }

    nh = std::make_shared<ros::NodeHandle>("~");
    br = std::make_shared<tf::TransformBroadcaster>();
    nh->param<std::string>("map_frame", map_frame, "map");
    nh->param<std::string>("camera_frame", camera_frame, "camera_link");
    nh->param<bool>("reloc_mode", reloc_mode, false);
    nh->param<bool>("stop", stop, false);
    nh->param<bool>("pauseslam", pauseslam, false);

    pose_pub = std::make_shared<ros::Publisher>(nh->advertise<geometry_msgs::PoseStamped>("pose", 1));
    pointcloud_pub = std::make_shared<ros::Publisher>(nh->advertise<sensor_msgs::PointCloud2>("pointcloud", 1));
    // mesh_pub = std::make_shared<ros::Publisher>(nh->advertise<mesh_msgs::MeshGeometryStamped>("/mesh", 1));

    Callback callback(&SLAM, detections_manager.get(), &osmap, output_path, data_name, save_images);
    sleep(3);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(*nh, "/camera/rgb/image_color", 100);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(*nh, "/camera/depth/image", 100);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&Callback::grab_rgbd, &callback, _1, _2));

    PointCloud pc;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3d> colors;
    std::vector<Eigen::Vector3i> triangles;

    while (!stop)
    {
        ORB_SLAM2::PointCloudMapping::GetSingleton()->getGlobalCloudMap(pc);
        // ORB_SLAM2::PointCloudMapping::GetSingleton()->getGlobalMeshMap(vertices, colors, triangles);
        auto now = ros::Time::now();
        if (pc.size() > 0)
        {
            sensor_msgs::PointCloud2 pc_msg;
            pcl::PCLPointCloud2 pcl_pc;
            pcl::toPCLPointCloud2(pc, pcl_pc);
            pcl_conversions::fromPCL(pcl_pc, pc_msg);
            pc_msg.header.frame_id = map_frame;
            pc_msg.header.stamp = now;
            pointcloud_pub->publish(pc_msg);
        }
        // if(vertices.size() > 0)
        // {
        //     mesh_msgs::MeshGeometryStamped mesh_msg;
        //     mesh_msg.header.frame_id = map_frame;
        //     mesh_msg.header.stamp =  now;
        //     mesh_msg.mesh_geometry.faces = triangles;
        //     mesh_msg.mesh_geometry.vertices = vertices;
        //     mesh_pub->publish(mesh_msg);
        // }
        nh->getParam("map_frame", map_frame);
        nh->getParam("reloc_mode", reloc_mode);
        nh->getParam("camera_frame", camera_frame);
        nh->getParam("stop", stop);
        nh->getParam("pauseslam", pauseslam);

        ros::spinOnce();
    }
    SLAM.Shutdown();

    std::ofstream trajectory_file(
        output_path + "/" + data_name + "/CameraTrajectory.txt", std::ios::trunc);
    std::ofstream trajectory_json_file(
        output_path + "/" + data_name + "/CameraTrajectory.json", std::ios::trunc);

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

    if (save_images)
    {
        std::ofstream rgb_outfile;
        rgb_outfile.open(output_path + "/" + data_name + "/rgb.txt", std::ios::trunc);
        if (!rgb_outfile.is_open())
        {
            std::cout << "Failed to open rgb.txt" << std::endl;
            return -1;
        }
        rgb_outfile << "# color images\n"
                    << "# timestamp filename\n";
        for (std::string line : rgb_files)
        {
            rgb_outfile << line << "\n";
        }
        rgb_outfile.close();

        std::ofstream depth_outfile;
        depth_outfile.open(output_path + "/" + data_name + "/depth.txt", std::ios::trunc);
        if (!depth_outfile.is_open())
        {
            std::cout << "Failed to open depth.txt" << std::endl;
            return -1;
        }
        depth_outfile << "# depth images\n"
                      << "# timestamp filename\n";
        for (std::string line : depth_files)
        {
            depth_outfile << line << "\n";
        }
        depth_outfile.close();
    }
    ros::shutdown();
    return 0;
}