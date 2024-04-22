/**
 * This file is part of OA-SLAM.
 *
 * Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
 * (Inria, LORIA, Université de Lorraine)
 * OA-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OA-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <stdlib.h> /* srand, rand */

#include <opencv2/core/core.hpp>

#include <ImageDetections.h>
#include <System.h>
#include "Osmap.h"
#include <nlohmann/json.hpp>
#include <experimental/filesystem>
#include "Utils.h"
#include "dataset.hpp"
#include "PointCloudMapping.h"

using json = nlohmann::json;

namespace fs = std::experimental::filesystem;

using namespace std;

int main(int argc, char **argv)
{
    srand(time(nullptr));
    std::cout << "C++ version: " << __cplusplus << std::endl;

    if (argc != 10)
    {
        cerr << endl
             << "Usage:\n"
                " ./oa-slam\n"
                "      vocabulary_file\n"
                "      camera_file\n"
                "      path_to_associate_file\n"
                "      detections_file (.json file with detections or .onnx yolov5 weights)\n"
                "      categories_to_ignore_file (file containing the categories to ignore (one category_id per line))\n"
                "      relocalization_mode ('points', 'objects' or 'points+objects')\n"
                "      output folder\n"
                "      output_name \n"
                "      runtype ( normal or relocalization map file path)\n";
        return 1;
    }

    std::string vocabulary_file = string(argv[1]);
    std::string parameters_file = string(argv[2]);
    string path_to_associate = string(argv[3]);
    std::string detections_file(argv[4]);
    std::string categories_to_ignore_file(argv[5]);
    string reloc_mode = string(argv[6]);
    string output_folder = string(argv[7]);
    string output_name = string(argv[8]);
    string runtype = string(argv[9]);

    if (output_folder.back() != '/')
        output_folder += "/";
    fs::create_directories(output_folder);
    fs::create_directories(output_folder + "/" + output_name);

    // Load categories to ignore
    vector<int> classes_to_ignore;
    if (categories_to_ignore_file != "null")
    {
        std::ifstream fin(categories_to_ignore_file);
        if (!fin.is_open())
        {
            std::cout << "Warning !! Failed to open the file with ignore classes. No class will be ignore.\n";
        }
        else
        {
            int cat;
            while (fin >> cat)
            {
                std::cout << "Ignore category: " << cat << "\n";
                classes_to_ignore.push_back(cat);
            }
        }
    }

    // Load object detections
    auto extension = get_file_extension(detections_file);
    std::shared_ptr<ORB_SLAM2::ImageDetectionsManager> detector = nullptr;
    bool detect_from_file = false;
    if (extension == "onnx")
    { // load network
        detector = std::make_shared<ORB_SLAM2::ObjectDetector>(detections_file, classes_to_ignore);
        detect_from_file = false;
    }
    else if (extension == "json")
    { // load from external detections file
        detector = std::make_shared<ORB_SLAM2::DetectionsFromFile>(detections_file, classes_to_ignore);
        detect_from_file = true;
    }
    else
    {
        std::cout << "Invalid detection file. It should be .json or .onnx\n"
                     "No detections will be obtained.\n";
    }

    // Relocalization mode
    ORB_SLAM2::enumRelocalizationMode relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    if (reloc_mode == string("points"))
        relocalization_mode = ORB_SLAM2::RELOC_POINTS;
    else if (reloc_mode == std::string("objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS;
    else if (reloc_mode == std::string("points+objects"))
        relocalization_mode = ORB_SLAM2::RELOC_OBJECTS_POINTS;
    else
    {
        std::cerr << "Error: Invalid parameter for relocalization mode. "
                     "It should be 'points', 'objects' or 'points+objects'.\n";
        return 1;
    }

    // Load images
    TUMDataset dataset(path_to_associate);
    dataset.init();
    int nImages = dataset.get_max_image_index();
    nImages = dataset.get_max_image_index();
    cout << "Loaded " << dataset.get_max_image_index() << " images\n";

    // Create system
    ORB_SLAM2::System::eSensor sensor = ORB_SLAM2::System::RGBD;
    cv::FileStorage fSettings(parameters_file, cv::FileStorage::READ);
    int visualize = fSettings["visualize"];
    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, sensor, visualize==1, visualize==1, false);
    SLAM.SetRelocalizationMode(relocalization_mode);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.reserve(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);
    
    if (runtype != "normal")
    {
        osmap.mapLoad(runtype);
        SLAM.ActivateLocalizationMode();
    }
    // Main loop
    cv::Mat im;
    cv::Mat imDepth;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    poses.reserve(nImages);
    std::vector<std::string> filenames;
    filenames.reserve(nImages);
    std::vector<double> timestamps;
    timestamps.reserve(nImages);
    Frame frame;
    double timestamp=0;
    while (1)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::string filename;
        if (!dataset.HasNext())
            break;
        timestamp = dataset.NextFrame(frame);
        im = frame[0];
        imDepth = frame[1];
        filename = dataset.get_filename();
        timestamps.push_back(timestamp);
        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image: "
                 << filename << endl;
            return 1;
        }
        filenames.push_back(filename);

        // Get object detections
        std::vector<ORB_SLAM2::Detection::Ptr> detections;
        if (detector)
        {
            if (detect_from_file)
                detections = detector->detect(filename); // from detections file
            else
                detections = detector->detect(im); // from neural network
        }

        // Pass the image and detections to the SLAM system
        cv::Mat m = SLAM.TrackRGBD(im, imDepth, timestamp, detections, false);

        if (m.rows && m.cols)
            poses.push_back(ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(m));
        else
            poses.push_back(Eigen::Matrix4d::Identity());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack.push_back(ttrack);
        std::cout << "time = " << ttrack << "\n";

        if (SLAM.ShouldQuit())
            break;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera tracjectory

    // TXT files
    std::ofstream file(output_folder + "/" + output_name + "/CameraTrajectory.txt");
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
        image_data["file_name"] = filenames[i];
        image_data["R"] = R;
        image_data["t"] = t;
        json_data.push_back(image_data);

        auto q = Eigen::Quaterniond(pose.block<3, 3>(0, 0));
        auto p = pose.block<3, 1>(0, 3);
        file << std::fixed << timestamps[i] << " " << p[0] << " " << p[1] << " " << p[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    file.close();

    // JSON files
    std::ofstream json_file(output_folder + "/" + output_name + "/CameraTrajectory.json");
    json_file << json_data;
    json_file.close();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    std::cout << "Average tracking time: " << accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0) / vTimesTrack.size() << "\n";
    std::cout << "Fastest tracking time: " << vTimesTrack.front() << "\n";
    std::cout << "Slowest tracking time: " << vTimesTrack.back() << "\n";

    // Save camera trajectory, points and objects
    SLAM.SaveKeyFrameTrajectoryTUM(output_folder + "/" + output_name + "/KeyFrameTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryJSON(output_folder + "/" + output_name + "/KeyFrameTrajectory.json", filenames);
    SLAM.SaveMapPointsOBJ(output_folder + "/" + output_name + "/MapPoints.obj");
    SLAM.SaveMapObjectsOBJ(output_folder + "/" + output_name + "/MapObjects.obj");
    ORB_SLAM2::PointCloudMapping::GetSingleton()->saveMesh(output_folder + "/" + output_name+"/", output_name);
    std::cout << "\n";

    // Save a reloadable map
    osmap.mapSave(output_folder + "/" + output_name + "/Map.osmap");

    return 0;
}
