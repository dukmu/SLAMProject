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

    if (argc != 6)
    {
        cerr << endl
             << "Usage:\n"
                " ./main\n"
                "      camera_file\n"
                "      output folder\n"
                "      output_name \n"
                "      relocalization map file path\n"
                "      vocabulary_file\n";
        return 1;
    }

    std::string parameters_file = string(argv[1]);
    string output_folder = string(argv[2]);
    string output_name = string(argv[3]);
    string map = string(argv[4]);
    string vocabulary_file = string(argv[5]);

    if (output_folder.back() != '/')
        output_folder += "/";
    fs::create_directories(output_folder);
    fs::create_directories(output_folder + "/" + output_name);

    // Relocalization mode
    ORB_SLAM2::enumRelocalizationMode relocalization_mode = ORB_SLAM2::RELOC_OBJECTS_POINTS;

    // Create system
    ORB_SLAM2::System::eSensor sensor = ORB_SLAM2::System::RGBD;
    cv::FileStorage fSettings(parameters_file, cv::FileStorage::READ);
    ORB_SLAM2::System SLAM(vocabulary_file, parameters_file, sensor, false, false, false);
    SLAM.SetRelocalizationMode(relocalization_mode);

    ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(SLAM);
    osmap.mapLoad(map);
    SLAM.Shutdown();
    Eigen::Matrix3d K = SLAM.mpTracker->GetK();
    
    // evaluate ellipsoid
    std::vector<ORB_SLAM2::MapObject*> objects = SLAM.mpMap->GetAllMapObjects();
    typedef std::tuple<unsigned int, unsigned int,
        ORB_SLAM2::Ellipse, ORB_SLAM2::Ellipse> Obj_Base;
    std::vector<Obj_Base> objbases;
    for (ORB_SLAM2::MapObject* obj : objects)
    {
        unsigned int cat = obj->GetTrack()->GetCategoryId();
        unsigned int id = obj->GetTrack()->GetId();
        ORB_SLAM2::Ellipsoid ellipsoid = obj->GetEllipsoid();
        auto [bboxes, Rts, scores] = obj->GetTrack()->CopyDetectionsInKeyFrames();
        for (int i = 0; i < bboxes.size(); i++)
        {
            ORB_SLAM2::Ellipse ellipse = ORB_SLAM2::Ellipse::FromBbox(bboxes[i]);
            ORB_SLAM2::Ellipse ellipse_proj = ellipsoid.project(K*Rts[i]);
            objbases.push_back({cat, id, ellipse, ellipse_proj});
        }
    }
    std::vector<double> errors;
    errors.reserve(objbases.size());
    for(auto [cat, id, ellipse, ellipse_proj] : objbases)
    {
        double error = ORB_SLAM2::gaussian_wasserstein_2d(ellipse, ellipse_proj);
        error = 1-std::exp(-std::sqrt(error)/100);
        errors.push_back(error);
    }

    // save results
    std::ofstream file(output_folder + "/" + output_name + "/ellipsoid_errors.txt");
    for (int i = 0; i < objbases.size(); i++)
    {
        auto [cat, id, ellipse, ellipse_proj] = objbases[i];
        // error: .4f
        file << cat << " " << id << " " << std::setprecision(4) << errors[i] << std::endl;
        printf("Error: %.4f\n", errors[i]);
    }
    file.close();
}