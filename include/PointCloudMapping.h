#pragma once
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <deque>
//#include <boost/make_shared.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>  // Eigen核心部分
#include <Eigen/Geometry> // 提供了各种旋转和平移的表示
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>

#include "KeyFrame.h"
#include "System.h"
#include "LoopClosing.h"
#include "Converter.h"
#include "tf_queue.h"
#include "tf_map.h"

#include <openvdb/openvdb.h>
#include <vdbfusion/VDBVolume.h>
//

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef pcl::PointXYZRGBA PointT; // A point structure representing Euclidean xyz coordinates, and the RGB color.
typedef pcl::PointCloud<PointT> PointCloud;

using namespace cv;
//using namespace utils;

namespace ORB_SLAM2 {

    class Converter;
    class KeyFrame;
    class System;
    class Tracking;
//    class Vdb;

    class PointCloudMapping {
    public:
        PointCloudMapping(System *psystem, std::string yaml_file);
        ~PointCloudMapping();

        static PointCloudMapping* GetSingleton()
        {
            return self;
        }
        bool isBusy()
        {
            std::unique_lock<std::mutex> lock(mMutexBusy);
            return mIsBusy;
        }
        void insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth); // 传入的深度图像的深度值单位已经是m
        void insertcurrentFrame(Frame &fr,const cv::Mat& color, const cv::Mat& depth); // 传入的深度图像的深度值单位已经是m
        void requestFinish();
        bool isFinished();
        void saveMesh(std::string save_path, std::string save_name);
        void vdbfu();
        void getGlobalCloudMap(PointCloud &outputMap);
        void getGlobalMeshMap(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3d> &colors, std::vector<Eigen::Vector3i> &triangles);
        void DrawMesh();
        void DrawPointCloud(double size);
        vector<Eigen::Vector3d> mPoints;
        vector<openvdb::Vec3i> mPointcolors;
        vector<vector<Eigen::Vector3d>> mvPoints;
    private:
        void generateMeshAndPointCloud();
        void generatePointCloud(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose, int nId);
        void colormeshwrite(string plyname, vector<Eigen::Vector3d>& vertices, vector<Eigen::Vector3d>& colors, vector<Eigen::Vector3i>& triangles);
        double mCx, mCy, mFx, mFy, mResolution;
        int pc_scale;
        std::vector<float> time_usage;

        pcl::visualization::PCLVisualizer::Ptr MeshViewer;
        pcl::visualization::PCLVisualizer::Ptr PCViewer;

        std::shared_ptr<std::thread>  vdbfuThread, PCThread;

        std::mutex mMutexMeshData;
        std::vector<Eigen::Vector3d> mVertices;
        std::vector<Eigen::Vector3d> mColors;
        std::vector<Eigen::Vector3i> mTriangles;
        PointCloud mPointCloudMap;
        System* mSystem;
    public:
        typedef struct {
            cv::Mat color;
            cv::Mat depth;
            cv::Mat pose;
            float weight;
            int id;
        } SDFFrame;
        threadsafe_queue<SDFFrame> mSDFFrameQueue;
    private:
        bool mbShutdown;
        bool mbFinish;
        bool mIsBusy = false;
        std::mutex mMutexBusy;

        System* mpsystem;
//        LoopClosing* mpLoopCloser;
//        Vdb* vdbf;
        bool mloop = false;

        std::mutex mPointCloudMtx;
        PointCloud::Ptr mPointCloud;

        static PointCloudMapping* self;

        // filter
        pcl::VoxelGrid<PointT> voxel;
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;

        std::mutex mPointsMtx; // 重建结果的互斥锁
        std::shared_ptr<vdbfusion::VDBVolume> Tsdf_Volume;
    };

}
