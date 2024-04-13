// edit by dumu 2024.04.01
#include "PointCloudMapping.h"
#include <GLFW/glfw3.h>
#include <openvdb/openvdb.h>
#include <vdbfusion/VDBVolume.h>

#include <pcl/PCLPointCloud2.h>
#include <pangolin/pangolin.h>

using namespace cv;
using namespace std;
//using namespace utils;
namespace fs = std::filesystem;
using namespace vdbfusion;

namespace ORB_SLAM2 {
PointCloudMapping* PointCloudMapping::self = nullptr;
//    class Vdb;
    PointCloudMapping::PointCloudMapping(System *system,double resolution, std::string yaml_file):mpsystem(system)
    {
        self=this;
        mSystem = system;
        mResolution = resolution;

        cv::FileStorage config(yaml_file, cv::FileStorage::READ);

        mCx = config["Camera.cx"];
        mCy = config["Camera.cy"];
        mFx = config["Camera.fx"];
        mFy = config["Camera.fy"];
        mbShutdown = false;
        mbFinish = true;
        cout<<"vdbfusion initialize!"<<endl;

        // Dataset specific configuration
        openvdb::initialize();
        // 参数设置，其中
        // voxel_size_ 体素大小
        // sdf_trunc_  sdf截断距离
        // space_carving_  开启后从传感器到表面的所有体素都会被更新，否则只有在传感器前方的表面附近的体素才会被更新
        int space_carving;
        config["space_carving"] >> space_carving;
        Tsdf_Volume = make_shared<VDBVolume>(config["voxel_size"], config["sdf_trunc"],
                                                        space_carving==1);
        voxel.setLeafSize( resolution, resolution, resolution);
        statistical_filter.setMeanK(20);
        statistical_filter.setStddevMulThresh(1.0); // The distance threshold will be equal to: mean + stddev_mult * stddev

        mPointCloud = std::make_shared<PointCloud>();
        vdbfuThread = std::make_shared<std::thread>(&PointCloudMapping::vdbfu, this);  // make_unique是c++14的
        PCThread = std::make_shared<std::thread>(&PointCloudMapping::generateMeshAndPointCloud, this);
    }

    PointCloudMapping::~PointCloudMapping()
    {
        mbShutdown = true;
        vdbfuThread->join();
        PCThread->join();
    }

    void PointCloudMapping::requestFinish()
    {
        mbShutdown = true;
    }

    bool PointCloudMapping::isFinished()
    {
        return mbFinish;
    }

    void PointCloudMapping::insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth)
    {
        mSDFFrameQueue.push({color.clone(), depth.clone(), kf->GetPose(), 1.0});

        cout << "receive a keyframe, id = " << kf->mnId << endl;
    }

    void PointCloudMapping::insertcurrentFrame(Frame &fr, const cv::Mat& color, const cv::Mat& depth)
    {
        mSDFFrameQueue.push({color.clone(), depth.clone(), fr.mTcw.clone(), 1.0});
        cout << "receive currentframe, id = "<<fr.mnId << endl;
    }

    void PointCloudMapping::generateMeshAndPointCloud()
    {
        while (!mbShutdown)
        {
            auto [vertices, triangles, colors] = Tsdf_Volume->ExtractTriangleMesh(true, 0.5);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            mMutexMeshData.lock();
            mVertices = vertices;
            mTriangles = triangles;
            mColors = colors;
            // copy mPointCloud to mPointCloudMap
            mPointCloudMap = *mPointCloud;
            mMutexMeshData.unlock();
        }
    }

    void PointCloudMapping::DrawMesh()
    {
        // draw in gl
        mMutexMeshData.lock();
        glBegin(GL_TRIANGLES);
        for (size_t i = 0; i < mTriangles.size(); i++)
        {
            glColor3f(mColors[mTriangles[i].x()].x(), mColors[mTriangles[i].x()].y(), mColors[mTriangles[i].x()].z());
            glVertex3f(mVertices[mTriangles[i].x()].x(), mVertices[mTriangles[i].x()].y(), mVertices[mTriangles[i].x()].z());
            glColor3f(mColors[mTriangles[i].y()].x(), mColors[mTriangles[i].y()].y(), mColors[mTriangles[i].y()].z());
            glVertex3f(mVertices[mTriangles[i].y()].x(), mVertices[mTriangles[i].y()].y(), mVertices[mTriangles[i].y()].z());
            glColor3f(mColors[mTriangles[i].z()].x(), mColors[mTriangles[i].z()].y(), mColors[mTriangles[i].z()].z());
            glVertex3f(mVertices[mTriangles[i].z()].x(), mVertices[mTriangles[i].z()].y(), mVertices[mTriangles[i].z()].z());
        }
        glEnd();
        mMutexMeshData.unlock();
    }

    void PointCloudMapping::DrawPointCloud(double size)
    {
        mMutexMeshData.lock();
        // draw in gl
        glPointSize(size);
        glBegin(GL_POINTS);
        for (auto &p:mPointCloudMap)
        {
            glColor3f(p.r/255.0, p.g/255.0, p.b/255.0);
            glVertex3d(p.x,  p.y, p.z);
        }
        glEnd();
        mMutexMeshData.unlock();
    }

    void PointCloudMapping::saveMesh(std::string save_path, std::string save_name)
    {
        // colored 存储点云
        string pc_path = save_path+"pointcloud_"+save_name+".ply";
        string mesh_path = save_path+"mesh_"+save_name+".ply";
        pcl::io::savePLYFile(pc_path, *mPointCloud);
        cout << "Saving pointcloud to :  " << pc_path << endl;

        auto [vertices, triangles, colors] =
            Tsdf_Volume->ExtractTriangleMesh(true, 0.5);

        colormeshwrite(mesh_path,vertices,colors,triangles);
        cout<<"Saving mesh to: " << mesh_path << endl;
    }

    void PointCloudMapping::vdbfu() // 对重定位过程中普通帧进行重建
    {
        vector<Eigen::Vector3d> poses;
        Eigen::Vector3d pose_t, origin, origin_last;
        origin_last[2] = Eigen::Infinity;
        while (true)
        {

            if (mbShutdown)
            {
                break;
            }
            if (mSDFFrameQueue.empty())
            {
                continue;
            }
            static float weight = 1.0;
            int id;
            {
                SDFFrame frame = mSDFFrameQueue.wait_and_pop();
                cv::Mat colorImg = frame.color;
                cv::Mat depthImg = frame.depth;
                cv::Mat pose = frame.pose;
                // print pose
                cout << "pose: " << pose << endl;
                id = frame.id;
                weight = frame.weight;

                Eigen::Matrix4d pos = ORB_SLAM2::cvToEigenMatrix<double, float, 4, 4>(pose).inverse();
                origin = pos.block<3, 1>(0, 3);
                if (origin_last[2] == Eigen::Infinity)
                {
                    origin_last = origin;
                }
                double dis = (origin - origin_last).norm();
                cout << dis << endl;
                if (dis > 0.5)
                { // 相机运动过于剧烈，跳过该帧
                    std::cout << "SKIP FRAME" << std::endl;
                    continue;
                };
                origin_last = origin;
                std::unique_lock<std::mutex> locker(mPointCloudMtx);
                generatePointCloud(colorImg, depthImg, pose, id);
                locker.unlock();
            }
            std::unique_lock<std::mutex> locker(mPointsMtx);
            cout << "points' numbers:" << mPoints.size() << endl;
            std::chrono::steady_clock::time_point T1 = std::chrono::steady_clock::now();
            Tsdf_Volume->Integrate(mPoints, mPointcolors, origin, [](float sdf /*unused*/)
                                   { return weight; });
            Tsdf_Volume->Prune(0.5);
            std::chrono::steady_clock::time_point T2 = std::chrono::steady_clock::now();
            double T = std::chrono::duration_cast<std::chrono::duration<double>>(T2 - T1).count();
            locker.unlock();
            std::cout << "Integrate " << id << " Cost = " << T << std::endl;
        }

        mbFinish = true;
    }

    void PointCloudMapping::generatePointCloud(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose, int nId)
    {
        mPoints.clear();
        mPointcolors.clear();
        std::cout << "Converting image: " << nId << endl;
        PointCloud::Ptr current(new PointCloud);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        for(size_t v = 0; v < imRGB.rows ; v+=3){
            for(size_t u = 0; u < imRGB.cols ; u+=3){
                double d = double(imD.at<unsigned short>(v,u))/5000;
                if(d <0.01 || d>3.5){ // 深度值为0 表示测量失败
                    continue;
                }
                PointT p;
                p.z = d;
                p.x = ( u - mCx) * p.z / mFx;
                p.y = ( v - mCy) * p.z / mFy;

                p.b = imRGB.data[v*imRGB.step1()+u*imRGB.channels()];
                p.g = imRGB.data[v*imRGB.step1()+u*imRGB.channels()+1];
                p.r = imRGB.data[v*imRGB.step1()+u*imRGB.channels()+2];
                current->points.push_back(p);
            }
        }

        Eigen::Isometry3d T = Converter::toSE3Quat( pose );
        current->is_dense = true;
                // tmp为转换到世界坐标系下的点云
        PointCloud::Ptr transformed(new PointCloud);
        pcl::transformPointCloud(*current, *transformed, T.inverse().matrix());
        current->clear();
        current->swap(*transformed);

        // depth filter and statistical removal，离群点剔除
        statistical_filter.setInputCloud(current);
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(2.0);
        statistical_filter.filter(*transformed);
        current->clear();
        current->swap(*transformed);
//利用体素滤波试一下
        for(auto point : *current){
            Eigen::Vector3d P;
            openvdb::Vec3i C;
            P.x() = point.x;
            P.y() = point.y;
            P.z() = point.z;
            C.x() = point.r;
            C.y() = point.g;
            C.z() = point.b;
            mPoints.push_back(P);
            mPointcolors.push_back(C);
        }
        //点云合并
        (*mPointCloud) += *current;
        mPointCloud->is_dense = true;
        // 加入新的点云后，对整个点云进行体素滤波
        std::cout<<"size of mPointCloud is" << mPointCloud->points.size()<<std::endl;
        voxel.setInputCloud(mPointCloud);
        voxel.setLeafSize(0.05f,0.05f,0.05f);
        voxel.filter(*transformed);
        std::cout<<"size of transformed is" << transformed->points.size()<<std::endl;
        mPointCloud->swap(*transformed);
        mPointCloud->is_dense = true;
        transformed->clear();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        std::cout << ", Cost = " << t << std::endl;
    }


    void PointCloudMapping::getGlobalCloudMap(PointCloud::Ptr &outputMap)
    {
        std::unique_lock<std::mutex> locker(mPointCloudMtx);
        outputMap = mPointCloud;
    }
    void PointCloudMapping::colormeshwrite(string plyname, vector<Eigen::Vector3d>& vertices, vector<Eigen::Vector3d>& colors, vector<Eigen::Vector3i>& triangles){
        ofstream ply;
        ply.open(plyname);
        ply<<"ply"<<endl;
        ply<<"format ascii 1.0"<<endl;
        ply<<"element vertex "<<vertices.size()<<endl;
        ply<<"property float x"<<endl;
        ply<<"property float y"<<endl;
        ply<<"property float z"<<endl;
        ply<<"property uchar red"<<endl;
        ply<<"property uchar green"<<endl;
        ply<<"property uchar blue"<<endl;
        ply<<"element face "<<triangles.size()<<endl;
        ply<<"property list uchar int vertex_index"<<endl;
        ply<<"end_header"<<endl;
        for(int i=0;i<vertices.size();i++){
            ply<<vertices[i][0]<<" "<<vertices[i][1]<<" "<<vertices[i][2]<<" "<<(int)(colors[i][0]*255)<<" "<<(int)(colors[i][1]*255)<<" "<<(int)(colors[i][2]*255)<<endl;
        }
        for(int j=0;j<triangles.size();j++){
            ply<<"3 "<<triangles[j][0]<<" "<<triangles[j][1]<<" "<<triangles[j][2]<<endl;
        }
        ply.close();
    }
}