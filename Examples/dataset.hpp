#pragma once
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdlib.h> /* srand, rand */

#include <opencv2/core/core.hpp>
typedef std::vector<cv::Mat> Frame;
class Dataset
{
public:
    virtual void init() = 0;
    virtual double NextFrame(Frame &f) = 0;
    virtual bool HasNext() = 0;
};

class Fr2DeskDataset : public Dataset
{
public:
    Fr2DeskDataset(std::string img_path, std::string depth_path);
    virtual void init() override;
    virtual double NextFrame(Frame &f) override;
    virtual bool HasNext() override;
    int get_current_image_index() { return current_image_index_; }
    int get_max_image_index() { return max_image_index_; }
    std::string get_filename(int index = -1) { return img_list[index == -1 ? current_image_index_ : index]; }

private:
    std::string img_path;
    std::string depth_path;
    std::string img_list_file;
    std::string depth_list_file;
    std::vector<std::string> img_list;
    std::vector<std::string> depth_list;
    std::vector<double> timestamps;

    int current_image_index_ = 0;
    int max_image_index_ = 0;
};