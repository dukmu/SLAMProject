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

class TUMDataset : public Dataset
{
public:
    TUMDataset(std::string associate_file, std::string data_path);
    TUMDataset(std::string associate_file);
    virtual void init() override;
    virtual double NextFrame(Frame &f) override;
    virtual bool HasNext() override;
    int get_current_image_index() { return current_image_index_; }
    int get_max_image_index() { return max_image_index_; }
    std::string get_filename(int index = -1) { return img_list[index == -1 ? current_image_index_ : index]; }

private:
    std::string data_path;
    std::string associate_file;
    std::vector<std::string> img_list;
    std::vector<std::string> depth_list;
    std::vector<double> timestamps;

    int current_image_index_ = -1;
    int max_image_index_ = 0;
};