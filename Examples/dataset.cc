#include "dataset.hpp"
#include "opencv2/imgcodecs.hpp"
#include "Utils.h"
TUMDataset::TUMDataset(std::string associate_file, std::string data_path)
{
    this->associate_file = associate_file;
    this->data_path = data_path;
    if (this->data_path.back() != '/')
        this->data_path += "/";
}

TUMDataset::TUMDataset(std::string associate_file)
{
    this->associate_file = associate_file;
    int pos = associate_file.find_last_of('/');
    this->data_path = associate_file.substr(0, pos + 1);
}

void TUMDataset::init()
{
    std::ifstream associate(associate_file);
    std::string line;
    while (std::getline(associate, line))
    {
        if(line[0] == '#')
            continue;
        std::stringstream line_stream(line);
        std::string var;
        double timestamp;
        line_stream >> timestamp >> var;
        timestamps.push_back(timestamp);
        img_list.push_back(var);
        line_stream >> timestamp >> var;
        depth_list.push_back(var);
    }
    max_image_index_ = img_list.size();
}

double TUMDataset::NextFrame(Frame &f)
{
    f.clear();
    current_image_index_++;
    std::string img_name = data_path + img_list[current_image_index_];
    std::string depth_name = data_path + depth_list[current_image_index_];
    cv::Mat img = cv::imread(img_name, cv::IMREAD_UNCHANGED);
    cv::Mat depth = cv::imread(depth_name, cv::IMREAD_UNCHANGED);
    f.push_back(img);
    f.push_back(depth);
    return timestamps[current_image_index_];
}

bool TUMDataset::HasNext()
{
    return current_image_index_+1 < max_image_index_;
}