#include "dataset.hpp"
#include "opencv2/imgcodecs.hpp"
Fr2DeskDataset::Fr2DeskDataset(std::string img_path, std::string depth_path)
{
    img_list_file = img_path + "rgb.txt";
    depth_list_file = depth_path + "depth.txt";
    this->img_path = img_path;
    this->depth_path = depth_path;
}

void Fr2DeskDataset::init()
{
    std::ifstream img_list_stream(img_list_file);
    std::ifstream depth_list_stream(depth_list_file);
    std::string img_line;
    std::string depth_line;
    while (std::getline(img_list_stream, img_line))
    {
        if(img_line[0] == '#')
            continue;
        std::stringstream img_line_stream(img_line);
        std::string img_name;
        double timestamp;
        img_line_stream >> timestamp >> img_name;
        img_list.push_back(img_name);
        timestamps.push_back(timestamp);
    }
    max_image_index_ = img_list.size();
    while (std::getline(depth_list_stream, depth_line))
    {
        if(depth_line[0] == '#')
            continue;
        std::stringstream depth_line_stream(depth_line);
        std::string depth_name;
        double timestamp;
        depth_line_stream >> timestamp >> depth_name;
        depth_list.push_back(depth_name);
    }
    max_image_index_ = std::min(max_image_index_, (int)depth_list.size());
}

double Fr2DeskDataset::NextFrame(Frame &f)
{
    f.clear();
    std::string img_name = img_path + img_list[current_image_index_];
    std::string depth_name = depth_path + depth_list[current_image_index_];
    cv::Mat img = cv::imread(img_name, cv::IMREAD_UNCHANGED);
    cv::Mat depth = cv::imread(depth_name, cv::IMREAD_UNCHANGED);
    f.push_back(img);
    f.push_back(depth);
    current_image_index_++;
    return timestamps[current_image_index_];
}

bool Fr2DeskDataset::HasNext()
{
    return current_image_index_ < max_image_index_;
}