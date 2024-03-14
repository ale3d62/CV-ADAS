#ifndef LANE_FINDER_H
#define LANE_FINDER_H

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<int> findLane(const cv::Mat& img);

#endif /* LANE_FINDER_H */