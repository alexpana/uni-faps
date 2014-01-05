#pragma once

#include "faps.h"

IplImage* process_dist_transform(const cv::Mat& src);
float process_distance(const cv::Mat& img_template, const cv::Mat& img_query);