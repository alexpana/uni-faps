#pragma once

#include "faps.h"

cv::Mat process_hough(const cv::Mat& src, int window_size, int peak_threshold = 180);

int lab03();