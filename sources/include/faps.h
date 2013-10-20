#pragma once

#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

void faps_init();

void faps_timer_start();

int faps_timer_stop();

cv::Mat faps_open_image(int argc, char* argv[], const std::string& image_path);

void faps_display_image(const std::string& name, const cv::Mat& image);

template<typename ProcessFunction>
cv::Mat faps_process_image(const std::string& processName, ProcessFunction processFunction, const cv::Mat& image);

template<typename T>
T faps_random(T min, T max);

void faps_wait();