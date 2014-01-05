#include "03_hough.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include <string>
#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static const string IMG_NAME = "hough/edge_simple.bmp";
static const int T = 25;

#define TO_PI 0.01745329251

Mat process_hough(const Mat& src, int window_size, float peak_threshold){
	int peak_count = 8;

	// Initialize Hough Accumulator Space
	int h_width = 360;
	int h_height = (int)sqrt(src.rows * src.rows + src.cols * src.cols) + 2;
	double* hough_space[360];
	for (int i = 0; i < 360; ++i){
		hough_space[i] = new double[h_height];
		memset(hough_space[i], 0, h_height * sizeof(double));
	}
	Mat hough_img(h_height, h_width, CV_8U);
	cout << "Hough space is: " << h_width << "x" << h_height << endl;

	// Fill the Hough Accumulator
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			for (int theta = 0; theta < 360; ++theta) {
				if (src.at<Vec3b>(i, j)[0] > 0) {
					int d = (int)((float)i * sin((float)theta * TO_PI) + (float)j * cos((float)theta * TO_PI));
					if (d > 0 && d < h_height - 1) {
						double intensity = src.at<uchar>(i, j);
						hough_space[theta][d] += 1;
					}
				}
			}
		}
	}
	cout << "Finished filling the hough accumulator" << endl;

	// Compute the maximum intensity
	double max_intensity = 0;
	for (int i = 0; i < hough_img.cols; ++i) {
		for (int j = 0; j < hough_img.rows; ++j) {
			max_intensity = max(max_intensity, hough_space[i][j]);
		}
	}

	// Normalize the Hough Space and build the image
	for (int i = 0; i < hough_img.rows; ++i) {
		for (int j = 0; j < hough_img.cols; ++j) {
			double acc = (hough_space[j][i] / max_intensity);
			hough_space[j][i] = acc;
			hough_img.at<uchar>(i, j) = acc * 255;
		}
	}

	// Display the Hough Accumulator
	faps_display_image("Hough Acc", hough_img);

	// Find the peaks
	vector<Point2i> peaks;
	int hws = window_size / 2 + 1;
	for (int i = hws; i < h_width - hws; ++i) {
		for (int j = hws; j < h_height - hws; ++j) {
			if (hough_space[i][j] > peak_threshold){
				bool is_peak = true;
				for (int wi = -hws+1; wi < hws-1; ++wi) {
					for (int wj = -hws+1; wj < hws-1; ++wj) {
						if (hough_space[i + wi][j + wj] > hough_space[i][j]) {
							is_peak = false;
						}
					}
				}
				if (is_peak){
					peaks.push_back(Point2i(i, j));
				}
			}
		}
	}

	cout << "Found " << peaks.size() << " peaks with intensity greater than " << peak_threshold << endl;

	sort(peaks.begin(), peaks.end(), [&hough_space](const Point2i& lhs, const Point2i& rhs){ return hough_space[lhs.x][lhs.y] > hough_space[rhs.x][rhs.y]; });

	// Draw the lines on the original image
	Mat source_image_copy(src);

	for (int i = 0; i < peaks.size() && i < peak_count; ++i) {
		double theta = peaks[i].x * TO_PI;
		double rho = peaks[i].y;

		int a = peaks[i].y / cos(3.1415 / 2.0 - peaks[i].x * TO_PI);
		int b = peaks[i].y / cos(peaks[i].x * TO_PI);

		int x = cos(theta) * rho;
		int y = sin(theta) * rho;

		int x1 = cos(-3.1415 / 2.0 + theta) * 600;
		int x2 = cos(-3.1415 / 2.0 + theta) * -600;

		int y1 = sin(-3.1415 / 2.0 + theta) * 600;
		int y2 = sin(-3.1415 / 2.0 + theta) * -600;

		line(source_image_copy, cvPoint(x, y), cvPoint(x + x1, y + y1), cvScalar(0, 0, 255), 1);
		line(source_image_copy, cvPoint(x, y), cvPoint(x + x2, y + y2), cvScalar(0, 0, 255), 1);
	}

	// Display the lines over the original image
	faps_display_image("Hough Lines", source_image_copy);

	// Cleanup
	for (int i = 0; i < 360; ++i){
		delete[] hough_space[i];
	}

	return hough_img;
}

int lab03() {
	Mat source_image = imread(IMG_NAME);
	int window_size = 11;
	float peak_threshold = 0.4;
	process_hough(source_image, window_size, peak_threshold);
	cvWaitKey(0);
	return 0;
}