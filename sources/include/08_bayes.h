#ifndef __LAB08_H__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include <string>
#include <cstdio>

using namespace std;
using namespace cv;

double imdist(const Mat& left, const Mat& right) {
	double sum = 0;
	double count = 0;
	for (int i = 0; i < left.rows; ++i) {
		for (int j = 0; j < left.cols; ++j) {
			int diff = (double)abs((int)left.at<uchar>(i, j) - (int)right.at<uchar>(i, j));
			sum += diff;
			if (diff != 0){
				count += 1;
			}
		}
	}
	return sum / (count);
}

void process_image(string image) {	
	double threshold = 70;

	Mat source = imread(image, CV_LOAD_IMAGE_GRAYSCALE);

	double total_sample_count = 4000;
	double sample_count[3] = {
		1200.0,		// one
		1800.0,		// two
		1000.0		// three
	};

	double prior[3] = {
		sample_count[0] / total_sample_count,	// one
		sample_count[1] / total_sample_count,	// two
		sample_count[2] / total_sample_count	// three
	};

	Mat* samples[3];
	for (int i = 0; i < 3; ++i) {
		char buffer[128];
		samples[i] = new Mat[(int)sample_count[i]];
		for (int j = 0; j < sample_count[i]; ++j) {
			sprintf(buffer, "digits\\template%d_%04d.bmp", (int)i+1, (int)j+1);
			samples[i][j] = imread(buffer, CV_LOAD_IMAGE_GRAYSCALE);
		}
	}

	double posterior[3];
	for (int i = 0; i < 3; ++i) {
		posterior[i] = 0;
		for (int j = 0; j < sample_count[i]; ++j) {
			if (imdist(source, samples[i][j]) < threshold) {
				posterior[i] += 1;
			}
		}
		posterior[i] /= sample_count[i];
	}

	char* class_names[3] = {"ONE", "TWO", "THREE"};

	for (int i = 0; i < 3; ++i) {
		printf("P(%s) = %.4lf\nP(T|%s) = %.4lf\nP(%s|T) = %.4lf\n\n", 
			class_names[i], prior[i],
			class_names[i], posterior[i],
			class_names[i], prior[i] * posterior[i]);
	}
}

void lab08() {
	string images[] = {
		"tests/testdigit1_01.bmp", "tests/testdigit1_02.bmp", "tests/testdigit1_03.bmp",
		"tests/testdigit2_01.bmp", "tests/testdigit2_02.bmp", "tests/testdigit2_03.bmp",
		"tests/testdigit3_01.bmp", "tests/testdigit3_02.bmp", "tests/testdigit3_03.bmp"};

	double threshold = 70;

	double total_sample_count = 4000;
	double sample_count[3] = {
		1200.0,		// one
		1800.0,		// two
		1000.0		// three
	};

	double prior[3] = {
		sample_count[0] / total_sample_count,	// one
		sample_count[1] / total_sample_count,	// two
		sample_count[2] / total_sample_count	// three
	};

	Mat* samples[3];
	for (int i = 0; i < 3; ++i) {
		char buffer[128];
		samples[i] = new Mat[(int)sample_count[i]];
		for (int j = 0; j < sample_count[i]; ++j) {
			sprintf(buffer, "digits\\template%d_%04d.bmp", (int)i+1, (int)j+1);
			samples[i][j] = imread(buffer, CV_LOAD_IMAGE_GRAYSCALE);
		}
	}

	for (int i = 0; i < 9; ++i) {
		string image = images[i];

		printf("Processing %s\n", image.c_str());
		Mat source = imread(image, CV_LOAD_IMAGE_GRAYSCALE);

		double posterior[3];
		for (int i = 0; i < 3; ++i) {
			posterior[i] = 0;
			for (int j = 0; j < sample_count[i]; ++j) {
				if (imdist(source, samples[i][j]) < threshold) {
					posterior[i] += 1;
				}
			}
			posterior[i] /= sample_count[i];
		}

		char* class_names[3] = {"ONE", "TWO", "THREE"};
		int class_id = 0;
		for (int i = 0; i < 3; ++i) {
			if (prior[i] * posterior[i] > prior[class_id] * posterior[class_id]) {
				class_id = i;
			}
		}
		printf("Class matched: %s\n", class_names[class_id]);
	}
}

#endif // !__LAB08_H__