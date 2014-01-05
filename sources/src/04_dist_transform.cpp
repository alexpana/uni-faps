#include "04_dist_transform.h"

using namespace std;
using namespace cv;

double weight_euclid(int wi, int wj) {
	// Origin
	if (wi == 0 && wj == 0) return 0;
	// H/W distance
	if (wi == 0 || wj == 0) return 1;
	// D distance
	return 1.41;
}

IplImage* process_dist_transform(const cv::Mat& src)
{
	IplImage sourceImg = (IplImage)src;
	IplImage *dst = cvCreateImage(cvGetSize(&sourceImg), sourceImg.depth, sourceImg.nChannels);

	for (int i = 0; i < sourceImg.width; ++i) {
		for (int j = 0; j < sourceImg.height; ++j) {
			cvSet2D(dst, j, i, cvGet2D(&sourceImg, j, i));
		}
	}

	int left_mask[]  = {1, 1, 1, 1, 1, 0, 0, 0, 0};
	int right_mask[] = {0, 0, 0, 0, 1, 1, 1, 1, 1};

	for (int i = 1; i < sourceImg.width-1; ++i) {
		for (int j = 1; j < sourceImg.height-1; ++j) {
			for (int wi = -1; wi <= 1; ++wi) {
				for (int wj = -1; wj <= 1; ++wj) {
					if (left_mask[(wi + 1) * 3 + wj + 1] == 0){
						continue;
					}
					double old_value = cvGet2D(dst, j, i).val[0];
					double new_value = cvGet2D(dst, j + wj, i + wi).val[0] + weight_euclid(wi, wj);
					if (new_value < old_value){
						cvSet2D(dst, j, i, cvScalar(new_value, new_value, new_value));
					}
				}
			}
		}
	}

	for (int i = sourceImg.width-2; i > 1; --i) {
		for (int j = sourceImg.height-2; j > 1; --j) {
			for (int wi = -1; wi <= 1; ++wi) {
				for (int wj = -1; wj <= 1; ++wj) {
					if (right_mask[(wi + 1) * 3 + wj + 1] == 0){
						continue;
					}

					double old_value = cvGet2D(dst, j, i).val[0];
					double new_value = cvGet2D(dst, j + wj, i + wi).val[0] + weight_euclid(wi, wj);
					if (new_value < old_value){
						cvSet2D(dst, j, i, cvScalar(new_value, new_value, new_value));
					}
				}
			}
		}
	}

	//faps_display_image("Distance Transform", Mat(dst));

	return dst;
}

float process_distance(const Mat& img_template, const Mat& img_query){
	IplImage* query_dt = process_dist_transform(img_query);
	IplImage ipl_template = (IplImage)img_template;

	float sum = 0;
	int count = 0;

	for (int i = 0; i < ipl_template.width; ++i) {
		for (int j = 0; j < ipl_template.height; ++j) {
			int template_value = cvGet2D(&ipl_template, j, i).val[0];
			if (template_value == 0)
			{
				sum += cvGet2D(query_dt, j, i).val[0];
				count += 1;
			}
		}
	}

	return sum / (float)count;
}