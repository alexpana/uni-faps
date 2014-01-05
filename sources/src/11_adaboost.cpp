#include "11_adaboost.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include <string>
#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static const string IMG_NAME = "lab11/class_0red_2blue.bmp";
static const int T = 25;

enum CLASS {
	CLASS_ONE = -1, 
	CLASS_TWO = 1
};

struct Sample{
	int x, y;
	CLASS cls;

	Sample(int _x, int _y, CLASS _cls): x(_x), y(_y), cls(_cls) {}
};

struct Classifier {
	int feature, sign, threshold;
	Classifier(): feature(0), sign(0), threshold(0) {}
	Classifier(int f, int s, int t): feature(f), sign(s), threshold(t) {}
	Classifier(const Classifier& c): feature(c.feature), sign(c.sign), threshold(c.threshold) {}
};

CLASS h(Vec2b P, Classifier c) {
	return (c.sign * P.val[c.feature]) < (c.sign * c.threshold) ? CLASS_ONE : CLASS_TWO;
}

CLASS f(Vec2b P, vector<pair<Classifier, float>>& classifiers) {
	float sum = 0;
	for (auto& cls : classifiers) {
		sum += h(P, cls.first) * cls.second;
	}

	return sum < 0 ? CLASS_ONE : CLASS_TWO;
}

void normalize_weights(vector<float>& d) {
	float sum = 0;

	for (auto& w : d) {
		sum += w;
	}

	for (auto& w : d) {
		w /= sum;
	}
}

void lab11() {
	Mat img = imread(IMG_NAME);

	vector<Sample> samples;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b v = img.at<Vec3b>(i, j);
			if (v.val[0] > 128 && v.val[1] < 128) {
				samples.push_back(Sample(j, i, CLASS_ONE));
			}
			if (v.val[2] > 128 && v.val[1] < 128) {
				samples.push_back(Sample(j, i, CLASS_TWO));
			}
		}
	}

	vector<float> D;
	for (unsigned int i = 0; i < samples.size(); ++i) {
		D.push_back(1.0f / samples.size());
	}

	vector<pair<Classifier, float>> classifiers;

	for (int iteration = 0; iteration < T; ++iteration) {
		normalize_weights(D);
		Classifier best;
		vector<int> best_wins;
		float best_error = 1.0f;

		// for each feature
		for (int ftr = 0; ftr < 2; ++ftr) {
			// for each sign
			for (int s = -1; s <=1; s += 2) {
				// for each threshold
				for (auto& p1 : samples) {
					int th = (ftr == 0 ? p1.x : p1.y);
					Classifier cls(ftr, s, th);

					float err = 0;
					vector<int> wins;

					// for each point
					int index = 0;
					for (auto& p2 : samples) {
						if (h(Vec2b(p2.x, p2.y), cls) != p2.cls) {
							err += D[index];
						} else {
							wins.push_back(index);
						}
						index += 1;
					}

					if (err < best_error) {
						best_wins = wins;
						best_error = err;
						best = cls;
					}
				}
			}
		}

		if (best_error >= 0.5) {
			continue;
		}

		float beta = best_error / (1.0f - best_error);
		float alpha = 0.5f * log(1.0f / beta);

		for (auto& iw : best_wins) {
			D[iw] *= beta;
		}

		classifiers.push_back(make_pair(best, alpha));
	}

	//  color the result
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b v = img.at<Vec3b>(i, j);
			if (v.val[0] > 128 && v.val[1] > 128 && v.val[2] > 128) {
				img.at<Vec3b>(i, j) = f(Vec2b(j, i), classifiers) == CLASS_ONE ? Vec3b(60, 60, 60) : Vec3b(120, 120, 120);
			}
		}
	}

	imshow("result", img);
	cvWaitKey(0);

	cout << samples.size() << endl;
}