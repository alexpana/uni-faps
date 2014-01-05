#include "05_hog.h"

#include <cmath>

using namespace std;

pair<IplImage*, IplImage*> gradient(const IplImage& image){
	IplImage* dx = cvCreateImage(cvGetSize(&image), image.depth, image.nChannels);
	IplImage* dy = cvCreateImage(cvGetSize(&image), image.depth, image.nChannels);

	for (int i = 1; i < image.width-1; ++i) {
		for (int j = 1; j < image.height-1; ++j) {
			double val_dx = cvGet2D(&image, j, i - 1).val[0] - cvGet2D(&image, j, i + 1).val[0];
			double val_dy = cvGet2D(&image, j - 1, i).val[0] - cvGet2D(&image, j + 1, i).val[0];
			cvSet2D(dx, j, i, cvScalar(val_dx, val_dx, val_dx));
			cvSet2D(dy, j, i, cvScalar(val_dy, val_dy, val_dy));
		}
	}

	return make_pair(dx, dy);
}

float **magnitude(float ** dx, float** dy, int w, int h) {
	float** res = new float*[w];
	for (int i = 0; i < w; ++i)
		res[i] = new float[w];

	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			float ix = dx[i][j];
			float iy = dx[i][j];
			float m = sqrt(ix * ix + iy * iy);
			res[i][j] = m;
		}
	}
	return res;
}

float **orientation(const IplImage& dx, const IplImage& dy){
	float** res = new float*[dx.width];
	for (int i = 0; i < dx.width; ++i)
		res[i] = new float[dx.height];

	for (int i = 0; i < dx.width; ++i) {
		for (int j = 0; j < dx.height; ++j) {
			float ix = cvGet2D(&dx, j, i).val[0];
			float iy = cvGet2D(&dy, j, i).val[0];
			float o = atan2(iy, ix);
			res[i][j] = o;
		}
	}
	return res;
}

IplImage* normalize(const IplImage& image){
	IplImage* res = cvCreateImage(cvGetSize(&image), image.depth, image.nChannels);
	float max = 0;
	for (int i = 0; i < image.width; ++i) {
		for (int j = 0; j < image.height; ++j) {
			if (cvGet2D(&image, j, i).val[0] > max){
				max = cvGet2D(&image, j, i).val[0];
			}
		}
	}

	for (int i = 0; i < image.width; ++i) {
		for (int j = 0; j < image.height; ++j) {
			float new_value = cvGet2D(&image, j, i).val[0] / max * 255;
			cvSet2D(res, j, i, cvScalar(new_value, new_value, new_value));
		}
	}

	return res;
}

int orientation_seg(float o){
	float or = (o * 57.29577);
	if (or < 0){
		or += 360;
	}
	or /= 40.0f;
	return (int)or;
}

IplImage* paint_gradients(float **g, const IplImage& tpl){
	IplImage* res = cvCreateImage(cvGetSize(&tpl), tpl.depth, tpl.nChannels);

	CvScalar colors[] = {
		cvScalar(255, 0, 0), 
		cvScalar(255, 128, 0),
		cvScalar(0, 255, 0),
		cvScalar(0, 255, 0),
		cvScalar(0, 255, 128),
		cvScalar(0, 255, 255),
		cvScalar(0, 0, 255),
		cvScalar(128, 0, 255),
		cvScalar(255, 0, 255)
	};

	for (int i = 0; i < tpl.width; ++i) {
		for (int j = 0; j < tpl.height; ++j) {
			float original_orientation = g[i][j];
			cvSet2D(res, j, i, colors[orientation_seg(original_orientation)]);
		}
	}

	return res;
}

float **orientation(float ** dx, float** dy, int w, int h){
	float** res = new float*[w];
	for (int i = 0; i < w; ++i)
		res[i] = new float[h];

	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			float o = atan2(dy[i][j], dx[i][j]);
			res[i][j] = o;
		}
	}
	return res;
}

std::pair<float**, float**> gradientf(const IplImage& image){
	float** dx = new float*[image.width];
	for (int i = 0; i < image.width; ++i)
		dx[i] = new float[image.height];

	float** dy = new float*[image.width];
	for (int i = 0; i < image.width; ++i)
		dy[i] = new float[image.height];

	for (int i = 1; i < image.width-1; ++i) {
		for (int j = 1; j < image.height-1; ++j) {
			double val_dx = cvGet2D(&image, j, i + 1).val[0] - cvGet2D(&image, j, i - 1).val[0];
			double val_dy = cvGet2D(&image, j + 1, i).val[0] - cvGet2D(&image, j - 1, i).val[0];
			dx[i][j] = val_dx;
			dy[i][j] = val_dy;
		}
	}

	return make_pair(dx, dy);
}

float* hog_histo(float** mag, float **or, int ws, int w, int h){
	float *histo = new float[(w / ws) * (h / ws) * 9];

	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			int o = orientation_seg(or[i][j]);
			float m = mag[i][j];
			int x = i / ws;
			int y = j / ws;
			histo[y * (w / ws) * 9 + x * 9 + o] += m;
		}
	}

	return histo;
}

IplImage* hog_paint_histo(float* histo, int ws, const IplImage& tpl){
	IplImage* res = cvCreateImage(cvGetSize(&tpl), tpl.depth, tpl.nChannels);

	CvScalar colors[] = {
		cvScalar(255, 0, 0), 
		cvScalar(255, 128, 0),
		cvScalar(0, 255, 0),
		cvScalar(0, 255, 0),
		cvScalar(0, 255, 128),
		cvScalar(0, 255, 255),
		cvScalar(0, 0, 255),
		cvScalar(128, 0, 255),
		cvScalar(255, 0, 255)
	};

	int w = tpl.width;
	int h = tpl.height;

	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			int max = 0;
			for (int o = 0; i < 9; ++o){
				int x = i / ws;
				int y = j / ws;
				if (histo[y * (w / ws) * 9 + x * 9 + o] > histo[y * (w / ws) * 9 + x * 9 + max])
					max = o;
			}
			cvSet2D(res, j, i, colors[max]);
		}
	}

	return res;
}