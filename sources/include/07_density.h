#include "faps.h"
#include <stdio.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

const int p = 400;

float miu[19*19];
float sigma[19*19];
float cov[19*19][19*19];
float cor[19*19][19*19];
Mat images[p];

float corelation(int x1, int y1, int x2, int y2) {
	return cor[(x1) * 19 + y1][(x2) * 19 + y2];
}

void read_corelation() 
{
	ifstream fin("corelation.txt");
	for (int i = 0; i < 19 * 19; ++i) {
		for (int j = 0; j < 19 * 19; ++j) {
			fin >> cor[i][j];
		}
	}
	fin.close();
}

void calculate_corelation() 
{
	for (int i = 0; i < 19; ++i) {
		for (int j = 0; j < 19; ++j) {
			float sum = 0;
			for (int k = 0; k < p; ++k) {
				sum += images[k].at<uchar>(i, j);
			}
			miu[i * 19 + j] = sum / (float)p;
		}
	}

	for (int i = 0; i < 19; ++i) {
		for (int j = 0; j < 19; ++j) {
			float sum = 0;
			for (int k = 0; k < p; ++k) {
				sum += pow((float)images[k].at<uchar>(i, j) - miu[i * 19 + j], 2.0f);
			}
			sigma[i * 19 + j] = sqrt(sum / (float)p);
		}
	}

	for (int i = 0; i < 19 * 19; ++i) {
		for (int j = 0; j < 19 * 19; ++j) {
			float sum = 0;
			for (int k = 0; k < p; ++k) {
				sum += ((float)images[k].at<uchar>(i / 19, i % 19) - miu[i]) * ((float)images[k].at<uchar>(j / 19, j % 19) - miu[j]);
			}
			cov[i][j] = sum / (float)p;
			cor[i][j] = cov[i][j] / (sigma[i] * sigma[j]);
		}
	}

	ofstream fout("corelation.txt");
	for (int i = 0; i < 19*19; ++i) {
		for (int j = 0; j < 19*19; ++j) {
			fout << cor[i][j] << " ";
		}
		fout << endl;
	}
	fout.close();
}

Mat corelation_graph(int x1, int y1, int x2, int y2){
	Mat res = Mat(256, 256, CV_8UC1);

	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			res.at<uchar>(i, j) = 255;
		}
	}

	for (int k = 0; k < p; ++k) {
		int i = images[k].at<uchar>(x1, y1);
		int j = images[k].at<uchar>(x2, y2);
		res.at<uchar>(256 - i, j) = 0;
	}

	return res;
}

void read_images() 
{
	char filename[1024];
	for (int i = 0; i < p; ++i){
		sprintf(filename, "faces\\face%05d.bmp", i+1);
		images[i] = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	}
}


void lab06(){
	read_images();

	read_corelation();

	namedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("result", corelation_graph(19 - 13 - 1, 4, 19 - 13 - 1, 14));
	printf("%f\n", corelation(19 - 13 - 1, 4, 19 - 13 - 1, 14));
	waitKey(0);

	printf("Done");
}

vector<Point2f> l7_read_points()
{
	vector<Point2f> points;

	ifstream fin("points.txt");
	int n;
	float x, y;
	fin >> n;
	for (int i = 0; i < n; ++i) 
	{
		fin >> x >> y;
		points.push_back(Point2f(x, y));
	}
	fin.close();

	return points;
}

void l7_distribution(const vector<Point2f>& points, float& mx, float& sx, float& my, float& sy)
{
	mx = sx = my = sy = 0;

	for (vector<Point2f>::const_iterator it = points.begin(); it != points.end(); ++it)
	{
		mx += (*it).x;
		my += (*it).y;
	}
	mx /= (float)points.size();
	my /= (float)points.size();
	for (vector<Point2f>::const_iterator it = points.begin(); it != points.end(); ++it) 
	{
		sx += ((*it).x - mx) * ((*it).x - mx);
		sy += ((*it).y - my) * ((*it).y - my);
	}
	sx = sqrt(sx / (float)points.size());
	sy = sqrt(sy / (float)points.size());
}

float l7_gauss(float x, float m, float s)
{
	return (float)(1.0 / (sqrt(2*3.14159) * s)) * exp(- ((x - m) * (x - m) / (2*s*s)));
}

float l7_gauss2d(float x, float y, float m, float s)
{
	return (float)(1.0 / (sqrt(2*3.14159) * s)) * exp(- ((x - m) * (x - m) / (2*s*s)));
}

template<typename T>
void l7_fill_mat(Mat& mat, T val)
{
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			mat.at<T>(i, j) = val;
		}
	}
}

void l7_draw_gaussian(Mat& dst, float m, float s, float scale)
{
	int prev = -1;
	for(int i = 0; i < 256; ++i)
	{
		float v = l7_gauss((float)i, m, s) * (255.0 / scale);
		int j = (int)v;
		j = max(0, min(255, j));
		dst.at<uchar>(255 - j, i) = 0;
		if (prev != -1)
		{
			int delta = prev > j ? 0 : -1;
			for (int j2 = min(prev, j); j2 < max(prev, j); ++j2) {
				dst.at<uchar>(255 - j2, i + delta) = 0;
			}
		}
		prev = j;
	}
}

void l7_histox(Mat& dst, vector<Point2f> pts)
{
	int values[256];
	int max_val = 0;
	memset(values, 0, sizeof(values));

	for (int i = 0; i < pts.size(); ++i)
	{
		values[(int)pts[i].x] += 1;
		if (values[(int)pts[i].x] > max_val) {
			max_val = values[(int)pts[i].x];
		}
	}

	for (int i = 0; i < 255; ++i)
	{
		for (int j = 0; j < values[i] * (255.0f / (float)max_val); ++j)
		{
			dst.at<uchar>(255 - j, i) = 0;
		}
	}
}

void l7_histoy(Mat& dst, vector<Point2f> pts)
{
	int values[256];
	int max_val = 0;
	memset(values, 0, sizeof(values));

	for (int i = 0; i < pts.size(); ++i)
	{
		values[(int)pts[i].y] += 1;
		if (values[(int)pts[i].y] > max_val) {
			max_val = values[(int)pts[i].y];
		}
	}

	for (int i = 0; i < 255; ++i)
	{
		for (int j = 0; j < values[i] * (255.0f / (float)max_val); ++j)
		{
			dst.at<uchar>(255 - j, i) = 0;
		}
	}
}

void lab07()
{
	Mat points_img = Mat(256, 256, CV_8U);
	Mat pdf_img = Mat(256, 256, CV_8U);
	Mat gauss_img = Mat(256, 256, CV_8U);
	Mat histx = Mat(256, 256, CV_8U);
	Mat histy = Mat(256, 256, CV_8U);

	vector<Point2f> points = l7_read_points();

	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); ++it)
	{
		points_img.at<uchar>(255 - (int)(*it).y, (int)(*it).x) = 0;
	}

	float mx, sx, my, sy;
	l7_distribution(points, mx, sx, my, sy);

	float mval = 0;
	for(int i = 0; i < 256; ++i)
	{
		float val = l7_gauss((float)i, mx, sx);
		mval = val > mval ? val : mval;
	}

	for(int i = 0; i < 256; ++i)
	{
		float val = l7_gauss((float)i, my, sy);
		mval = val > mval ? val : mval;
	}

	l7_draw_gaussian(pdf_img, mx, sx, mval);
	l7_draw_gaussian(pdf_img, my, sy, mval);
	l7_histox(histx, points);
	l7_histoy(histy, points);

	imshow("points", points_img);
	imshow("pdf", pdf_img);
	imshow("2dgauss", gauss_img);
	imshow("histy", histx);
	imshow("histx", histy);

	waitKey(0);
}
