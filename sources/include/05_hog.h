#pragma once

#include <utility>

#include "faps.h"

std::pair<IplImage*, IplImage*> gradient(const IplImage& image);
std::pair<float**, float**> gradientf(const IplImage& image);

float **magnitude(float ** dx, float** dy, int w, int h);

float **orientation(const IplImage& dx, const IplImage& dy);
float **orientation(float ** dx, float** dy, int w, int h);

IplImage* normalize(const IplImage& image);
IplImage* paint_gradients(float **g, const IplImage& tpl);

IplImage* normalize(const IplImage& image);

float* hog_histo(float** mag, float **or, int wSize);

IplImage* hog_paint_histo(float* histo, int ws, const IplImage& tpl);