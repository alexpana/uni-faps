#include <fstream>
#include <string>
#include <stdio.h>
#include <iostream>

#undef FLANN_DEPRECATED

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "Timer.h"

using namespace std;
using namespace cv;

int main( int argc, char* argv[]){
	string imageSourcePath;

	if( argc >= 2 ){
		imageSourcePath = argv[1];
	} else {
		return 1;
	}

	Timer timer;
	timer.Reset();

	Mat sourceMat = cvLoadImage( imageSourcePath.c_str() );
	IplImage sourceImg = (IplImage)sourceMat;

	IplImage *resultImg = cvCreateImage(cvGetSize(&sourceImg), sourceImg.depth, sourceImg.nChannels);

	vector<CvPoint> dots;

	for (int i = 0; i < sourceImg.width; ++i){
		for (int j = 0; j < sourceImg.height; ++j){
			CvScalar color = cvGet2D(&sourceImg, j, i);
			if (color.val[0] == 0){
				dots.push_back(cvPoint(j, i));
				cvSet2D(resultImg, j, i, cvScalar(255, 255, 255));
			} else {
				cvSet2D(resultImg, j, i, cvScalar(0, 0, 0));
			}
		}
	}

	std::cout << "Found " << dots.size() << " black dots in " << timer.Count(Timer::MILISECONDS)<< " ms, nigga!" << std::endl;

	namedWindow( "Source Image", CV_WINDOW_AUTOSIZE );
	imshow( "Source Image", sourceMat );

	namedWindow( "Result Image", CV_WINDOW_AUTOSIZE );
	imshow( "Result Image", Mat(resultImg));


	waitKey(0);

	return 0;
}