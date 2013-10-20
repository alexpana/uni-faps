#include "faps.h"

#include <chrono>

#undef FLANN_DEPRECATED

using namespace std;
using namespace cv;

static chrono::high_resolution_clock::time_point faps_time_point;

void faps_init(){
	// pass
}

void faps_timer_start(){
	faps_time_point = chrono::high_resolution_clock::now();
}

int faps_timer_stop(){
	using namespace chrono;
	high_resolution_clock::time_point now = high_resolution_clock::now();
	return (int)duration_cast<chrono::milliseconds>(now - faps_time_point).count();
}

Mat faps_open_image(int argc, char* argv[], const string& image_path){
	string imageSourcePath = image_path;

	if( argc >= 2 ){
		imageSourcePath = argv[1];
	}

	return cvLoadImage(imageSourcePath.c_str());
}

void faps_display_image(const string& name, const Mat& image){
	namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, image);
}

template<typename ProcessFunction>
Mat faps_process_image(const std::string& processName, ProcessFunction processFunction, const cv::Mat& image){
	cout << "Running process " << processName << endl;

	faps_timer_start();
	IplImage *result = processFunction(image);
	cout << "Process " << processName << " finished in " << faps_timer_stop() << "ms." << endl;

	faps_display_image("Source Image", src);
	faps_display_image("Result Image", Mat(result));
}

template<typename T>
T faps_random(T min, T max){
	return ((T)(rand()) * max) / (T)RAND_MAX;
}

void faps_wait(){
	waitKey(0);
}