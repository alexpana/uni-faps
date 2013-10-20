#include "faps.h"
#include "impl_ransac_circle.h"

int main(int argc, char* argv[]){
	faps_init();

	faps_process_image(
		"Ransac Circle",
		process_ransac_circle,
		faps_open_image(argc, argv, "lab_02e_circle.bmp"));

	faps_wait();
	return 0;
}