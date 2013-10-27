#include <string>
#include <iostream>
#include <map>
#include <thread>

#include "faps.h"
#include "impl_ransac_circle.h"

using namespace std;

bool done = false;
bool process_lock = false;

vector<string> command_tokens;

map<string, cv::Mat> symbols;

void repl(){
	string command = "";
	char buffer[256];

	faps_init();

	while (command != "exit"){
		if ( process_lock ){
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		cout << "> ";

		cin.getline(buffer, 256);
		command = buffer;
		command_tokens = faps_string_split(command, " ");
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	done = true;
}

void high_gui_refresh(){
	while (!done){
		if (command_tokens.size() != 0) {
			process_lock = true;
			if (command_tokens[0] == "load") {
				symbols[command_tokens[1]] = faps_open_image(command_tokens[2]);
			}

			if (command_tokens[0] == "show") {
				faps_display_image(command_tokens[1], symbols[command_tokens[1]]);
			}

			if (command_tokens[0] == "hough") {
				// todo
			}

			if (command_tokens[0] == "ransac") {
				faps_process_image( "Ransac Circle", process_ransac_circle, symbols[command_tokens[1]]);
			}
			command_tokens.clear();
			process_lock = false;
		}
		cv::waitKey(1);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

int main(int argc, char* argv[]){
	thread edt(repl);
	high_gui_refresh();
	edt.join();
	return 0;
}