#include <atomic>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "faps.h"
#include "synchronized_queue.hpp"
#include "win32_helper.h"

#include "impl_ransac_circle.h"
#include "impl_hough.h"

using namespace std;

atomic_bool done;
synchronized_queue<string> commands;


void repl(){
	string command = "";
	faps_init();
	done = false;
	while (command != "exit"){
		cout << "> ";
		getline(cin, command);
		commands.push(command);		
	}
	done = true;
}

void high_gui_refresh(){
	map<string, cv::Mat> symbols;
	while (!done){
		while (!commands.empty())
		{
			string comm = commands.front();
			commands.pop();

			vector<string> command_tokens = faps_string_split(comm, " ");
			if (command_tokens.size() != 0) {
				if (command_tokens[0] == "load") {
					string filepath;
					OpenFileModal(filepath);
					symbols[command_tokens[1]] = faps_open_image(filepath);
				}
				if (command_tokens[0] == "show") {
					faps_display_image(command_tokens[1], symbols[command_tokens[1]]);
				}

				if (command_tokens[0] == "hough") {
					if (command_tokens.size() < 3){
						cout << "Missing arguments." << endl;
					} else {
						int peak_window_size = faps_to_int(command_tokens[2]);
						int peak_threshold = 180;
						if (command_tokens.size() == 4){
							peak_threshold = faps_to_int(command_tokens[3]);
						}
						process_hough(symbols[command_tokens[1]], peak_window_size, peak_threshold);
					}
				}

				if (command_tokens[0] == "ransac") {
					faps_process_image( "Ransac Circle", process_ransac_circle, symbols[command_tokens[1]]);
				}				
			}

		}
		cv::waitKey(1);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

int main(int argc, char* argv[]){

	thread hgr(high_gui_refresh);
	repl();
	hgr.join();

	return 0;
}