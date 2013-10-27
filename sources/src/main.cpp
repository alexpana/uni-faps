#include <string>
#include <iostream>
#include <map>
#include <thread>

#include "faps.h"
#include "impl_ransac_circle.h"
#include "synchronized_queue.hpp"
#include <atomic>
#include "win32_helper.h"

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
					// todo
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