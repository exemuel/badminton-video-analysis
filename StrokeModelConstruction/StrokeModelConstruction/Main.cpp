/// C++
#include <iostream>
#include <vector>
#include <fstream>

/// OpenCV
#include <opencv2\opencv.hpp>

/// ChartDirector
#include "chartdir.h"

/// Additional
#include "Helper.h"

int main() {
	int tmp = 0;
	std::string content;
	std::ifstream ifs;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	Helper helper;

	// >>>>> Seperation procedure for five fold cross validation
	std::string clear_trainlist = "data/clear_training_list.txt";
	std::string drive_trainlist = "data/drive_training_list.txt";
	std::string drop_trainlist = "data/drop_training_list.txt";
	std::string lob_trainlist = "data/lob_training_list.txt";
	std::string smash_trainlist = "data/smash_training_list.txt";
	std::string other_trainlist = "data/other_training_list.txt";

	// >>>>> Data preparation
	ifs.open(clear_trainlist);
	std::vector<std::string> data;
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(drive_trainlist);
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(drop_trainlist);
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(lob_trainlist);
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(smash_trainlist);
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(other_trainlist);
	while (ifs >> content) {
		std::string tmpstring = content;
		data.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();

	/************************************** Uncomment to See the Result **************************************/
	//for (size_t i = 0; i < part1.size(); i++) {
	//	std::cout << part1[i] << std::endl;
	//}
	/*********************************************************************************************************/
	// >>>>> End of Data preparation

	
	SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_INTENSITY | FOREGROUND_GREEN));
	std::cout << "================================================================================" << std::endl;
	std::cout << "                            Stroke Model Construction                           " << std::endl;
	std::cout << "================================================================================" << std::endl;
	SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN));

	// >>>>> Training and Testing
	float accuracy = helper.TrainingAndTesting(data, data);

	SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_INTENSITY | FOREGROUND_GREEN));
	std::cout << std::endl << "Five fold accuracy : " << accuracy * 100 << "%" << std::endl;
	SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN));

	return 0;
}