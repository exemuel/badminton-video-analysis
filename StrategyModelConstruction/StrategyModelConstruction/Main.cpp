/// C++
#include <iostream>
#include <fstream>

/// OpenCV
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>

/// Additional
#include "CvHMM.h"
#include "Intermediary.h"
#include "FeatureExtraction.h";
#include "StrategyPrediction.h"

int main() {
	int tmp = 0;
	std::string content;
	std::ifstream ifs;
	std::cout << "                                                                               " << std::endl;
	std::cout << "===============================================================================" << std::endl;
	std::cout << "                                                                               " << std::endl;
	std::cout << "                              Strategy Evaluation                              " << std::endl;
	std::cout << "                                                                               " << std::endl;
	std::cout << "===============================================================================" << std::endl;
	std::cout << "                                                                               " << std::endl;

	// >>>>> Seperation procedure for five fold cross validation
	std::string offlist = "utilities/offensivelist.txt";
	std::string deflist = "utilities/defensivelist.txt";
	//std::string fofflist = "utilities/foffensivelist.txt";
	//std::string fdeflist = "utilities/fdefensivelist.txt";

	//// >>>>> Part I
	//ifs.open(offlist);
	//std::vector<std::string> part1tr;
	//std::vector<std::string> part1te;
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 20) {
	//		part1tr.push_back(tmpstring);
	//	}
	//	else {
	//		part1te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//ifs.open(deflist);
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 20) {
	//		part1tr.push_back(tmpstring);
	//	}
	//	else {
	//		part1te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//// <<<<< End of Part I

	//// >>>>> Part II
	//ifs.open(offlist);
	//std::vector<std::string> part2tr;
	//std::vector<std::string> part2te;
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 15) {
	//		part2tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 20) {
	//		part2tr.push_back(tmpstring);
	//	}
	//	else {
	//		part2te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//ifs.open(deflist);
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 15) {
	//		part2tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 20) {
	//		part2tr.push_back(tmpstring);
	//	}
	//	else {
	//		part2te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//// <<<<< End of Part II

	//// >>>>> Part III
	//ifs.open(offlist);
	//std::vector<std::string> part3tr;
	//std::vector<std::string> part3te;
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 10) {
	//		part3tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 15) {
	//		part3tr.push_back(tmpstring);
	//	}
	//	else {
	//		part3te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//ifs.open(deflist);
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 10) {
	//		part3tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 15) {
	//		part3tr.push_back(tmpstring);
	//	}
	//	else {
	//		part3te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//// <<<<< End of Part III

	//// >>>>> Part IV
	//ifs.open(offlist);
	//std::vector<std::string> part4tr;
	//std::vector<std::string> part4te;
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 5) {
	//		part4tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 10) {
	//		part4tr.push_back(tmpstring);
	//	}
	//	else {
	//		part4te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//ifs.open(deflist);
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp < 5) {
	//		part4tr.push_back(tmpstring);
	//	}
	//	else if (tmp >= 10) {
	//		part4tr.push_back(tmpstring);
	//	}
	//	else {
	//		part4te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//// <<<<< End of Part IV

	//// >>>>> Part V
	//ifs.open(offlist);
	//std::vector<std::string> part5tr;
	//std::vector<std::string> part5te;
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp >= 5) {
	//		part5tr.push_back(tmpstring);
	//	}
	//	else {
	//		part5te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//ifs.open(deflist);
	//while (ifs >> content) {
	//	std::string tmpstring = content;
	//	if (tmp >= 5) {
	//		part5tr.push_back(tmpstring);
	//	}
	//	else {
	//		part5te.push_back(tmpstring);
	//	}
	//	tmp++;
	//}
	//tmp = 0;
	//ifs.clear(); ifs.close();
	//// <<<<< End of Part V
	//// <<<<< End of Seperation procedure for five fold cross validation

	// >>>>> Part VI
	ifs.open(offlist);
	std::vector<std::string> part6tr;
	std::vector<std::string> part6te;
	while (ifs >> content) {
		std::string tmpstring = content;
		part6tr.push_back(tmpstring);
		part6te.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	ifs.open(deflist);
	while (ifs >> content) {
		std::string tmpstring = content;
		part6tr.push_back(tmpstring);
		part6te.push_back(tmpstring);
	}
	ifs.clear(); ifs.close();
	// <<<<< End of Part VI


	/************************************** Uncomment to See the Result **************************************/
	//std::cout << std::endl;
	//for (size_t i = 0; i < part1tr.size(); i++) {
	//	std::cout << part1tr[i] << " ";
	//}
	//std::cout << std::endl;
	//for (size_t i = 0; i < part2tr.size(); i++) {
	//	std::cout << part2tr[i] << " ";
	//}
	//std::cout << std::endl;
	//for (size_t i = 0; i < part3tr.size(); i++) {
	//	std::cout << part3tr[i] << " ";
	//}
	//std::cout << std::endl;
	//for (size_t i = 0; i < part4tr.size(); i++) {
	//	std::cout << part4tr[i] << " ";
	//}
	//std::cout << std::endl;
	//for (size_t i = 0; i < part5tr.size(); i++) {
	//	std::cout << part5tr[i] << " ";
	//}
	//std::cout << std::endl;
	//for (size_t i = 0; i < part6tr.size(); i++) {
	//	std::cout << part6tr[i] << " ";
	//	std::cout << part6te[i] << " ";
	//}
	//std::cin.get();
	/*********************************************************************************************************/

	//// >>>>> Training and Testing
	//std::cout << std::endl << "Part I" << std::endl;
	//intermediary::Training(part1tr);
	//float acc_part1 = intermediary::Testing(part1te);

	//std::cout << "Accuracy Part I : " << acc_part1 * 100 << "%" << std::endl;

	//std::cout << std::endl << "Part II" << std::endl;
	//intermediary::Training(part2tr);
	//float acc_part2 = intermediary::Testing(part2te);

	//std::cout << "Accuracy Part II : " << acc_part2 * 100 << "%" << std::endl;

	//std::cout << std::endl << "Part III" << std::endl;
	//intermediary::Training(part3tr);
	//float acc_part3 = intermediary::Testing(part3te);

	//std::cout << "Accuracy Part III : " << acc_part3 * 100 << "%" << std::endl;

	//std::cout << std::endl << "Part IV" << std::endl;
	//intermediary::Training(part4tr);
	//float acc_part4 = intermediary::Testing(part4te);

	//std::cout << "Accuracy Part IV : " << acc_part4 * 100 << "%" << std::endl;

	//std::cout << std::endl << "Part V" << std::endl;
	//intermediary::Training(part5tr);
	//float acc_part5 = intermediary::Testing(part5te);

	//std::cout << "Accuracy Part V : " << acc_part5 * 100 << "%" << std::endl;

	std::cout << std::endl << "Part VI" << std::endl;
	intermediary::Training(part6tr);
	float acc_part6 = intermediary::Testing(part6te);

	std::cout << "Accuracy Part VI : " << acc_part6 * 100 << "%" << std::endl;

	//// <<<<< End of Training and Testing

	//float avg_acc = (acc_part1 + acc_part2 + acc_part3 + acc_part4 + acc_part5) / 5;

	//std::cout << std::endl << "Five fold accuracy : " << avg_acc * 100 << "%" << std::endl;

	return 0;
}