#ifndef HELPER_H
#define HELPER_H

/// C++
#include <iostream>
#include <vector>
#include <fstream>
#include <windows.h>

/// OpenCV
#include <opencv2\opencv.hpp>

class Helper {
public:
	Helper() {};
	/**
	Extract HOG feature.

	@param fa Filename of training image list.
	*/
	std::vector<std::vector<float>> ExtractHOGFeature(std::vector<std::string> list) {
		std::ifstream ifs;
		std::string content;
		std::vector<std::vector<float>> v_descriptors_values;
		std::vector<std::vector<cv::Point>> v_locations;
		std::vector<float> descriptors_values;
		std::vector<cv::Point> locations;
		cv::Mat img_mat;
		cv::Size size(120, 150);

		for (size_t i = 0; i < list.size(); i++) {
			img_mat = cv::imread(list[i], 0);

			resize(img_mat, img_mat, size);

			// computes the hog features 
			cv::HOGDescriptor hog;
			hog.compute(img_mat, descriptors_values, cv::Size(32, 32), cv::Size(0, 0), locations);

			v_descriptors_values.push_back(descriptors_values);
			v_locations.push_back(locations);

			descriptors_values.clear();
			locations.clear();
		}

		/************************************** Uncomment to See the Result **************************************/
		//int row = v_descriptors_values.size(), col = v_descriptors_values[0].size();
		//for (size_t j = 0; j < row; j++) {
		//	for (size_t i = 0; i < col; i++) {
		//		std::cout << v_descriptors_values[j][i] << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		return v_descriptors_values;
	}

	/**
	Calculate the accuracy.

	@param aop Array consists of prediction result on dataset
	*/
	float AccuracyCalculation(std::vector<int> aop) {
		float output;

		std::vector<int> pouch;

		for (size_t i = 0; i < 6; i++) {
			pouch.push_back(0);
		}

		int flabel = 0;
		int partition = aop.size() / 6;
		cv::Mat labels(aop.size(), 1, CV_32SC1);
		for (size_t i = 0; i < labels.rows; i++) {
			if (i < partition) {
				labels.at<int>(i, 0) = (int)flabel;
			}
			if (i >= partition && i < partition * 2) {
				flabel = 1;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= partition * 2 && i < partition * 3) {
				flabel = 2;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= partition * 3 && i < partition * 4) {
				flabel = 3;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= partition * 4 && i < partition * 5) {
				flabel = 4;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= partition * 5 && i < partition * 6) {
				flabel = 5;
				labels.at<int>(i, 0) = (int)flabel;
			}
		}

		for (size_t i = 0; i < aop.size(); i++) {
			int ltmp = labels.at<int>(i, 0);
			if (aop[i] == ltmp) {
				pouch[aop[i]] += 1;
			}
		}

		int t = 0;
		for (size_t i = 0; i < pouch.size(); i++) {
			t += pouch[i];
		}

		output = (float)t / aop.size();

		return output;
	}

	/**
	Training and Testing.

	@param trdata List of training data.
	@param tedata List of testing data.
	*/
	float TrainingAndTesting(std::vector<std::string> trdata, std::vector<std::string> tedata) {
		std::vector<std::vector<float>> descriptors, descriptors2;
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_INTENSITY | FOREGROUND_GREEN));
		std::cout << std::endl << "Feature extraction ...  ";

		descriptors = ExtractHOGFeature(trdata);

		std::cout << std::endl << "Feature extraction is done!  ";
		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN));

		/// Set up training data
		cv::Mat training_mat = cv::Mat::zeros(descriptors.size(), descriptors[0].size(), CV_32FC1);
		for (size_t i = 0; i < training_mat.rows; i++) {
			for (size_t j = 0; j < training_mat.cols; j++) {
				training_mat.at<float>(i, j) = descriptors[i][j];
			}
		}

		int flabel = 0;
		cv::Mat labels(descriptors.size(), 1, CV_32SC1);
		for (size_t i = 0; i < labels.rows; i++) {
			if (i < 100) {
				labels.at<int>(i, 0) = (int)flabel;
			}
			if (i >= 100 && i < 100 * 2) {
				flabel = 1;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= 100 * 2 && i < 100 * 3) {
				flabel = 2;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= 100 * 3 && i < 100 * 4) {
				flabel = 3;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= 100 * 4 && i < 100 * 5) {
				flabel = 4;
				labels.at<int>(i, 0) = (int)flabel;
			}
			else if (i >= 100 * 5 && i < 100 * 6) {
				flabel = 5;
				labels.at<int>(i, 0) = (int)flabel;
			}
		}

		/// Set up SVM's parameters
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR);
		svm->setC(1);
		cv::Ptr<cv::ml::TrainData> TrainData = cv::ml::TrainData::create(training_mat, cv::ml::ROW_SAMPLE, labels);

		/// Train the SVM
		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_INTENSITY | FOREGROUND_GREEN));
		std::cout << std::endl << "Training ...  ";

		svm->trainAuto(TrainData);

		std::cout << std::endl << "Training is done!  ";
		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN));

		

		///Testing
		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_INTENSITY | FOREGROUND_GREEN));
		std::cout << std::endl << "Testing ...  ";

		descriptors2 = ExtractHOGFeature(tedata);

		cv::Mat testing_mat = cv::Mat::zeros(descriptors2.size(), descriptors2[0].size(), CV_32FC1);
		for (size_t j = 0; j < testing_mat.rows; j++) {
			for (size_t i = 0; i < testing_mat.cols; i++) {
				testing_mat.at<float>(j, i) = descriptors2[j][i];
			}
		}

		std::vector<int> allprediction;
		for (size_t i = 0; i < testing_mat.rows; i++) {
			int res = svm->predict(testing_mat.row(i));
			allprediction.push_back(res);
		}

		float accuracy = AccuracyCalculation(allprediction);

		std::cout << std::endl << "Testing is done!  " << std::endl;
		SetConsoleTextAttribute(hConsole, (FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN));

		return accuracy;
	}
};

#endif HELPER_H