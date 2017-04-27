#ifndef INTERMEDIARY_H
#define INTERMEDIARY_H

#include "FeatureExtraction.h"
#include "StrategyPrediction.h"

namespace intermediary {
	/**
	Check frame size
	
	@param filename Video filename
	*/
	cv::Size CheckFrameSize(std::string filename) {
		cv::VideoCapture stream(filename);
		if (!stream.isOpened()) {
			std::cerr << "Unable to open video file: " << filename << " !" << std::endl;
			exit(EXIT_FAILURE);
		}

		// first frame
		cv::Mat frame_rgb;
		stream.read(frame_rgb);

		cv::Size framesize = frame_rgb.size();

		return framesize;
	}

	/**
	Accuracy Calculation.

	@param allprediction All prediction result on testing data.
	*/
	float AccuracyCalculation(std::vector<int> aop) {
		float output;

		std::vector<int> pouch;

		for (size_t i = 0; i < 2; i++) {
			pouch.push_back(0);
		}

		int flabel = 0;
		int partition = aop.size() / 2;
		cv::Mat labels(aop.size(), 1, CV_32SC1);
		for (size_t i = 0; i < labels.rows; i++) {
			if (i < partition) {
				labels.at<int>(i, 0) = (int)flabel;
			}
			else {
				flabel = 1;
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

		return output;;
	}

	void Training(std::vector<std::string> training_data) {
		int offwidth = 0, offheight = 0, defwidth = 0, defheight = 0;

		std::vector<std::string> offlist;
		std::vector<std::string> deflist;
		std::ifstream ifs;
		std::ofstream ofss, ofsx, ofsy;

		std::cout << std::endl << "Training is in progress...";

		// make sure training_data is not empty
		if (training_data.empty()) {
			std::cerr << "Fill the training data first!" << std::endl;
			exit(EXIT_FAILURE);
		}

		cv::Size framesize = CheckFrameSize(training_data[0]);

		// seperate the offensive and defensive training_data into two array
		for (size_t i = 0; i < training_data.size(); i++) {
			if (i < training_data.size() / 2) {
				offlist.push_back(training_data[i]);
			}
			else {
				deflist.push_back(training_data[i]);
			}
		}

		// save offensive stroke and bottom player position (x,y) feature in a file
		ofss.open("features/os_feature.txt");
		ofsx.open("features/ox_feature.txt");
		ofsy.open("features/oy_feature.txt");

		// process offensive list
		for (size_t i = 0; i < offlist.size(); i++) {
			std::vector<cv::Vec3i> features;
			features = featureextraction::StrokePositionExtraction(offlist[i]);
			std::cout << std::endl << "Features extracted from " << offlist[i];
			for (size_t h = 0; h < features.size(); h++) {
				ofss << features[h][0] << " ";
				ofsx << features[h][1] << " ";
				ofsy << features[h][2] << " ";
			}
			ofss << "-" << std::endl;
			ofsx << "-" << std::endl;
			ofsy << "-" << std::endl;
			if (offwidth < features.size()) {
				offwidth = features.size();
			}
		}
		offheight = offlist.size();

		cv::Size offsize(offwidth, offheight);
		ofss.close();
		ofsx.close();
		ofsy.close();

		// save offensive stroke and bottom player position (x,y) feature in a file
		ofss.open("features/ds_feature.txt");
		ofsx.open("features/dx_feature.txt");
		ofsy.open("features/dy_feature.txt");

		// process defensive list
		for (size_t i = 0; i < deflist.size(); i++) {
			std::vector<cv::Vec3i> features;
			features = featureextraction::StrokePositionExtraction(deflist[i]);
			std::cout << std::endl << "Features extracted from " << deflist[i];
			for (size_t h = 0; h < features.size(); h++) {
				ofss << features[h][0] << " ";
				ofsx << features[h][1] << " ";
				ofsy << features[h][2] << " ";
			}
			ofss << "-" << std::endl;
			ofsx << "-" << std::endl;
			ofsy << "-" << std::endl;
			if (defwidth < features.size()) {
				defwidth = features.size();
			}
		}
		defheight = deflist.size();

		cv::Size defsize(defwidth, defheight);
		ofss.close();
		ofsx.close();
		ofsy.close();

		// build the model for offensive
		strategyprediction::TrainStrokeOffensive("features/os_feature.txt", offsize);
		std::cout << std::endl << "Stroke offensive model is ready to use!";
		strategyprediction::TrainXYOffensive("features/ox_feature.txt", "features/oy_feature.txt", framesize, offsize);
		std::cout << std::endl << "XY offensive model is ready to use!";

		// build the model for defensive
		strategyprediction::TrainStrokeDefensive("features/ds_feature.txt", defsize);
		std::cout << std::endl << "Stroke defensive model is ready to use!";
		strategyprediction::TrainXYDefensive("features/dx_feature.txt", "features/dy_feature.txt", framesize, defsize);
		std::cout << std::endl << "XY defensive model is ready to use!";
		std::cout << std::endl << "Training is finished!" << std::endl;
	}

	float Testing(std::vector<std::string> testing_data) {
		int width = 0, height = 0;

		std::ifstream ifs;
		std::ofstream ofss, ofsx, ofsy;

		std::cout << std::endl << "Testing is in progress...";

		// make sure testing_data is not empty
		if (testing_data.empty()) {
			std::cerr << std::endl << "Fill the training data first!" << std::endl;
			exit(EXIT_FAILURE);
		}

		cv::Size framesize = CheckFrameSize(testing_data[0]);

		// save offensive stroke and bottom player position (x,y) feature in a file
		ofss.open("features/ts_feature.txt");
		ofsx.open("features/tx_feature.txt");
		ofsy.open("features/ty_feature.txt");

		// process offensive list
		for (size_t i = 0; i < testing_data.size(); i++) {
			std::vector<cv::Vec3i> features;
			features = featureextraction::StrokePositionExtraction(testing_data[i]);
			std::cout << std::endl << "Features extracted from " << testing_data[i];

			for (size_t h = 0; h < features.size(); h++) {
				ofss << features[h][0] << " ";
				ofsx << features[h][1] << " ";
				ofsy << features[h][2] << " ";
			}
			ofss << "-" << std::endl;
			ofsx << "-" << std::endl;
			ofsy << "-" << std::endl;
			if (width < features.size()) {
				width = features.size();
			}
		}
		height = testing_data.size();

		cv::Size testsize(width, height);
		ofss.close();
		ofsx.close();
		ofsy.close();

		// Offensive check (all features from every play videos combined into these three file)
		std::vector<float> offtest = strategyprediction::TestingOffensive("features/ts_feature.txt", "features/tx_feature.txt", "features/ty_feature.txt", framesize, testsize);

		// Defensive check (all features from every play videos combined into these three file)
		std::vector<float> deftest = strategyprediction::TestingDefensive("features/ts_feature.txt", "features/tx_feature.txt", "features/ty_feature.txt", framesize, testsize);

		std::cout << std::endl << "The offensive log probabilities for each testing play are ";
		for (size_t i = 0; i < offtest.size(); i++) {
			std::cout << offtest[i] << " ";
		}
		std::cout << std::endl;

		std::cout << std::endl << "The defensive log probabilities for each testing play are ";
		for (size_t i = 0; i < deftest.size(); i++) {
			std::cout << deftest[i] << " ";
		}
		std::cout << std::endl;

		// variable to store final - 0 (offensive) and 1 (defensive)
		std::vector<int> tprobability;

		// offtest.size() equal to deftest.size()
		for (size_t i = 0; i < offtest.size(); i++) {
			if (offtest[i] < deftest[i]) {
				tprobability.push_back(0);
			}
			else {
				tprobability.push_back(1);
			}
		}

		for (size_t i = 0; i < tprobability.size(); i++) {
			std::cout << tprobability[i] << " ";
		}
		
		float accuracy = AccuracyCalculation(tprobability);
		std::cout << std::endl;

		return accuracy;
	}
} // namespace Intermediary

#endif INTERMEDIARY_H