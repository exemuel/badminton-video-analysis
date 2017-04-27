#ifndef STRATEGYPREDICTION_H
#define STRATEGYPREDICTION_H

namespace strategyprediction {
	/**
	Train stroke for HMM Offensive.

	@param filename Stroke offensive.
	@param osize Dimension of the input.
	*/
	void TrainStrokeOffensive(std::string filename, cv::Size osize) {
		int cnt = 0, rows = osize.height, cols = osize.width;

		std::string content;
		cv::Mat seq = cv::Mat(rows, cols, CV_64F, double(0));
		std::ifstream tso(filename);

		// Feature selection
		int tr = 0, tc = 0;
		while (tso >> content) {
			if (content != "-") {
				if (tc <= cols) {
					seq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						seq.at<int>(tr, tc + i) = seq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}

		///************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq.rows; j++) {
		//	for (size_t i = 0; i < seq.cols; i++) {
		//		std::cout << seq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		///*********************************************************************************************************/

		CvHMM hmm;
		double TRGUESSdata[] = { 2.0 / 2.0 , 0.0 / 2.0 ,
			2.0 / 2.0 , 0.0 / 2.0 };
		cv::Mat TRGUESS = cv::Mat(2, 2, CV_64F, TRGUESSdata).clone();
		double EMITGUESSdata[] = { 1.0 / 3.0 , 1.0 / 3.0 , 1.0 / 3.0 ,
			1.0 / 3.0 , 1.0 / 3.0, 1.0 / 3.0 };
		cv::Mat EMITGUESS = cv::Mat(2, 3, CV_64F, EMITGUESSdata).clone();
		double INITGUESSdata[] = { 0.5  , 0.5 };
		cv::Mat INITGUESS = cv::Mat(1, 2, CV_64F, INITGUESSdata).clone();

		hmm.train(seq, 100, TRGUESS, EMITGUESS, INITGUESS);

		cv::FileStorage storage1("model/TRANStso.yml", cv::FileStorage::WRITE);
		storage1 << "TRANStso" << TRGUESS;
		storage1.release();
		cv::FileStorage storage2("model/EMIStso.yml", cv::FileStorage::WRITE);
		storage2 << "EMIStso" << EMITGUESS;
		storage2.release();
		cv::FileStorage storage3("model/INITtso.yml", cv::FileStorage::WRITE);
		storage3 << "INITtso" << INITGUESS;
		storage3.release();
	}

	/**
	Train stroke for HMM Defensive.

	@param filename Stroke defensive.
	@param dsize Dimension of the input.
	*/
	void TrainStrokeDefensive(std::string filename, cv::Size dsize) {
		int cnt = 0, rows = dsize.height, cols = dsize.width;

		std::string content;
		cv::Mat seq = cv::Mat(rows, cols, CV_64F, double(0));
		std::ifstream tsd(filename);

		// Feature selection
		int tr = 0, tc = 0;
		while (tsd >> content) {
			if (content != "-") {
				if (tc <= cols) {
					seq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						seq.at<int>(tr, tc + i) = seq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq.rows; j++) {
		//	for (size_t i = 0; i < seq.cols; i++) {
		//		std::cout << seq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		CvHMM hmm;
		double TRGUESSdata[] = { 1.0 / 2.0 , 1.0 / 2.0 ,
			1.0 / 2.0 , 1.0 / 2.0 };
		cv::Mat TRGUESS = cv::Mat(2, 2, CV_64F, TRGUESSdata).clone();
		double EMITGUESSdata[] = { 1.0 / 3.0 , 1.0 / 3.0 , 1.0 / 3.0 ,
			1.0 / 3.0 , 1.0 / 3.0, 1.0 / 3.0 };
		cv::Mat EMITGUESS = cv::Mat(2, 3, CV_64F, EMITGUESSdata).clone();
		double INITGUESSdata[] = { 0.5  , 0.5 };
		cv::Mat INITGUESS = cv::Mat(1, 2, CV_64F, INITGUESSdata).clone();

		hmm.train(seq, 100, TRGUESS, EMITGUESS, INITGUESS);

		cv::FileStorage storage1("model/TRANStsd.yml", cv::FileStorage::WRITE);
		storage1 << "TRANStsd" << TRGUESS;
		storage1.release();
		cv::FileStorage storage2("model/EMIStsd.yml", cv::FileStorage::WRITE);
		storage2 << "EMIStsd" << EMITGUESS;
		storage2.release();
		cv::FileStorage storage3("model/INITtsd.yml", cv::FileStorage::WRITE);
		storage3 << "INITtsd" << INITGUESS;
		storage3.release();
	}

	/**
	Train xy position of bottom player for HMM Offensive.

	@param xfilename A file contain x position of bottom player.
	@param yfilename A file contain y position of bottom player.
	@param osize Dimension of frame_size.
	@param osize Dimension of xfilename and yfilename (must be same).
	*/
	void TrainXYOffensive(std::string xfilename, std::string yfilename, cv::Size fsize, cv::Size osize) {
		int cnt = 0, rows = osize.height, cols = osize.width;

		std::string content;
		cv::Mat xseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		cv::Mat yseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		std::ifstream tsd;

		// X Feature selection
		tsd.open(xfilename);
		int tr = 0, tc = 0;
		while (tsd >> content) {
			if (content != "-") {
				if (tc <= cols) {
					xseq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						xseq.at<int>(tr, tc + i) = xseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		tsd.close();

		// Y Feature selection
		tsd.open(yfilename);
		tr = 0, tc = 0;
		while (tsd >> content) {
			if (content != "-") {
				if (tc <= cols) {
					yseq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						yseq.at<int>(tr, tc + i) = yseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		tsd.close();

		// CV_64F is CvHMM requirement 
		cv::Mat res = cv::Mat(rows, cols, CV_64F, double(0));

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq.rows; j++) {
		//	for (size_t i = 0; i < seq.cols; i++) {
		//		std::cout << seq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		int binsize = 3;
		int xrange[] = { (int)19 * fsize.width / 100, (int)53 * fsize.width / 100 };
		int yrange[] = { (int)49.2 * fsize.height / 100, (int)92.7 * fsize.height / 100 };
		std::vector<cv::Point2i> pos = {};

		for (size_t j = 0; j < binsize; j++) {
			if (j == 0) {
				int x = xrange[0] + ((xrange[1] - xrange[0]) / binsize);
				int y = yrange[0] + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
			else {
				int x = pos[j - 1].x + ((xrange[1] - xrange[0]) / binsize);
				int y = pos[j - 1].y + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
		}

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < pos.size(); j++) {
		//	std::cout << pos[j] << " ";
		//}
		/*********************************************************************************************************/

		// indexing each int value to the nearest bin
		for (size_t j = 0; j < xseq.rows; j++) {
			for (size_t i = 0; i < xseq.cols; i++) {
				for (size_t h = 0; h < binsize; h++) {
					if (xseq.at<int>(j, i) >= pos[h].x && xseq.at<int>(j, i) < pos[h + 1].x) {
						for (size_t g = 0; g < binsize; g++) {
							if (yseq.at<int>(j, i) >= pos[g].y && yseq.at<int>(j, i) < pos[g + 1].y) {
								res.at<int>(j, i) = (g + (2 * g) + 1) + h;
							}
						}
					}
				}
			}
		}

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < res.rows; j++) {
		//	for (size_t i = 0; i < res.cols; i++) {
		//		std::cout << res.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		CvHMM hmm;
		double TRGUESSdata[] = { 1.0 / 2.0 , 1.0 / 2.0 ,
			1.0 / 2.0 , 1.0 / 2.0 };
		cv::Mat TRGUESS = cv::Mat(2, 2, CV_64F, TRGUESSdata).clone();
		double EMITGUESSdata[] = { 1.0 / 3.0 , 1.0 / 3.0 , 1.0 / 3.0 ,
			1.0 / 3.0 , 1.0 / 3.0, 1.0 / 3.0 };
		cv::Mat EMITGUESS = cv::Mat(2, 3, CV_64F, EMITGUESSdata).clone();
		double INITGUESSdata[] = { 0.5  , 0.5 };
		cv::Mat INITGUESS = cv::Mat(1, 2, CV_64F, INITGUESSdata).clone();

		hmm.train(res, 100, TRGUESS, EMITGUESS, INITGUESS);

		/************************************** Uncomment to See the Result **************************************/
		//hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
		/*********************************************************************************************************/

		cv::FileStorage storage1("model/TRANStxyo.yml", cv::FileStorage::WRITE);
		storage1 << "TRANStxyo" << TRGUESS;
		storage1.release();
		cv::FileStorage storage2("model/EMIStxyo.yml", cv::FileStorage::WRITE);
		storage2 << "EMIStxyo" << EMITGUESS;
		storage2.release();
		cv::FileStorage storage3("model/INITtxyo.yml", cv::FileStorage::WRITE);
		storage3 << "INITtxyo" << INITGUESS;
		storage3.release();
	}

	/**
	Train xy position of bottom player for HMM Defensive.

	@param xfilename A file contain x position of bottom player.
	@param yfilename A file contain y position of bottom player.
	@param osize Dimension of frame_size.
	@param osize Dimension of xfilename and yfilename (must be same).
	*/
	void TrainXYDefensive(std::string xfilename, std::string yfilename, cv::Size fsize, cv::Size dsize) {
		int cnt = 0, rows = dsize.height, cols = dsize.width;

		std::string content;
		cv::Mat xseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		cv::Mat yseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		std::ifstream tsd;

		// X Feature selection
		tsd.open(xfilename);
		int tr = 0, tc = 0;
		while (tsd >> content) {
			if (content != "-") {
				if (tc <= cols) {
					xseq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						xseq.at<int>(tr, tc + i) = xseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		tsd.close();

		// Y Feature selection
		tsd.open(yfilename);
		tr = 0, tc = 0;
		while (tsd >> content) {
			if (content != "-") {
				if (tc <= cols) {
					yseq.at<int>(tr, tc) = atoi(content.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						yseq.at<int>(tr, tc + i) = yseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		tsd.close();

		// CV_64F is CvHMM requirement 
		cv::Mat res = cv::Mat(rows, cols, CV_64F, double(0));

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq.rows; j++) {
		//	for (size_t i = 0; i < seq.cols; i++) {
		//		std::cout << seq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		int binsize = 3;
		int xrange[] = { (int)19 * fsize.width / 100, (int)53 * fsize.width / 100 };
		int yrange[] = { (int)49.2 * fsize.height / 100, (int)92.7 * fsize.height / 100 };
		std::vector<cv::Point2i> pos = {};

		for (size_t j = 0; j < binsize; j++) {
			if (j == 0) {
				int x = xrange[0] + ((xrange[1] - xrange[0]) / binsize);
				int y = yrange[0] + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
			else {
				int x = pos[j - 1].x + ((xrange[1] - xrange[0]) / binsize);
				int y = pos[j - 1].y + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
		}

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < pos.size(); j++) {
		//	std::cout << pos[j] << " ";
		//}
		/*********************************************************************************************************/

		// indexing each int value to the nearest bin
		for (size_t j = 0; j < xseq.rows; j++) {
			for (size_t i = 0; i < xseq.cols; i++) {
				for (size_t h = 0; h < binsize; h++) {
					if (xseq.at<int>(j, i) >= pos[h].x && xseq.at<int>(j, i) < pos[h + 1].x) {
						for (size_t g = 0; g < binsize; g++) {
							if (yseq.at<int>(j, i) >= pos[g].y && yseq.at<int>(j, i) < pos[g + 1].y) {
								res.at<int>(j, i) = (g + (2 * g) + 1) + h;
							}
						}
					}
				}
			}
		}

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < res.rows; j++) {
		//	for (size_t i = 0; i < res.cols; i++) {
		//		std::cout << res.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		/*********************************************************************************************************/

		CvHMM hmm;
		double TRGUESSdata[] = { 1.0 / 2.0 , 1.0 / 2.0 ,
			1.0 / 2.0 , 1.0 / 2.0 };
		cv::Mat TRGUESS = cv::Mat(2, 2, CV_64F, TRGUESSdata).clone();
		double EMITGUESSdata[] = { 1.0 / 3.0 , 1.0 / 3.0 , 1.0 / 3.0 ,
			1.0 / 3.0 , 1.0 / 3.0, 1.0 / 3.0 };
		cv::Mat EMITGUESS = cv::Mat(2, 3, CV_64F, EMITGUESSdata).clone();
		double INITGUESSdata[] = { 0.5  , 0.5 };
		cv::Mat INITGUESS = cv::Mat(1, 2, CV_64F, INITGUESSdata).clone();

		hmm.train(res, 100, TRGUESS, EMITGUESS, INITGUESS);

		/************************************** Uncomment to See the Result **************************************/
		//hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
		/*********************************************************************************************************/

		cv::FileStorage storage1("model/TRANStxyd.yml", cv::FileStorage::WRITE);
		storage1 << "TRANStxyd" << TRGUESS;
		storage1.release();
		cv::FileStorage storage2("model/EMIStxyd.yml", cv::FileStorage::WRITE);
		storage2 << "EMIStxyd" << EMITGUESS;
		storage2.release();
		cv::FileStorage storage3("model/INITtxyd.yml", cv::FileStorage::WRITE);
		storage3 << "INITtxyd" << INITGUESS;
		storage3.release();
	}

	/**
	Testing HMM Offensive.

	@param f1 Offensive's stroke feature file.
	@param f2 Offensive's x feature file.
	@param f3 Offensive's y feature file.
	@param fsize Dimension of frame_size.
	@param osize Dimension of f1, f2, and f3 (must be same).
	*/
	std::vector<float> TestingOffensive(std::string f1, std::string f2, std::string f3, cv::Size fsize, cv::Size osize) {
		int cnt = 0, rows = osize.height, cols = osize.width;
		std::ifstream ifs;
		cv::FileStorage storage;

		cv::Mat TRANStso;
		storage.open("model/TRANStso.yml", cv::FileStorage::READ);
		storage["TRANStso"] >> TRANStso;
		storage.release();

		cv::Mat TRANStxyo;
		storage.open("model/TRANStxyo.yml", cv::FileStorage::READ);
		storage["TRANStxyo"] >> TRANStxyo;
		storage.release();

		cv::Mat EMIStso;
		storage.open("model/EMIStso.yml", cv::FileStorage::READ);
		storage["EMIStso"] >> EMIStso;
		storage.release();

		cv::Mat EMIStxyo;
		storage.open("model/EMIStxyo.yml", cv::FileStorage::READ);
		storage["EMIStxyo"] >> EMIStxyo;
		storage.release();

		cv::Mat INITtso;
		storage.open("model/INITtso.yml", cv::FileStorage::READ);
		storage["INITtso"] >> INITtso;
		storage.release();

		cv::Mat INITtxyo;
		storage.open("model/INITtxyo.yml", cv::FileStorage::READ);
		storage["INITtxyo"] >> INITtxyo;
		storage.release();

		/************************************************** Stroke **************************************************/

		std::string content1;
		cv::Mat sseq = cv::Mat(rows, cols, CV_64F, double(0));
		ifs.open(f1);

		// Feature selection
		int tr = 0, tc = 0;
		while (ifs >> content1) {
			if (content1 == "-") {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						sseq.at<int>(tr, tc + i) = sseq.at<int>(tr, tc - 1);
					}
				}
				tr++;
				tc = 0;
			}
			sseq.at<int>(tr, tc) = atoi(content1.c_str());
			tc++;
		}
		ifs.clear(); ifs.close();

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq1.rows; j++) {
		//	for (size_t i = 0; i < seq1.cols; i++) {
		//		std::cout << seq1.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		//std::cin.get(); // pause
		/*********************************************************************************************************/

		/*********************************************************************************************************/

		/************************************************** XY ***************************************************/

		std::string content2;
		cv::Mat xseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		cv::Mat yseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		ifs.open(f2);

		// X Feature selection
		ifs.open(f2);
		tr = 0, tc = 0;
		while (ifs >> content2) {
			if (content2 != "-") {
				if (tc <= cols) {
					xseq.at<int>(tr, tc) = atoi(content2.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						xseq.at<int>(tr, tc + i) = xseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		ifs.clear(); ifs.close();

		// Y Feature selection
		ifs.open(f3);
		tr = 0, tc = 0;
		while (ifs >> content2) {
			if (content2 != "-") {
				if (tc <= cols) {
					yseq.at<int>(tr, tc) = atoi(content2.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						yseq.at<int>(tr, tc + i) = yseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		ifs.clear(); ifs.close();

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < xseq.rows; j++) {
		//	for (size_t i = 0; i < xseq.cols; i++) {
		//		std::cout << xseq.at<int>(j, i) << ", " << yseq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		//std::cin.get(); // pause
		/*********************************************************************************************************/

		cv::Mat xyseq = cv::Mat(rows, cols, CV_64F, double(0)); // CV_64F is CvHMM requirement 

		int binsize = 3;
		int xrange[] = { (int)19 * fsize.width / 100, (int)53 * fsize.width / 100 };
		int yrange[] = { (int)49.2 * fsize.height / 100, (int)92.7 * fsize.height / 100 };
		std::vector<cv::Point2i> pos = {};

		for (size_t j = 0; j < binsize; j++) {
			if (j == 0) {
				int x = xrange[0] + ((xrange[1] - xrange[0]) / binsize);
				int y = yrange[0] + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
			else {
				int x = pos[j - 1].x + ((xrange[1] - xrange[0]) / binsize);
				int y = pos[j - 1].y + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
		}

		// indexing each int value to the nearest bin
		for (size_t j = 0; j < xseq.rows; j++) {
			for (size_t i = 0; i < xseq.cols; i++) {
				for (size_t h = 0; h < binsize; h++) {
					if (xseq.at<int>(j, i) >= pos[h].x && xseq.at<int>(j, i) < pos[h + 1].x) {
						for (size_t g = 0; g < binsize; g++) {
							if (yseq.at<int>(j, i) >= pos[g].y && yseq.at<int>(j, i) < pos[g + 1].y) {
								xyseq.at<int>(j, i) = (g + (2 * g) + 1) + h;
							}
						}
					}
				}
			}
		}

		/************************************************************************************************************/

		CvHMM hmm;
		//float epsilon = 0.0001;
		std::vector<float> alllogpseq;

		for (int i = 0; i < sseq.rows; i++) {
			double slogpseq, xylogpseq;
			cv::Mat spstates, sforward, sbackward, xypstates, xyforward, xybackward;
			hmm.decode(sseq.row(i), TRANStso, EMIStso, INITtso, slogpseq, spstates, sforward, sbackward);
			hmm.decode(xyseq.row(i), TRANStxyo, EMIStxyo, INITtxyo, xylogpseq, xypstates, xyforward, xybackward);

			if (std::isnan(slogpseq) == 1) {
				slogpseq = 0;
			}
			if (std::isnan(xylogpseq) == 1) {
				xylogpseq = 0;
			}

			float oflogpseq = slogpseq + xylogpseq;
			alllogpseq.push_back(oflogpseq);
		}

		return alllogpseq;
	}

	/**
	Testing HMM Defensive.

	@param f1 Defensive's stroke feature file.
	@param f2 Defensive's x feature file.
	@param f3 Defensive's y feature file
	@param fsize Dimension of frame_size.
	@param osize Dimension of f1, f2, and f3 (must be same).
	*/
	std::vector<float> TestingDefensive(std::string f1, std::string f2, std::string f3, cv::Size fsize, cv::Size osize) {
		int cnt = 0, rows = osize.height, cols = osize.width;
		std::ifstream ifs;
		cv::FileStorage storage;

		cv::Mat TRANStsd;
		storage.open("model/TRANStsd.yml", cv::FileStorage::READ);
		storage["TRANStsd"] >> TRANStsd;
		storage.release();

		cv::Mat TRANStxyd;
		storage.open("model/TRANStxyd.yml", cv::FileStorage::READ);
		storage["TRANStxyd"] >> TRANStxyd;
		storage.release();

		cv::Mat EMIStsd;
		storage.open("model/EMIStsd.yml", cv::FileStorage::READ);
		storage["EMIStsd"] >> EMIStsd;
		storage.release();

		cv::Mat EMIStxyd;
		storage.open("model/EMIStxyd.yml", cv::FileStorage::READ);
		storage["EMIStxyd"] >> EMIStxyd;
		storage.release();

		cv::Mat INITtsd;
		storage.open("model/INITtsd.yml", cv::FileStorage::READ);
		storage["INITtsd"] >> INITtsd;
		storage.release();

		cv::Mat INITtxyd;
		storage.open("model/INITtxyd.yml", cv::FileStorage::READ);
		storage["INITtxyd"] >> INITtxyd;
		storage.release();

		/************************************************** Stroke **************************************************/

		std::string content1;
		cv::Mat sseq = cv::Mat(rows, cols, CV_64F, double(0));
		ifs.open(f1);

		// Feature selection
		int tr = 0, tc = 0;
		while (ifs >> content1) {
			if (content1 == "-") {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						sseq.at<int>(tr, tc + i) = sseq.at<int>(tr, tc - 1);
					}
				}
				tr++;
				tc = 0;
			}
			sseq.at<int>(tr, tc) = atoi(content1.c_str());
			tc++;
		}
		ifs.clear(); ifs.close();

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < seq1.rows; j++) {
		//	for (size_t i = 0; i < seq1.cols; i++) {
		//		std::cout << seq1.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		//std::cin.get(); // pause
		/*********************************************************************************************************/

		/*********************************************************************************************************/

		/************************************************** XY ***************************************************/

		std::string content2;
		cv::Mat xseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		cv::Mat yseq = cv::Mat(rows, cols, CV_32FC1, double(0));
		ifs.open(f2);

		// X Feature selection
		ifs.open(f2);
		tr = 0, tc = 0;
		while (ifs >> content2) {
			if (content2 != "-") {
				if (tc <= cols) {
					xseq.at<int>(tr, tc) = atoi(content2.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						xseq.at<int>(tr, tc + i) = xseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		ifs.clear(); ifs.close();

		// Y Feature selection
		ifs.open(f3);
		tr = 0, tc = 0;
		while (ifs >> content2) {
			if (content2 != "-") {
				if (tc <= cols) {
					yseq.at<int>(tr, tc) = atoi(content2.c_str());
					tc++;
				}
			}
			else {
				if (tc < cols) {
					int gap = cols - tc;
					for (size_t i = 0; i < gap; i++) {
						yseq.at<int>(tr, tc + i) = yseq.at<int>(tr, tc - 1);
					}
				}
				tr++; tc = 0;
			}
		}
		ifs.clear(); ifs.close();

		/************************************** Uncomment to See the Result **************************************/
		//for (size_t j = 0; j < xseq.rows; j++) {
		//	for (size_t i = 0; i < xseq.cols; i++) {
		//		std::cout << xseq.at<int>(j, i) << ", " << yseq.at<int>(j, i) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		//std::cin.get(); // pause
		/*********************************************************************************************************/

		cv::Mat xyseq = cv::Mat(rows, cols, CV_64F, double(0)); // CV_64F is CvHMM requirement 

		int binsize = 3;
		int xrange[] = { (int)19 * fsize.width / 100, (int)53 * fsize.width / 100 };
		int yrange[] = { (int)49.2 * fsize.height / 100, (int)92.7 * fsize.height / 100 };
		std::vector<cv::Point2i> pos = {};

		for (size_t j = 0; j < binsize; j++) {
			if (j == 0) {
				int x = xrange[0] + ((xrange[1] - xrange[0]) / binsize);
				int y = yrange[0] + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
			else {
				int x = pos[j - 1].x + ((xrange[1] - xrange[0]) / binsize);
				int y = pos[j - 1].y + ((yrange[1] - yrange[0]) / binsize);
				pos.push_back(cv::Point2i(x, y));
			}
		}

		// indexing each int value to the nearest bin
		for (size_t j = 0; j < xseq.rows; j++) {
			for (size_t i = 0; i < xseq.cols; i++) {
				for (size_t h = 0; h < binsize; h++) {
					if (xseq.at<int>(j, i) >= pos[h].x && xseq.at<int>(j, i) < pos[h + 1].x) {
						for (size_t g = 0; g < binsize; g++) {
							if (yseq.at<int>(j, i) >= pos[g].y && yseq.at<int>(j, i) < pos[g + 1].y) {
								xyseq.at<int>(j, i) = (g + (2 * g) + 1) + h;
							}
						}
					}
				}
			}
		}
		
		/************************************************************************************************************/

		CvHMM hmm;
		float epsilon = 0.0001;
		std::vector<float> alllogpseq;
		for (int i = 0; i < sseq.rows; i++) {
			double slogpseq, xylogpseq;
			cv::Mat spstates, sforward, sbackward, xypstates, xyforward, xybackward;
			hmm.decode(sseq.row(i), TRANStsd, EMIStsd, INITtsd, slogpseq, spstates, sforward, sbackward);
			hmm.decode(xyseq.row(i), TRANStxyd, EMIStxyd, INITtxyd, xylogpseq, xypstates, xyforward, xybackward);

			if (std::isnan(slogpseq) == 1) {
				slogpseq = epsilon;
			}
			if (std::isnan(xylogpseq) == 1) {
				xylogpseq = epsilon;
			}

			float dflogpseq = slogpseq + xylogpseq;
			alllogpseq.push_back(dflogpseq);
		}

		return alllogpseq;
	}
} // namespace strategyprediction

#endif STRATEGYPREDICTION_H
