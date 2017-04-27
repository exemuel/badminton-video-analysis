#ifndef FEATUREEXTRACTION_H
#define FEATUREEXTRACTION_H

namespace featureextraction {
	cv::Mat TopPlayerFrame(cv::Mat src) {
		cv::Mat dst = src.clone();

		// constrains
		int left = floor((21 * src.cols) / 100);
		//std::cout << left << std::endl;
		int right = floor((79 * src.cols) / 100);
		//std::cout << right << std::endl;
		int top = floor((18 * src.rows) / 100);
		//std::cout << top << std::endl;
		int bottom = floor((58 * src.rows) / 100);
		//std::cout << bottom << std::endl;

		for (size_t j = 0; j < src.rows; j++) {
			for (size_t i = 0; i < src.cols; i++) {
				if (j < top || j > bottom) {
					dst.at<uchar>(j, i) = 255;
				}
				if (i < left || i > right) {
					dst.at<uchar>(j, i) = 255;
				}
			}
		}

		return dst;
	}

	cv::Mat BottomPlayerFrame(cv::Mat src) {
		cv::Mat dst = src.clone();

		// constrains
		int left = floor((21 * src.cols) / 100);
		//std::cout << left << std::endl;
		int right = floor((79 * src.cols) / 100);
		//std::cout << right << std::endl;
		int top = floor((45 * src.rows) / 100);
		//std::cout << top << std::endl;
		int bottom = floor((94 * src.rows) / 100);
		//std::cout << bottom << std::endl;

		for (size_t j = 0; j < src.rows; j++) {
			for (size_t i = 0; i < src.cols; i++) {
				if (j < top || j > bottom) {
					dst.at<uchar>(j, i) = 255;
				}
				if (i < left || i > right) {
					dst.at<uchar>(j, i) = 255;
				}
			}
		}

		return dst;
	}

	cv::Rect NormalizeTopROI(cv::Mat frm, cv::Rect boundingBox) {
		cv::Point center = cv::Point(boundingBox.x + (boundingBox.width / 2), boundingBox.y + (boundingBox.height / 2));
		cv::Rect returnRect = cv::Rect(center.x - 30, center.y - 45, 60, 90);

		if (returnRect.x < 0)returnRect.x = 0;
		if (returnRect.y < 0)returnRect.y = 0;
		if (returnRect.x + returnRect.width >= frm.cols)returnRect.width = frm.cols - returnRect.x;
		if (returnRect.y + returnRect.height >= frm.rows)returnRect.height = frm.rows - returnRect.y;

		return returnRect;
	}

	cv::Rect NormalizeBottomROI(cv::Mat frm, cv::Rect boundingBox) {
		cv::Point center = cv::Point(boundingBox.x + (boundingBox.width / 2), boundingBox.y + (boundingBox.height / 2));
		cv::Rect returnRect = cv::Rect(center.x - 35, center.y - 63, 70, 126);

		if (returnRect.x < 0)returnRect.x = 0;
		if (returnRect.y < 0)returnRect.y = 0;
		if (returnRect.x + returnRect.width >= frm.cols)returnRect.width = frm.cols - returnRect.x;
		if (returnRect.y + returnRect.height >= frm.rows)returnRect.height = frm.rows - returnRect.y;

		return returnRect;
	}

	cv::Rect TopPlayerRectEstimation(cv::Mat frame, cv::Ptr<cv::BackgroundSubtractor> gmm) {
		cvtColor(frame, frame, CV_BGR2GRAY);

		cv::Mat tpimage = frame.clone();

		// >>>>> Noise smoothing
		cv::GaussianBlur(frame, tpimage, cv::Size(3, 3), 3.0, 3.0);
		// <<<<< Noise smoothing

		tpimage = TopPlayerFrame(tpimage);

		// Background Subtraction
		cv::Mat fgMaskGMM;
		gmm->apply(tpimage, fgMaskGMM);

		cv::Mat skinimage;
		// shapes for morphology operators kernel
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::dilate(fgMaskGMM, skinimage, element, cv::Point(-1, -1), 2, 1, 1);
		cv::erode(skinimage, skinimage, element, cv::Point(-1, -1), 2, 1, 1);

		// Find contours
		cv::Mat contourimage;
		contourimage = skinimage.clone();
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(contourimage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		cv::Rect up_bounding_rect, up_bounding_rect2(tpimage.cols, tpimage.rows, 0, 0);

		/// Iterate through each contour
		for (size_t i = 0; i < contours.size(); i++) {
			//  Find the area of contour
			up_bounding_rect = boundingRect(contours[i]);
			if (up_bounding_rect.y < up_bounding_rect2.y && up_bounding_rect.width > 15 && up_bounding_rect.height > 15) {
				up_bounding_rect2 = up_bounding_rect;
			}
		}

		cv::Scalar color(255, 255, 255);

		// Check the size of the bounding box
		if (up_bounding_rect2.width < 60 && up_bounding_rect2.height < 90) {
			up_bounding_rect2 = NormalizeTopROI(tpimage, up_bounding_rect2);
		}

		return up_bounding_rect2;
	}

	cv::Rect BottomPlayerRectEstimation(cv::Mat frame, cv::Ptr<cv::BackgroundSubtractor> gmm) {
		cvtColor(frame, frame, CV_BGR2GRAY);

		cv::Mat bpimage = frame.clone();

		// >>>>> Noise smoothing
		cv::GaussianBlur(frame, bpimage, cv::Size(3, 3), 3.0, 3.0);
		// <<<<< Noise smoothing

		bpimage = BottomPlayerFrame(bpimage);


		// Background Subtraction
		cv::Mat fgMaskGMM;
		gmm->apply(bpimage, fgMaskGMM);

		cv::Mat skinimage;
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::dilate(fgMaskGMM, skinimage, element, cv::Point(-1, -1), 2, 1, 1);
		cv::erode(skinimage, skinimage, element, cv::Point(-1, -1), 2, 1, 1);

		// Find contours
		cv::Mat contourimage;
		contourimage = skinimage.clone();
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> bpimage_hierarchy;
		findContours(contourimage, contours, bpimage_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		cv::Rect bp_bounding_rect, bp_bounding_rect2(0, 0, 0, 0);

		// Iterate through each contour
		for (size_t i = 0; i < contours.size(); i++) {
			//  Find the area of contour
			bp_bounding_rect = boundingRect(contours[i]);
			if (bp_bounding_rect.y > bp_bounding_rect2.y && bp_bounding_rect.width > 15 && bp_bounding_rect.height > 15) {
				bp_bounding_rect2 = bp_bounding_rect;
			}
		}

		cv::Scalar color(255, 255, 255);

		// Check the size of the bounding box
		if (bp_bounding_rect2.width < 70 && bp_bounding_rect2.height < 126) {
			bp_bounding_rect2 = NormalizeBottomROI(frame, bp_bounding_rect2);
		}

		// Draw the largest contour using previously stored index
		cv::rectangle(frame, bp_bounding_rect2, color, 1, 8, 0);

		return bp_bounding_rect2;
	}

	std::vector<cv::Rect> PlayerRectExtraction(std::string filename) {
		cv::VideoCapture stream(filename);
		if (!stream.isOpened()) {
			// error in opening the video input
			std::cerr << "Unable to open video file: " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		cv::Mat image, pimage, rimage, tpimage, bpimage, stpimage, sbpimage, tpimage2, bpimage2;
		cv::Mat tpfgmaskGMM, bpfgmaskGMM;
		cv::Rect tplayer_rect, bplayer_rect;
		cv::Ptr<cv::BackgroundSubtractor> GMM; // Gaussian Mixture Model
		
		std::vector<cv::Rect> playersposition;

		GMM = cv::createBackgroundSubtractorMOG2(500, 16, false);

		while (true) {
			stream >> image;
			if (image.empty()) {
				//std::cout << "End...";
				break;
			}

			cvtColor(image, tpimage, CV_BGR2GRAY);
			cvtColor(image, bpimage, CV_BGR2GRAY);

			tplayer_rect = TopPlayerRectEstimation(image, GMM);
			bplayer_rect = BottomPlayerRectEstimation(image, GMM);

			playersposition.push_back(tplayer_rect);
			playersposition.push_back(bplayer_rect);
		}

		return playersposition;
	}

	std::vector<cv::Rect> TopPlayerRectRefinement(std::vector<cv::Rect> inputrectangle, std::string filename) {
		cv::VideoCapture stream(filename);
		if (!stream.isOpened()) {
			std::cerr << "Unable to open video file: " << filename << " !" << std::endl;
			exit(EXIT_FAILURE);
		}

		// grab the first frame
		cv::Mat frame_rgb;
		stream.read(frame_rgb);

		std::vector<cv::Rect> outputrectangle;

		for (size_t i = 0; i < inputrectangle.size(); i++) {
			outputrectangle.push_back(inputrectangle[i]);
		}

		for (size_t i = 0; i < inputrectangle.size(); i++) {
			// check and correct the x value
			if (inputrectangle[i].x == 0) {
				outputrectangle[i].x = (int)(frame_rgb.cols / 2) - (int)(4.1 * frame_rgb.cols / 100);
			}

			// check and correct the y value
			if (inputrectangle[i].y == 0) {
				outputrectangle[i].y = (int)(frame_rgb.rows / 2) - (int)(18.75 * frame_rgb.rows / 100);
			}

			// check and correct the width value
			if (inputrectangle[i].width < (5.86 * frame_rgb.cols / 100) || inputrectangle[i].width > (19.91 * frame_rgb.cols / 100)) {
				outputrectangle[i].width = (int)(15 * frame_rgb.cols / 100);
			}

			// check and correct the height value
			if (inputrectangle[i].height < (10.42 * frame_rgb.rows / 100) || inputrectangle[i].height > (35.42 * frame_rgb.rows / 100)) {
				outputrectangle[i].height = (int)(26.25 * frame_rgb.rows / 100);
			}
		}

		// correction so that the rectangle not exceed the frame size
		for (size_t i = 0; i < outputrectangle.size() - 1; i++) {
			if (outputrectangle[i].x + outputrectangle[i].width > frame_rgb.cols) {
				outputrectangle[i].x = outputrectangle[i].x - ((outputrectangle[i].x + outputrectangle[i].width) - frame_rgb.cols);
			}
			if (outputrectangle[i].y + outputrectangle[i].height > frame_rgb.rows) {
				outputrectangle[i].y = outputrectangle[i].y - ((outputrectangle[i].y + outputrectangle[i].height) - frame_rgb.rows);
			}
		}

		return outputrectangle;
	}

	std::vector<cv::Rect> BottomPlayerRectRefinement(std::vector<cv::Rect> inputrectangle, std::string filename) {
		cv::VideoCapture stream(filename);
		if (!stream.isOpened()) {
			std::cerr << "Unable to open video file: " << filename << " !" << std::endl;
			exit(EXIT_FAILURE);
		}

		// grab the first frame
		cv::Mat frame_rgb;
		stream.read(frame_rgb);

		std::vector<cv::Rect> outputrectangle;

		for (size_t i = 0; i < inputrectangle.size(); i++) {
			outputrectangle.push_back(inputrectangle[i]);
		}

		for (size_t i = 0; i < inputrectangle.size(); i++) {
			// Check and correct the x value
			if (inputrectangle[i].x == 0) {
				outputrectangle[i].x = (int)(frame_rgb.cols / 2) - (int)(4.1 * frame_rgb.cols / 100);
			}

			// Check and correct the y value
			if (inputrectangle[i].y == 0) {
				outputrectangle[i].y = (int)(frame_rgb.rows / 2);
			}

			// Check and correct the width value
			if (inputrectangle[i].width < (5.86 * frame_rgb.cols / 100) || inputrectangle[i].width > (19.91 * frame_rgb.cols / 100)) {
				outputrectangle[i].width = (int)(15 * frame_rgb.cols / 100);
			}

			// Check and correct the height value
			if (inputrectangle[i].height < (10.42 * frame_rgb.rows / 100) || inputrectangle[i].height > (35.42 * frame_rgb.rows / 100)) {
				outputrectangle[i].height = (int)(26.25 * frame_rgb.rows / 100);
			}
		}

		// Correction so that the rectangle not exceed the frame size
		for (size_t i = 0; i < outputrectangle.size(); i++) {
			if (frame_rgb.cols < outputrectangle[i].x + outputrectangle[i].width) {
				outputrectangle[i].x = outputrectangle[i].x - ((outputrectangle[i].x + outputrectangle[i].width) - frame_rgb.cols);
			}
		}
		for (size_t i = 0; i < outputrectangle.size(); i++) {
			if (frame_rgb.rows < outputrectangle[i].y + outputrectangle[i].height) {
				outputrectangle[i].y = outputrectangle[i].y - ((outputrectangle[i].y + outputrectangle[i].height) - frame_rgb.rows);
			}
		}

		return outputrectangle;
	}
	
	std::vector<cv::Vec3i> StrokePositionExtraction(std::string filename) {
		cv::VideoCapture stream(filename);
		if (!stream.isOpened()) {
			std::cerr << "Unable to open video file: " << filename << " !" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		// first frame
		cv::Mat frame_rgb;

		std::vector<cv::Rect> playersrect = PlayerRectExtraction(filename);
		std::vector<cv::Rect> topplayerrect;
		std::vector<cv::Rect> bottomplayerrect;

		for (size_t i = 0; i < playersrect.size(); i+=2) {
			topplayerrect.push_back(playersrect[i]);
			bottomplayerrect.push_back(playersrect[i + 1]);
		}

		std::vector<cv::Rect> rtopplayerrect = TopPlayerRectRefinement(topplayerrect, filename);
		std::vector<cv::Rect> rbottomplayerrect = BottomPlayerRectRefinement(bottomplayerrect, filename);

		// variables for the following loop
		int count = 0;
		cv::Mat testingmat;
		std::vector<cv::Vec3i> features;

		while (true) {
			stream >> frame_rgb;
			if (frame_rgb.empty()) {
				//std::cout << "End...\n" << std::endl;
				break;
			}

			cv::Mat bottomplayer = cv::Mat(frame_rgb, rbottomplayerrect[count]);
			cv::Size size(120, 150);
			cv::resize(bottomplayer, bottomplayer, size);

			// load stroke model
			cv::Ptr<cv::ml::SVM> msvm = cv::ml::SVM::create();
			msvm = cv::Algorithm::load<cv::ml::SVM>("utilities/stroke_model.xml");

			// classify the stroke (stroke feature)
			cv::HOGDescriptor test_hog;
			std::vector<float> test_descriptors_values;
			std::vector<cv::Point> test_locations;
			test_hog.compute(bottomplayer, test_descriptors_values, cv::Size(32, 32), cv::Size(0, 0), test_locations);

			std::vector<std::vector<float>> test_v_descriptors_values;
			test_v_descriptors_values.push_back(test_descriptors_values);

			int row2 = test_v_descriptors_values.size(), col2 = test_v_descriptors_values[0].size();

			testingmat = cv::Mat::zeros(row2, col2, CV_32FC1);
			for (size_t j = 0; j < testingmat.rows; j++) {
				for (size_t i = 0; i < testingmat.cols; i++) {
					testingmat.at<float>(j, i) = test_v_descriptors_values[j][i];
				}
			}

			int stroke = msvm->predict(testingmat);

			// calculate bottom player position (x and y feature)
			int x = rbottomplayerrect[count].x + (rbottomplayerrect[count].width / 2);
			int y = rbottomplayerrect[count].y + rbottomplayerrect[count].height;

			features.push_back(cv::Vec3i(stroke, x, y));

			//std::cout << features[count] << std::endl;

			count++;
			bottomplayer.release();
		}

		return features;
	}
} // namespace featureextraction

#endif FEATUREEXTRACTION_H
