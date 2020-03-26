
#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/core/core.hpp"                                                                                                     
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;

int main(int argc, char** argv){
	Mat frame, fgmask, segm;

	Ptr<BackgroundSubtractorGMG> fgbg =
	Algorithm::create<BackgroundSubtractorGMG>
	("BackgroundSubtractor.GMG");//基于自适应混合高斯背景建模的背景减除法

	if (fgbg.empty()){
		std::cerr << "Failed to create BackgroundSubtractor.GMG Algorithm." << std::endl;
		return -1;
	}

	fgbg->set("initializationFrames", 20);
	fgbg->set("decisionThreshold", 0.7);

	VideoCapture cap;
	cap.open(0);
// 	if (!cap.isOpened()){
// 		std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
// 		return -1;
// 	}
	namedWindow("FG Segmentation", WINDOW_NORMAL);
	for (;;){
		if (!cap.read(frame)){
			break;
		}

		(*fgbg)(frame, fgmask);

		frame.copyTo(segm);
		IplImage ImaskCodeBook = fgmask;
		cvSegmentFGMask(&ImaskCodeBook);
		add(frame, Scalar(100, 100, 0), segm, fgmask);

		imshow("FG Segmentation", segm);
		imshow("mask", fgmask);

		int c = waitKey(30);
		if (c == 'q' || c == 'Q' || (c & 255) == 27)
			break;
	}

	return 0;
}