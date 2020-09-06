#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector> 
#include "FastConerDetector.h"

void FastConerDetector::Fast(cv::Mat& src, std::vector<cv::KeyPoint>& KP){
	std::vector<cv::KeyPoint> keyPoints;
	cv::Ptr<cv::FeatureDetector> fast = cv::FastFeatureDetector::create(TH, nomaxsuprression, detect_type);
	fast->detect(src, keyPoints);
	//KP.assign(keyPoints.begin(), keyPoints.end());
	KP = keyPoints;
}

void FastConerDetector::NMS(cv::Mat &src, std::vector<cv::KeyPoint> & kp){
	int r = diameter / 2;
	int w = Size, h = Size;
	for (int i = 0; i < kp.size(); ++i){
		for (int j = i + 1; j < kp.size(); ++j){
			int pta_x = kp[i].pt.x, pta_y = kp[i].pt.y;
			int ptb_x = kp[j].pt.x, ptb_y = kp[j].pt.y;

			if (((pta_x - ((w - 1) / 2))>0 && (pta_x + ((w - 1) / 2))<src.cols && (pta_y - ((h - 1) / 2))>0 && (pta_y + ((h - 1) / 2)) < src.rows) && ((ptb_x - ((w - 1) / 2))>0 && (ptb_x + ((w - 1) / 2))<src.cols && (ptb_y - ((h - 1) / 2))>0 && (ptb_y + ((h - 1) / 2)) < src.rows))  {
				int d_x = abs(kp[i].pt.x - kp[j].pt.x);
				int d_y = abs(kp[i].pt.y - kp[j].pt.y);
				if ((d_x*d_x) + (d_y * d_y) <= r * r){
					cv::Rect area_a(pta_x - ((w - 1) / 2), pta_y - ((h - 1) / 2), w, h);
					cv::Rect area_b(ptb_x - ((w - 1) / 2), ptb_y - ((h - 1) / 2), w, h);
					double mean_a = cv::mean(src(area_a))[0];
					double mean_b = cv::mean(src(area_b))[0];
					if (mean_a < mean_b){
						std::vector<cv::KeyPoint>::iterator it = kp.begin() + i;
						kp.erase(it);
						NMS(src, kp);
					}
					else if (mean_a >= mean_b){
						std::vector<cv::KeyPoint>::iterator it = kp.begin() + j;
						kp.erase(it);
						NMS(src, kp);
					}
				}
			}
			else{
				std::vector<cv::KeyPoint>::iterator it1 = kp.begin() + i;
				std::vector<cv::KeyPoint>::iterator it2 = kp.begin() + j;
				kp.erase(it1);
				kp.erase(it2);
				NMS(src, kp);
			}
		}
	}
}