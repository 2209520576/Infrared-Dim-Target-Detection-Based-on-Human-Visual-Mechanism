#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector> 
#include <iostream>
#include "MultiGray_Measure.h"


void MultiGray_Measure::Threshod_Seg(cv::Mat src, cv::Mat& dst, double k){

	cv::Mat mean_src, stddev_src;
	cv::meanStdDev(src, mean_src, stddev_src); // the mean and the standard deviation of the final contrast map
	double Ic = mean_src.at<double>(0, 0);
	double sigma = stddev_src.at<double>(0, 0);
	double TH = Ic + k*sigma;
	for (int i = 0; i < src.rows; ++i){
		for (int j = 0; j < src.cols; ++j){
			double pix = src.at<uchar>(i, j);
			if (pix <= TH)
				dst.at<uchar>(i, j) = 0.0;
		}
	}
}

void MultiGray_Measure::Multi_grayscale(cv::Mat src, cv::Mat& dst){

	if (src.channels() > 1) cvtColor(src, src, CV_RGB2GRAY);
	if (src.type() == CV_8U)
		src.convertTo(src, CV_64F, 1 / 255.0, 0);
	int r = src.rows;
	int c = src.cols;
	
	//meanfilter
	cv::Mat I3, I5, I7, I9, I11;
	cv::blur(src, I3, cv::Size(3, 3));
	cv::blur(src, I5, cv::Size(5, 5));
	cv::blur(src, I7, cv::Size(7, 7));
	cv::blur(src, I9, cv::Size(9, 9));
	cv::blur(src, I11, cv::Size(11, 11));

	std::vector<cv::Mat> ave_filted; 
	cv::Mat H3 = cv::Mat(r, c, CV_64F);
	cv::Mat H5 = cv::Mat(r, c, CV_64F);
	cv::Mat H7 = cv::Mat(r, c, CV_64F);
	cv::Mat H9 = cv::Mat(r, c, CV_64F);
	H3 = I3 - I11;
	H5 = I5 - I11;
	H7 = I7 - I11;
	H9 = I9 - I11;
	
	ave_filted.push_back(H3);
	ave_filted.push_back(H5);
	ave_filted.push_back(H7);
	ave_filted.push_back(H9);

	cv::Mat II3, II5, II7, II9;
	cv::blur(src.mul(src), II3, cv::Size(3, 3));
	cv::blur(src.mul(src), II5, cv::Size(5, 5));
	cv::blur(src.mul(src), II7, cv::Size(7, 7));
	cv::blur(src.mul(src), II9, cv::Size(9, 9));

	cv::Mat Lvar3 = cv::Mat(r, c, CV_64F);
	cv::Mat Lvar5 = cv::Mat(r, c, CV_64F);
	cv::Mat Lvar7 = cv::Mat(r, c, CV_64F);
	cv::Mat Lvar9 = cv::Mat(r, c, CV_64F);
	Lvar3 = II3 - I3.mul(I3);
	Lvar5 = II5 - I5.mul(I5);
	Lvar7 = II7 - I7.mul(I7);
	Lvar9 = II9 - I9.mul(I9);
	std::vector<cv::Mat> Lvar; 
	Lvar.push_back(Lvar3);
	Lvar.push_back(Lvar5);
	Lvar.push_back(Lvar7);
	Lvar.push_back(Lvar9);
	
	//Computation of gray scale measure
	optimalSize_map = cv::Mat(r, c, CV_8U);
	dst = cv::Mat(r, c, CV_64F);
	size_t n = ave_filted.size();
	double DK[4] = {};
	double DK_lvar[4] = {};
	std::vector<double>::iterator vector_DK_lvar;
	double pix_Max, lvar_Max;
	for (int i = 0; i < r; ++i){
		for (int j = 0; j < c; ++j){
			for (int k = 0; k < n; ++k){
				double pix = ave_filted[k].at<double>(i, j); 
				double lvar = Lvar[k].at<double>(i, j);
				if (pix < 0)   
					pix = 0.0;
				DK[k] = pix*pix;
				DK_lvar[k] = DK[k] * lvar * lvar* lvar;
			}
			pix_Max = *(std::max_element)(DK, DK + 4);
			lvar_Max = *(std::max_element)(DK_lvar, DK_lvar + 4);
			int lvar_index = std::max_element(DK_lvar, DK_lvar + 4) - DK_lvar;
			optimalSize_map.at<uchar>(i, j) = (lvar_index + 1) * 2 + 1;
			dst.at<double>(i, j) = pix_Max;
		}
	}
	
	cv::normalize(dst, dst, 0.0, 1.0, CV_MINMAX);
	dst.convertTo(dst, CV_8U, 255, 0);
	src.convertTo(src, CV_8U, 255, 0);

}

