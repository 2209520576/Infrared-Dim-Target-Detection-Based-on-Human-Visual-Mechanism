#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


inline cv::Mat get_up_left_data(cv::Mat &src, int r1){
	cv::Rect area(0, 0, r1 + 1, r1 + 1);
	cv::Mat up_left_data = src(area);
	return up_left_data;
}

inline cv::Mat get_up_data(cv::Mat &src, int r1, int optsize){
	cv::Rect area(r1 + 1, 0, optsize, r1 + 1);
	cv::Mat up_data = src(area);
	return up_data;
}

inline cv::Mat get_up_right_data(cv::Mat &src, int r1, int r2, int optsize){
	cv::Rect area(r2, 0, r1 + 1, r1 + 1);
	cv::Mat up_right_data = src(area);
	return up_right_data;
}

// 
inline cv::Mat get_left_data(cv::Mat &src, int r1, int optsize){
	cv::Rect area(0, r1 + 1, r1 + 1, optsize);
	cv::Mat left_data = src(area);
	return left_data;
}

inline cv::Mat get_target_data(cv::Mat &src, int r1, int optsize){
	cv::Rect area(r1 + 1, r1 + 1, optsize, optsize);
	cv::Mat target_data = src(area);
	return target_data;
}

inline cv::Mat get_right_data(cv::Mat &src, int r1, int r2, int optsize){
	cv::Rect area(r2, r1 + 1, r1 + 1, optsize);
	cv::Mat right_data = src(area);
	return right_data;
}

//
inline cv::Mat get_down_left_data(cv::Mat &src, int r1, int r2){
	cv::Rect area(0, r2, r1 + 1, r1 + 1);
	cv::Mat down_left_data = src(area);
	return down_left_data;
}

inline cv::Mat get_down_data(cv::Mat &src, int r1, int r2, int optsize){
	cv::Rect area(r1 + 1, r2, optsize, r1 + 1);
	cv::Mat down_data = src(area);
	return down_data;
}

inline cv::Mat get_down_right_data(cv::Mat &src, int r1, int r2, int optsize){
	cv::Rect area(r2, r2, r1 + 1, r1 + 1);
	cv::Mat down_right_data = src(area);
	return down_right_data;
}

//
inline cv::Mat get_N_data(cv::Mat &src, int r1){
	cv::Rect area(0, 0, src.cols, r1 + 1);
	cv::Mat N_data = src(area);
	return N_data;
}

inline cv::Mat get_S_data(cv::Mat &src, int r1, int r2){
	cv::Rect area(0, r2, src.cols, r1 + 1);
	cv::Mat S_data = src(area);
	return S_data;
}

inline cv::Mat get_W_data(cv::Mat &src, int r1){
	cv::Rect area(0, 0, r1 + 1, src.rows);
	cv::Mat W_data = src(area);
	return W_data;
}

inline cv::Mat get_E_data(cv::Mat &src, int r1, int r2){
	cv::Rect area(r2, 0, r1 + 1, src.rows);
	cv::Mat E_data = src(area);
	return E_data;
}

//
inline cv::Mat get_NW_data(cv::Mat &src, int ROI_center){
	cv::Rect area(0, 0, ROI_center + 1, ROI_center + 1);
	cv::Mat	NW_data = src(area);
	return NW_data;
}

inline cv::Mat get_NE_data(cv::Mat &src, int ROI_center){
	cv::Rect area(ROI_center, 0, ROI_center + 1, ROI_center + 1);
	cv::Mat	NE_data = src(area);
	return NE_data;
}

inline cv::Mat get_SW_data(cv::Mat &src, int ROI_center){
	cv::Rect area(0, ROI_center, ROI_center + 1, ROI_center + 1);
	cv::Mat	SW_data = src(area);
	return SW_data;
}

inline cv::Mat get_SE_data(cv::Mat &src, int ROI_center){
	cv::Rect area(ROI_center, ROI_center, ROI_center + 1, ROI_center + 1);
	cv::Mat	SE_data = src(area);
	return SE_data;
}

//
inline double Mean(cv::Mat &src){
	double sum = 0.0, mean = 0.0;
	cv::MatIterator_<double> it = src.begin<double>();
	cv::MatIterator_<double> it_end = src.end<double>();
	for (; it != it_end; ++it){
		sum = sum + *it;
	}
	mean = sum / (src.rows*src.cols);
	return mean;
}

void Calc_gradient(cv::Mat &img, cv::Mat_<double> &Gx, cv::Mat_<double> &Gy)
{
	cv::Mat_<double> dImg;
	img.convertTo(dImg, CV_64F);
	int rows = img.rows;
	int cols = img.cols;

	cv::Mat_<double> xTopVec = dImg.col(1) - dImg.col(0);
	cv::Mat_<double> xBotVec = dImg.col(cols - 1) - dImg.col(cols - 2);
	cv::Mat_<double> xForwMat = dImg(cv::Range(0, rows), cv::Range(0, cols - 2));
	cv::Mat_<double> xBackMat = dImg(cv::Range(0, rows), cv::Range(2, cols));
	cv::Mat_<double> centGx = (xBackMat - xForwMat) / 2;
	cv::Mat_<double> tmpGx = cv::Mat::zeros(rows, cols, CV_64F);
	for (int i = 1; i < cols - 1; i++) {
		centGx.col(i - 1).copyTo(tmpGx.col(i));
	}
	xTopVec.copyTo(tmpGx.col(0));
	xBotVec.copyTo(tmpGx.col(cols - 1));

	cv::Mat_<double> yTopArr = dImg.row(1) - dImg.row(0);
	cv::Mat_<double> yBotArr = dImg.row(rows - 1) - dImg.row(rows - 2);
	cv::Mat_<double> yForwMat = dImg(cv::Range(0, rows - 2), cv::Range(0, cols));
	cv::Mat_<double> yBackMat = dImg(cv::Range(2, rows), cv::Range(0, cols));
	cv::Mat_<double> centGy = (yBackMat - yForwMat) / 2;
	cv::Mat_<double> tmpGy = cv::Mat::zeros(rows, cols, CV_64F);
	for (int i = 1; i < rows - 1; i++) {
		centGy.row(i - 1).copyTo(tmpGy.row(i));
	}
	yTopArr.copyTo(tmpGy.row(0));
	yBotArr.copyTo(tmpGy.row(rows - 1));

	Gx = tmpGx;
	Gy = tmpGy;
}

cv::Mat Feature_extraction(cv::Mat &ROI, int optsize){
	int r1 = ((ROI.rows - optsize) / 2) - 1;
	int r2 = r1 + optsize + 1;
	int ROI_center = (ROI.rows - 1) / 2;
	cv::Mat up_left_data = get_up_left_data(ROI, r1);
	cv::Mat up_data = get_up_data(ROI, r1, optsize);
	cv::Mat up_right_data = get_up_right_data(ROI, r1, r2, optsize);
	cv::Mat left_data = get_left_data(ROI, r1, optsize);
	cv::Mat target_data = get_target_data(ROI, r1, optsize);
	cv::Mat right_data = get_right_data(ROI, r1, r2, optsize);
	cv::Mat down_left_data = get_down_left_data(ROI, r1, r2);
	cv::Mat down_data = get_down_data(ROI, r1, r2, optsize);
	cv::Mat down_right_data = get_down_right_data(ROI, r1, r2, optsize);

	cv::Mat N_data = get_N_data(ROI, r1);
	cv::Mat S_data = get_S_data(ROI, r1, r2);
	cv::Mat W_data = get_W_data(ROI, r1);
	cv::Mat E_data = get_E_data(ROI, r1, r2);


	//Mean_target
	double Mean_target = Mean(target_data);

	//Contrast
	double Mean_up_left = Mean(up_left_data);
	double Mean_up = Mean(up_data);
	double Mean_up_right = Mean(up_right_data);
	double Mean_left = Mean(left_data);
	double Mean_right = Mean(right_data);
	double Mean_down_left = Mean(down_left_data);
	double Mean_down = Mean(down_data);
	double Mean_down_right = Mean(down_right_data);
	double mean_surround_data[8] = { Mean_up_left, Mean_up, Mean_up_right, Mean_left, Mean_right, Mean_down_left, Mean_down, Mean_down_right };
	double max_surround_data = *(std::max_element)(mean_surround_data, mean_surround_data + 8);
	double mean_surround = (Mean_up_left + Mean_up + Mean_up_right + Mean_left + Mean_right + Mean_down_left + Mean_down + Mean_down_right) / 8;
	double Contrast = (Mean_target - mean_surround) / mean_surround;

	//PCM
	double filtered_pix1, filtered_pix2, filtered_pix3, filtered_pix4,
		   filtered_pix5, filtered_pix6, filtered_pix7, filtered_pix8;
	filtered_pix1 = Mean_target - Mean_up_left;
	filtered_pix2 = Mean_target - Mean_up;
	filtered_pix3 = Mean_target - Mean_up_right;
	filtered_pix4 = Mean_target - Mean_right;
	filtered_pix5 = Mean_target - Mean_down_right;
	filtered_pix6 = Mean_target - Mean_down;
	filtered_pix7 = Mean_target - Mean_down_left;
	filtered_pix8 = Mean_target - Mean_left;
	double  diss_1_5 = filtered_pix1*filtered_pix5;
	double  diss_2_6 = filtered_pix2*filtered_pix6;
	double  diss_3_7 = filtered_pix3*filtered_pix7;
	double  diss_4_8 = filtered_pix4*filtered_pix8;

	double arry_pix[4] = {};
	arry_pix[0] = diss_1_5;
	arry_pix[1] = diss_2_6;
	arry_pix[2] = diss_3_7;
	arry_pix[3] = diss_4_8;
	double PCM = *(std::min_element)(arry_pix, arry_pix + 4);

	//Target Max graysclae
	double Maxgray_target = 0.0;
	cv::minMaxLoc(target_data, NULL, &Maxgray_target, NULL, NULL);

	//Variance ratio
	cv::Mat meanN, meanS, meanW, meanE, meanT;
	cv::Mat stddevN, stddevS, stddevW, stddevE, stddevT;
	cv::meanStdDev(N_data, meanN, stddevN);
	cv::meanStdDev(S_data, meanS, stddevS);
	cv::meanStdDev(W_data, meanW, stddevW);
	cv::meanStdDev(E_data, meanE, stddevE);
	cv::meanStdDev(target_data, meanT, stddevT);
	double VarN = stddevN.at<double>(0, 0);
	double VarS = stddevS.at<double>(0, 0);
	double VarW = stddevW.at<double>(0, 0);
	double VarE = stddevE.at<double>(0, 0);
	double VarT = stddevT.at<double>(0, 0);
	VarN = VarN*VarN;
	VarS = VarS*VarS;
	VarW = VarW*VarW;
	VarE = VarE*VarE;
	VarT = VarT*VarT;
	double Var_smean = (VarN + VarS + VarW + VarE) / 4.0;
	double Var_ration = VarT / Var_smean; //std::cout << Var_ration;

	//Saliency Index
	double Saliency_Index = (Maxgray_target*Mean_target) / mean_surround;

	//Gradient characteristics
	cv::Mat_<double > img_gx, img_gy, img_gvalue;
	Calc_gradient(ROI, img_gx, img_gy);
	cv::magnitude(img_gx, img_gy, img_gvalue);

	//First quadrant - NE
	cv::Mat GX1 = get_NE_data(img_gx, ROI_center);
	cv::Mat GY1 = get_NE_data(img_gy, ROI_center);
	cv::Mat Gvalue1 = get_NE_data(img_gvalue, ROI_center);
	int num = 0;
	double gm1 = 0.0;
	double G1 = 0.0;
	for (int i = 0; i < GX1.rows; ++i){
		for (int j = 0; j < GX1.cols; ++j){
			double x = GX1.at<double>(i, j);
			double y = GY1.at<double>(i, j);
			double value = Gvalue1.at<double>(i, j);
			if (x<0 && y>0){
				gm1 = gm1 + value*value;
				++num;
			}
		}
	}
	if (num>0)
		G1 = gm1 / num;

	//The second quadrant - NW
	cv::Mat GX2 = get_NW_data(img_gx, ROI_center);
	cv::Mat GY2 = get_NW_data(img_gy, ROI_center);
	cv::Mat Gvalue2 = get_NW_data(img_gvalue, ROI_center);
	num = 0;
	double gm2 = 0.0;
	double G2 = 0.0;
	for (int i = 0; i< GX2.rows; ++i){
		for (int j = 0; j < GX2.cols; ++j){
			double x = GX2.at<double>(i, j);
			double y = GY2.at<double>(i, j);
			double value = Gvalue2.at<double>(i, j);
			if (x>0 && y>0){
				gm2 = gm2 + value*value;
				++num;
			}
		}
	}
	if (num > 0)
		G2 = gm2 / num;

	//The third quadrant -SW
	cv::Mat GX3 = get_SW_data(img_gx, ROI_center);
	cv::Mat GY3 = get_SW_data(img_gy, ROI_center);
	cv::Mat Gvalue3 = get_SW_data(img_gvalue, ROI_center);
	num = 0;
	double gm3 = 0.0;
	double G3 = 0.0;
	for (int i = 0; i < GX3.rows; ++i){
		for (int j = 0; j < GX3.cols; ++j){
			double x = GX3.at<double>(i, j);
			double y = GY3.at<double>(i, j);
			double value = Gvalue3.at<double>(i, j);
			if (x>0 && y<0){
				gm3 = gm3 + value*value;
				++num;
			}
		}
	}
	if (num>0)
		G3 = gm3 / num;

	//The fourth quadrant -SE
	cv::Mat GX4 = get_SE_data(img_gx, ROI_center);
	cv::Mat GY4 = get_SE_data(img_gy, ROI_center);
	cv::Mat Gvalue4 = get_SE_data(img_gvalue, ROI_center);
	num = 0;
	double gm4 = 0.0;
	double G4 = 0.0;
	for (int i = 0; i < GX4.rows; ++i){
		for (int j = 0; j < GX4.cols; ++j){
			double x = GX4.at<double>(i, j);
			double y = GY4.at<double>(i, j);
			double value = Gvalue4.at<double>(i, j);
			if (x<0 && y<0){
				gm4 = gm4 + value*value;
				++num;
			}
		}
	}
	if (num>0)
		G4 = gm4 / num;

	//Ration
	double G[4] = { G1, G2, G3, G4 };
	double max_G = *(std::max_element)(G, G + 4);
	double min_G = *(std::min_element)(G, G + 4);
	double gradient_char = min_G / max_G;

	//
	cv::Mat feature(1, 6, CV_64F);
	feature.at<double>(0, 0) = Contrast;
	feature.at<double>(0, 1) = Var_ration;
	feature.at<double>(0, 2) = Saliency_Index;
	feature.at<double>(0, 3) = gradient_char;
	feature.at<double>(0, 4) = Mean_target;
	feature.at<double>(0, 5) = Maxgray_target;
	//feature.at<double>(0, 6) = PCM;
	
	return feature;
}

#endif