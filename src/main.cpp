#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <algorithm>
#include "feature_extraction.h"
#include "FART.h"
#include "core_func.h"
#include "FastConerDetector.h"
#include "MultiGray_Measure.h"

enum method{ Fart , SoftFart };

int trainMain(method tm){

	FART ImprovedFart;
	//Get Feature 
	std::string Root_Path = "I:\\Dim_target_Data\\train_ROI_data\\Train_targetROI_total_data1\\";
	cv::Mat feature_map;
	int sample_num = 4032;
	getROIdata(Root_Path, feature_map, sample_num, true); // feature Will normlized in complement_coder ,so there is false

	//FART_Train
	ImprovedFart.setrho_low(0.97);
	ImprovedFart.setrho_up(0.99);
	ImprovedFart.setepoch(10);
	int maxepoch = ImprovedFart.getepoch();
	int Thd = 0.58;

	if (tm == Fart){
		ImprovedFart.FART_train(ImprovedFart, feature_map, maxepoch, 0);//Similary_method:0 —— 1,1 —— Jaccard coefficient,2 ——  Cosine similarity
		cv::Mat Model = ImprovedFart.getWeight();
		WriteMatToFile(Model, "FART_Model", "FART_Model.xml");
	}
	else if (tm == SoftFart){
		ImprovedFart.SoftFART_train(ImprovedFart, feature_map, maxepoch, Thd, 2);//Similary_method:0 —— 1,1 —— Jaccard coefficient,2 ——  Cosine similarity
		cv::Mat Model = ImprovedFart.getWeight();
		WriteMatToFile(Model, "SoftFART_Model", "SoftFART_Model.xml");
	}
	return 0;
}



int testMain(method tm){

	std::string in_rootfile = "I:\\Dim_target_Data\\";
	std::string out_rootfile = "I:\\FuzzzyART_experiment\\";
	std::string backgroug_type_file = "architecture\\";
	std::string xml_rootfile = in_rootfile + "test_ROI_lable\\" + backgroug_type_file;
	std::string result_target_rootfile = out_rootfile + "testData_result_target\\" + backgroug_type_file;
	std::string result_background_rootfile = out_rootfile + "testData_result_background\\" + backgroug_type_file;
	std::ofstream Nodetected(result_target_rootfile + "Nodetected.txt", std::ofstream::out);//创建txt文件并打开，用于保存没有检测到ROI的图像名称
	int Ptrue_rtrue = 0, Ptrue_rfalse = 0, Pfalse_rtrue = 0;

	FastConerDetector FastPoint;
	FastPoint.setTH(20);
	FastPoint.set_detect_type(1);
	FastPoint.set_diameter(70);
	FastPoint.set_Size(5);

	cv::Mat Model;
	if (tm == Fart)
		LoadModel(Model, "FART_Model", "FART_Model.xml");
	else if (tm == SoftFart)
		LoadModel(Model, "SoftFART_Model", "SoftFART_Model.xml");

	double time, ave_time = 0.0;
	int cout = 0;
	for (int n = 1; n <= 1001; ++n){
		std::string img_infile = in_rootfile + "test_data\\" + backgroug_type_file + std::to_string(n) + ".bmp";
		std::string img_outfile = out_rootfile + "draw_result\\" + backgroug_type_file + std::to_string(n) + ".bmp";
		std::string result_outfile = out_rootfile + "final_result\\" + backgroug_type_file + std::to_string(n) + ".bmp";
		std::string MGVM_outfile = out_rootfile + "MGVM_result\\" + backgroug_type_file + std::to_string(n) + ".bmp";

		std::string xml_infile = xml_rootfile + std::to_string(n) + ".xml";
		std::string result_target_outfile = result_target_rootfile + std::to_string(n) + "_";
		std::string result_background_outfile = result_background_rootfile + std::to_string(n) + "_";
		cv::Mat src = cv::imread(img_infile);

		if (src.empty()){
			continue;
		}

		if (src.channels() > 1)
			cvtColor(src, src, CV_RGB2GRAY);
		cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_64F);
		cv::Mat optimalSize_map;

		double t2 = (double)cv::getTickCount();

		//Computation of Multi-gray scale measure
		MultiGray_Measure multigay_measure;
		multigay_measure.Multi_grayscale(src, dst);
		optimalSize_map = multigay_measure.getoptimalSize_map();

		//FAST Coner Detection
		std::vector<cv::KeyPoint> KP; KP.reserve(100);
		FastPoint.set_nms(false);
		FastPoint.set_detect_type(1);
		FastPoint.Fast(dst, KP);
		FastPoint.NMS(src, KP);

		//Match
		std::cout << "--------------------------------------" << std::endl;
		if (KP.size()>1)
			Match(src, Model, optimalSize_map, KP, 21, 21, 0.99);

		//Suppress background and Enhance target
		cv::Mat result = Supb_Enht(dst, KP, 21, 21);

		t2 = (double)cv::getTickCount() - t2;
		time = (1000 * t2) / ((double)cv::getTickFrequency());
		std::cout << n << "th Image: " << "Process time=" << time << " ms. " << std::endl;
		ave_time += time;
		++cout;

		//ROI extract  and Count nums
		int ptnum = KP.size();
		bool hasxml = true;
		std::ifstream in(xml_infile);
		if (!in.is_open()){  //无xml文件
			hasxml = false;
			if (Nodetected.is_open()){
				Nodetected << n << ".bmp" << "----->(No .xml file!)" << std::endl;
			}
		}

		if (ptnum == 0){ //没有一个ROI被检测到
			if (hasxml) ++Pfalse_rtrue;
			if (Nodetected.is_open()){
				Nodetected << n << ".bmp" << "----->(NO KeyPoint detected!)" << std::endl;
			}
		}

		bool tag = false;
		bool save = false;
		if (hasxml) ROI_extract(src, xml_infile, result_target_outfile, result_background_outfile, KP, 21, 21, tag, save, Ptrue_rtrue, Ptrue_rfalse);

		if (ptnum>0 && tag == false){   //有背景被检测，但目标区域未被检测到
			if (hasxml) ++Pfalse_rtrue;
			if (Nodetected.is_open()){
				Nodetected << n << ".bmp" << "----->(Target Not detected but have backgroud detected!)" << std::endl;
			}
		}

		//Draw rectangular boxes
		drawrec(src, 10, KP);
		drawrec(dst, 10, KP);

		//show
		cv::namedWindow("dst", CV_WINDOW_NORMAL);
		cv::imshow("dst", dst);
		cv::namedWindow("result", CV_WINDOW_NORMAL);
		cv::imshow("result", result);
		cv::namedWindow("src", CV_WINDOW_NORMAL);
		cv::imshow("src", src);
		cv::waitKey(1);

		std::cout << "Ptnum=" << ptnum << std::endl;
		std::cout << n << "th img " << "end of processing！" << std::endl;

		//Save
		//cv::imwrite(img_outfile, src);
		//cv::imwrite(MGVM_outfile, dst);
		//cv::imwrite(result_outfile, result);

	}
	Nodetected.close();
	double Precision = double(Ptrue_rtrue) / (Ptrue_rtrue + Ptrue_rfalse);
	double Call = double(Ptrue_rtrue) / (Ptrue_rtrue + Pfalse_rtrue);
	double F1_index = 2 * Precision*Call / (Precision + Call);
	std::cout << std::endl << "Total_rtrue_num：" << Ptrue_rtrue + Pfalse_rtrue;
	std::cout << std::endl << "Total_Ptrue_num：" << Ptrue_rtrue + Ptrue_rfalse << std::endl;
	std::cout << "Ptrue_rtrue：" << Ptrue_rtrue << std::endl;   //predict(p),real(r)
	std::cout << "Ptrue_rfalse：" << Ptrue_rfalse << std::endl;
	std::cout << "Pfalse_rtrue：" << Pfalse_rtrue << std::endl << std::endl;
	std::cout << "Precision ：" << Precision * 100 << "%" << std::endl;
	std::cout << "Call：" << Call * 100 << "%" << std::endl;
	std::cout << "F1_index：" << F1_index << std::endl;
	std::cout << "Ave_time：" << ave_time / cout << " ms" << std::endl;

	cv::waitKey(0);
	return 0;
}


int main(){

	method tm =SoftFart;
	testMain(tm);

	return 0;
}
