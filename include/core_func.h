#ifndef CORE_H
#define CORE_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector> 
#include <iostream>
#include <fstream>
#include <string>
#include "MultiGray_Measure.h"
#include "FART.h"
#include "feature_extraction.h"

void WriteMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;
		return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<double>(i, j) << "\t";
		}
		fout << std::endl;
	}

	fout.close();
}

void WriteMatToFile(cv::Mat& data,std::string lable, std:: string filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << lable << data;
	fs.release();
}

void LoadModel(cv::Mat& data, std::string lable, std::string filename)
{
	std::fstream ReadFile;
	ReadFile.open(filename, std::ios::in);//ios::in 表示以只读的方式读取文件
	if (ReadFile.fail())//文件打开失败:返回0
	{
		std::cout<<"Can't Open Model file !!!"<<std::endl;
		system("pause");
	}
	cv:: FileStorage fs(filename, cv::FileStorage::READ);;
	fs[lable] >> data;
}

//Get ROI feature_map
void getROIdata(const std::string &imgpath, cv::Mat &feature_map, int sample_num,bool normlize){

	cv::Mat dst, optsize_map, feature;
	for (int n = 1; n <= sample_num; ++n){
		std::string  Read_img_Path = imgpath + std::to_string(n) + ".bmp";
		cv::Mat src = cv::imread(Read_img_Path);
		if (src.empty()){
			continue;
		}
		if (src.channels() > 1)
			cvtColor(src, src, CV_RGB2GRAY);

		MultiGray_Measure multigay_measure;
		multigay_measure.Multi_grayscale(src, dst);
		optsize_map = multigay_measure.getoptimalSize_map();

		int ROI_center = (src.rows - 1) / 2;
		int optsize = optsize_map.at<uchar>(ROI_center, ROI_center);

		src.convertTo(src, CV_64F, 1 / 255.0, 0);
		feature = Feature_extraction(src, optsize);
		if (normlize) cv::normalize(feature, feature, 0.0, 1.0, CV_MINMAX);
		feature_map.push_back(feature);
		std::cout << "Completion of feature extraction img: " << n << std::endl;
	}

}

//Get similarity matrix
void Get_similarity_matrix(cv::Mat src1, cv::Mat src2, cv::Mat &dst){
	dst = cv::Mat::zeros(src1.rows, src1.rows, src1.type());
	for (int i = 0; i < src1.rows; ++i){
		cv::Mat src1_row = src1.rowRange(i, i + 1);
		for (int j = 0; j < src2.rows; ++j){
			cv::Mat src2_row = src2.rowRange(j, j + 1);
			dst.at<double>(i, j) = cv::norm(src1_row, src2_row, cv::NORM_L2);
		}
	}
}

//Draw rectangular boxes
void drawrec(cv::Mat& src, int r, std::vector<cv::KeyPoint> & kp){
	cvtColor(src, src, CV_GRAY2BGR, 0);
	for (int i = 0; i < kp.size(); ++i){
		cv::Point2f pt1;
		pt1.x = kp[i].pt.x - r;
		pt1.y = kp[i].pt.y - r;
		cv::Point2f pt2;
		pt2.x = kp[i].pt.x + r;
		pt2.y = kp[i].pt.y + r;
		cv::rectangle(src, pt1, pt2, cv::Scalar(255, 255, 0), 2, 1, 0);
		//circle(src, cv::Point(kp[i].pt.x, kp[i].pt.y), 0, cv::Scalar(0, 0, 255));
	}
}

//Match
void Match(cv::Mat src, cv::Mat Model,cv::Mat optsizeMap ,std::vector<cv::KeyPoint> & kp,int w,int h,double rho){
	if (src.channels() > 1)
		cvtColor(src, src, CV_RGB2GRAY);
	src.convertTo(src, CV_64F, 1 / 255.0, 0);
	cv::Mat ROI_feature, ROI_feature_map; ROI_feature_map.reserve(100);
	int count = 0;
	FART Improvedfart;
		for (auto it = kp.begin(); it != kp.end(); ){
			int kp_x = (*it).pt.x, kp_y = (*it).pt.y;
			if ((kp_x - ((w - 1) / 2))>0 && (kp_x + ((w - 1) / 2))<src.cols && (kp_y - ((h - 1) / 2))>0 && (kp_y + ((h - 1) / 2))<src.rows) {
			cv::Rect area(kp_x - ((w - 1) / 2), kp_y - ((h - 1) / 2), w, h);
			cv::Mat ROI = src(area);
			int optsize = optsizeMap.at<uchar>(kp_y, kp_x);
			ROI_feature = Feature_extraction(ROI, optsize);
			ROI_feature_map.push_back(ROI_feature);
			++count;
			++it;
			}
			else{
				it=kp.erase(it);
			}
		
	  }
		cv::Mat lable = cv::Mat::zeros(count, 1, CV_8U);
		ROI_feature_map = Improvedfart.complement_coder(ROI_feature_map);
		Improvedfart.test(Model, ROI_feature_map, rho, 1e-6, lable);

		std::vector<cv::KeyPoint> tmp_kp = kp;
		kp.clear();
		for (int i = 0; i < tmp_kp.size(); ++i){
			int lable_element = lable.at<uchar>(i, 0);
			//std::cout << lable_element << std::endl;
			if (lable_element == 1){
				kp.push_back(tmp_kp[i]);
			}
		}	
}

//Suppress background and Enhance target
 cv::Mat Supb_Enht(cv::Mat src, std::vector<cv::KeyPoint> kp, int w, int h ){
	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
	int kp_x , kp_y;
	for (auto it = kp.begin(); it != kp.end(); ++it){
		kp_x = (*it).pt.x;
		kp_y = (*it).pt.y;
		if ((kp_x - ((w - 1) / 2)) > 0 && (kp_x + ((w - 1) / 2))<src.cols && (kp_y - ((h - 1) / 2))>0 && (kp_y + ((h - 1) / 2)) < src.rows){
			int begin_rows = kp_y - ((h - 1) / 2);
			int end_rows = kp_y + ((h - 1) / 2);
			int begin_cols = kp_x - ((w - 1) / 2);
			int end_cols = kp_x + ((w - 1) / 2);
			for (int i = begin_rows; i <= end_rows; ++i){
				for (int j = begin_cols; j <= end_cols; ++j){
					dst.at<uchar>(i, j) = src.at<uchar>(i, j);
				}
			}
		}
	}
	cv::normalize(dst, dst, 0, 255, CV_MINMAX);
	return dst;
}

 //读取txt指定行数据
 void ReadLineData(const std::string &fileName, int lineNum, char* data)
 {
	 std::ifstream in;
	 in.open(fileName);  //打开文件（fileName为文件路径）

	 int line = 1;
	 while (in.getline(data, 1024)) //读取每行数据存至data(data只存一行的数据)
	 {
		 if (lineNum == line) //判断当前行是否是指定行
		 {
			 break;//若是，则读取完毕退出
		 }
		 line++; //否则继续读下一行
	 }
	 in.close(); 
 }

 //获取txt文件行数
 int CountLines(const std::string &filename)
 {
	 std::fstream ReadFile;  
	 int n = 0;
	 std::string tmp;
	 ReadFile.open(filename, std::ios::in);//ios::in 表示以只读的方式读取文件
	 if (ReadFile.fail())//文件打开失败:返回0
	 {
		 return 0;
	 }
	 else//文件存在
	 {
		 while (getline(ReadFile, tmp, '\n'))  //读取每行，每行字符结束自动换行，直到最后一行
		 {
			 n++; //行数
		 }
		 ReadFile.close(); //关闭文件
		 return n; 
	 }
 }

 //截取ROI
 void ROI_extract(cv::Mat &src, const std::string &txt_filename, const std::string &ROI_filname, const std::string &back_filname, std::vector<cv::KeyPoint> & kp, int w, int h, bool &tag, bool &save, int &Ptrue_rtrue, int &Ptrue_rfalse){
	 int tmp;
	 int xmin1 = 0, xmax1 = 0, ymin1 = 0, ymax1 = 0;
	 int xmin2 = 0, xmax2 = 0, ymin2 = 0, ymax2 = 0;
	 int txtline = CountLines(txt_filename);

	 char lineData[1024] = { 0 };
	 std::string l;
	 for (int i = 20; i <= 23; ++i){
		 int flag = 100;
		 int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
		 ReadLineData(txt_filename, i, lineData);
		 l = lineData;
		 for (auto &j : l){
			 if (isdigit(j) && i == 20){
				 tmp = j - '0';
				 xmin1 += tmp*flag;//列
				 flag /= 10;
				 ++count1;
			 }
			 else if (isdigit(j) && i == 21){
				 tmp = j - '0';
				 ymin1 += tmp*flag; //行
				 flag /= 10;
				 ++count2;
			 }
			 else if (isdigit(j) && i == 22){
				 tmp = j - '0';
				 xmax1 += tmp*flag;
				 flag /= 10;
				 ++count3;
			 }
			 else if (isdigit(j) && i == 23){
				 tmp = j - '0';
				 ymax1 += tmp*flag;
				 flag /= 10;
				 ++count4;
			 }
		 }
		 xmin1 = count1 == 2 ? xmin1 / 10 : count1 == 1 ? xmin1 / 100 : xmin1;
		 ymin1 = count2 == 2 ? ymin1 / 10 : count2 == 1 ? ymin1 / 100 : ymin1;
		 xmax1 = count3 == 2 ? xmax1 / 10 : count3 == 1 ? xmax1 / 100 : xmax1;
		 ymax1 = count4 == 2 ? ymax1 / 10 : count3 == 1 ? ymax1 / 100 : ymax1;
	 }

	 if (CountLines(txt_filename) > 26){
		 for (int i = 32; i <= 35; ++i){
			 int flag = 100;
			 int count1 = 0, count2 = 0, count3 = 0, count4 = 0;;
			 ReadLineData(txt_filename, i, lineData);
			 l = lineData;
			 for (auto &j : l){
				 if (isdigit(j) && i == 32){
					 tmp = j - '0';
					 xmin2 += tmp*flag;//列最小值
					 flag /= 10;
					 ++count1;
				 }
				 else if (isdigit(j) && i == 33){
					 tmp = j - '0';
					 ymin2 += tmp*flag; //行最小值
					 flag /= 10;
					 ++count2;
				 }
				 else if (isdigit(j) && i == 34){
					 tmp = j - '0';
					 xmax2 += tmp*flag;//列最大值
					 flag /= 10;
					 ++count3;
				 }
				 else if (isdigit(j) && i == 35){
					 tmp = j - '0';
					 ymax2 += tmp*flag;//行最大值
					 flag /= 10;
					 ++count4;
				 }
			 }
			 xmin2 = count1 == 2 ? xmin2 / 10 : count1 == 1 ? xmin2 / 100 : xmin2;
			 ymin2 = count2 == 2 ? ymin2 / 10 : count2 == 1 ? ymin2 / 100 : ymin2;
			 xmax2 = count3 == 2 ? xmax2 / 10 : count3 == 1 ? xmax2 / 100 : xmax2;
			 ymax2 = count4 == 2 ? ymax2 / 10 : count4 == 1 ? ymax2 / 100 : ymax2;
		 }

	 }

	 //ROI提取
	 int count = 0;
	 for (auto it = kp.begin(); it != kp.end(); ++it){
		 ++count;
		 int kp_x = (*it).pt.x, kp_y = (*it).pt.y;
		 if ((kp_x - ((w - 1) / 2))>0 && (kp_x + ((w - 1) / 2))<src.cols && (kp_y - ((w - 1) / 2))>0 && (kp_y + ((w - 1) / 2))<src.rows)  {
			 if (((kp_x < xmax1 && kp_x>xmin1) && (kp_y < ymax1 && kp_y>ymin1)) || ((kp_x < xmax2 && kp_x>xmin2) && (kp_y < ymax2 && kp_y>ymin2))){
				 cv::Rect area(kp_x - ((w - 1) / 2), kp_y - ((h - 1) / 2), w, h);
				 cv::Mat ROI = src(area);
				 //circle(ROI, cv::Point((ROI.cols - 1) / 2, (ROI.rows - 1) / 2), 0, cv::Scalar(0, 0, 255));
				 if (save==true) cv::imwrite(ROI_filname + std::to_string(count) + ".bmp", ROI);
				 tag = true;
				 ++Ptrue_rtrue;
			 }
			 else{
				 cv::Rect area(kp_x - ((w - 1) / 2), kp_y - ((h - 1) / 2), w, h);
				 cv::Mat background = src(area);
				 if (save == true) cv::imwrite(back_filname + std::to_string(count) + ".bmp", background);
				 ++Ptrue_rfalse;
			 }
		 }

	 }
 }

#endif