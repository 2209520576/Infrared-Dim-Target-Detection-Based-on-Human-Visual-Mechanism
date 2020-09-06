#ifndef MULTIGAR_MEASURE_H
#define MULTIGAR_MEASURE_H

#include <opencv2/core.hpp>

class MultiGray_Measure{

public:
	void Multi_grayscale(cv::Mat src, cv::Mat& dst);
	void setoptimalSize_map(cv::Mat param) { optimalSize_map = param; }
	cv::Mat getoptimalSize_map() const { return  optimalSize_map; }
	void Threshod_Seg(cv::Mat src, cv::Mat& dst, double k);

private:
	cv::Mat optimalSize_map;
	
};

#endif