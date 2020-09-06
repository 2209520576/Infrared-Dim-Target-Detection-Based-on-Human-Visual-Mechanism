#ifndef FASTCONERDETECTOR_H
#define FASTCONERDETECTOR_H

#include <opencv2/core.hpp>


class FastConerDetector{
public:
	FastConerDetector(){
		TH = 20;
		nomaxsuprression = false;
		diameter = 60;
		Size = 5;
		detect_type = 1;
	}

	void Fast(cv::Mat& src, std::vector<cv::KeyPoint>& KP);

	void NMS(cv::Mat &src, std::vector<cv::KeyPoint> & kp);

	inline void setTH(int param) { TH = param; }

	inline void set_nms(bool param) { nomaxsuprression = param; }
     
	inline void set_detect_type(int param) { detect_type = param; }

	inline void set_diameter(int param) { diameter = param; }

	inline void set_Size(int param) { Size = param; }

private:
	int TH;
	bool nomaxsuprression;
	int detect_type;
	int	diameter;
	int Size;

};

#endif