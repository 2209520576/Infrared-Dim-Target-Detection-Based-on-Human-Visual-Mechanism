#ifndef FART_H
#define FART_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class FART{

public :
   FART()
   {
	   rho_low = 0.95;
	   rho_up = 0.98;
	   alpha = 1e-6;
	   beta = 1.0;
	   beta0 = 1.0;
	   W = cv::Mat();
	   map = cv::Mat();
	   nClusters = 0;
	   nCategorie = 0;
	   epoch = 0;
	   T=cv::Mat();
	   M = cv::Mat();
	   W_old = cv::Mat();
	   labels = cv::Mat();
	}

   cv::Mat complement_coder(cv::Mat &data);

   bool stopping_conditions(cv::Mat &w, cv::Mat &w_old, int iter);

   double fart_norm(cv::Mat &data);

   cv::Mat fuzzy_min(cv::Mat &data1, cv::Mat &data2);

   cv::Mat fuzzy_max(cv::Mat &data1, cv::Mat &data2);

   void activation_match(FART &obj, cv::Mat &data);

   void learn(FART &obj, cv::Mat &data, int index, double Similarity);

   double get_Similarity(cv::Mat data1, cv::Mat data2, int method);

   //************
   cv::Mat lateral_inhibition_index(FART obj, cv::Mat winnerMat, int index);
   
   cv::Mat learn_index( cv::Mat data, cv::Mat lateral_inhb, cv::Mat winnerMat, double Thd, int similary_method);

   void Softlearn(FART &obj, cv::Mat &data, cv::Mat learn_index, cv::Mat IN, cv::Mat index);
   
   void SoftFART_train(FART &obj, cv::Mat &data, int iter, double Thd, int similary_method);
   //************

   void IFART_train(FART &obj, cv::Mat &data, int iter,int similary_method);

   void FART_train(FART &obj, cv::Mat &data, int iter, int similary_method);

   void test(cv::Mat &model, cv::Mat &test_data, double rho, double alpha, cv::Mat &lable);

   void setrho_low(double param) { rho_low = param; }
   double getrho_low() const { return rho_low; }

   void setrho_up(double param) { rho_up = param; }
   double getrho_up() const { return rho_up; }
    
   void setalpha(double param) { alpha = param; }
   double getalpha() const { return alpha; }

   void setbeta(double param) { beta = param; }
   double getbeta() const { return  beta; }

   void setepoch(int param) { epoch = param; }
   int getepoch() const { return  epoch; }

   void setnClusters(int param) { nClusters = param; }
   int getnClusters() const { return  nClusters; }

   void setnCategorie(int param) { nCategorie = param; }
   int getnCategorie() const { return  nCategorie; }

   void setWeight(cv::Mat param) { W = param; }
   cv::Mat getWeight() const { return  W; }

   void setmap(cv::Mat param) { map = param; }
   cv::Mat getmap() const { return  map; }

   void setlabels(cv::Mat param) { labels = param; }
   cv::Mat getlabels() const { return  labels; }

   void setT(cv::Mat param) { T = param; }
   cv::Mat getT() const { return  T; }

   void setW_old(cv::Mat param) { W_old = param; }
   cv::Mat getW_old() const { return  W_old; }


private:
	double  rho_low;
	double  rho_up;
	double  alpha;
	double  beta;
	double  beta0;
	cv::Mat  W;
	cv::Mat map;
	cv::Mat labels;
	int nCategorie;
	int nClusters;
	int epoch;
	cv::Mat  T;
	cv::Mat  M;
	cv::Mat  W_old;
};

#endif