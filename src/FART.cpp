#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include "FART.h"

cv::Mat FART::complement_coder(cv::Mat &data){

	int r = data.rows;
	int c = data.cols;
	cv::Mat coded_data = cv::Mat(r, 2*c, CV_64F);

	for (int i = 0; i<r; ++i)
	{
		cv::Mat dataRow = data.rowRange(i, i + 1).clone();
		cv::normalize(dataRow, dataRow, 0.0, 1.0, CV_MINMAX);
		for (int j = 0; j<c; ++j)
		{
			coded_data.at<double>(i, j) = dataRow.at<double>(0,j);
		}
		for (int j = c; j < 2 * c; ++j){
			coded_data.at<double>(i, j) = 1 - dataRow.at<double>(0, j-c);
		}
	}
	return coded_data;
}

bool FART::stopping_conditions(cv::Mat &w, cv::Mat &w_old,int maxiter){
	bool stop=false;
	int epoch = FART::getepoch();

	if (epoch>maxiter){
		stop = true;
		return  stop;
	}
	 
	int rw = w.rows;
	int rw_old = w_old.rows;
	if (rw != rw_old){
		return  stop;
	}
	else{
		cv::Mat diff = w != w_old;
		if (cv::countNonZero(diff) == 0)	// Equal if no elements disagree
			stop = true;
		    return  stop;
	}
}

double FART::fart_norm(cv::Mat &data){
	int r = data.rows;
	double norm_data=0.0;
	if (r > 1){
		std::cerr << " Only one row of data can be Input to norm!"<<std::endl;
		system("pause");
	}
	
	norm_data = cv::norm(data, cv::NORM_L1);

	return norm_data;
}

cv::Mat FART::fuzzy_min(cv::Mat &data1, cv::Mat &data2){

	int r1 = data1.rows, c1 = data1.cols;
	int r2 = data2.rows, c2 = data2.cols;
	
	if (r1 > 1 || r2>1){
		std::cerr << " Only one row of data can be Input to norm!" << std::endl;
		system("pause");
	}

	if (c1 != c2){
		std::cerr << " Columns of two rows of data must be equal!" << std::endl;
		system("pause");
	}
	cv::Mat min = cv::Mat(r1, c1, CV_64F);
	for (int i = 0; i < r1; ++i){
		for (int j = 0; j < c1; ++j){
			double d1 = data1.at<double>(i, j);
			double d2 = data2.at<double>(i, j);
			min.at<double>(i, j) = d1 <= d2 ? d1 : d2;
		}
	}
	return min;
}

cv::Mat FART::fuzzy_max(cv::Mat &data1, cv::Mat &data2){

	int r1 = data1.rows, c1 = data1.cols;
	int r2 = data2.rows, c2 = data2.cols;

	if (r1 > 1 || r2>1){
		std::cerr << " Only one row of data can be Input to norm!" << std::endl;
		system("pause");
	}

	if (c1 != c2){
		std::cerr << " Columns of two rows of data must be equal!" << std::endl;
		system("pause");
	}
	cv::Mat max = cv::Mat(r1, c1, CV_64F);
	for (int i = 0; i < r1; ++i){
		for (int j = 0; j < c1; ++j){
			double d1 = data1.at<double>(i, j);
			double d2 = data2.at<double>(i, j);
			max.at<double>(i, j) = d1 >= d2 ? d1 : d2;
		}
	}
	return max;
}

void FART::activation_match(FART &obj, cv::Mat &data){
	int r = data.rows;
	if (r > 1){
		std::cerr << " Only one row of data can be Input to norm!" << std::endl;
		system("pause");
	}
	obj.T = cv::Mat::zeros(obj.nCategorie, 1, CV_64F);
	obj.M = cv::Mat::zeros(obj.nCategorie, 1, CV_64F);
	cv::Mat W_TMP;
	for (int i = 0; i < obj.nCategorie; ++i){
		W_TMP = obj.W.rowRange(i, i + 1).clone();
		double numerator = fart_norm(fuzzy_min(data, W_TMP));
		(obj.T).at<double>(i, 0) = numerator / (obj.alpha + fart_norm(W_TMP));
		(obj.M).at<double>(i, 0) = numerator / fart_norm(data);
	}
}

void FART::learn(FART &obj, cv::Mat &data, int index, double Similarity){
	int r = data.rows;
	if (r > 1){
		std::cerr << " Only one row of data can be Input to norm!" << std::endl;
		system("pause");
	}

	obj.beta = Similarity*obj.beta0;
	cv::Mat W_TMP = obj.W.rowRange(index, index + 1).clone();
	W_TMP = obj.beta*(fuzzy_min(data, W_TMP)) + (1.0 - obj.beta) * W_TMP ;
	for (int j = 0; j < W_TMP.cols; ++j){
		obj.W.at<double>(index, j) = W_TMP.at<double>(0, j);
	}
}

double FART::get_Similarity(cv::Mat data1, cv::Mat data2, int method){
	double Similarity = 0.0;
	if (method == 0){
		Similarity = 1;//1.0
		return Similarity;
	}
	else if (method == 1){
		Similarity = fart_norm(fuzzy_min(data1, data2)) / fart_norm(fuzzy_max(data1, data2)); // Jaccard coefficient
		return Similarity;
	}
	else if(method==2){
		double dotVec = data1.dot(data2);  //Dot product
		double normFirst = cv::norm(data1);  //modulus
		double normSecond = cv::norm(data2);
		if (normFirst != 0 && normSecond != 0){
			Similarity = dotVec / (normFirst*normSecond);  // Cosine similarity
		}
		return Similarity;
	}
	else if (method == 3){
		cv::Scalar mean1,mean2;
		mean1=cv::mean(data1);
		mean2 = cv::mean(data2);
		double m1 = mean1.val[0];
		double m2 = mean2.val[0];
		cv::Mat sub1 = data1 - m1;
		cv::Mat sub2 = data2 - m2;
		cv::Scalar Sum12 = cv::sum(sub1.mul(sub2));
		cv::Scalar Sum1 = cv::sum(sub1.mul(sub1));
		cv::Scalar Sum2 = cv::sum(sub2.mul(sub2));
		Similarity = Sum12[0] / (sqrt(Sum1[0])*sqrt(Sum2[0])); // Pearson similarity
		return Similarity;
	}
	else if(method == 4){
		double numerator = fart_norm(fuzzy_min(data1, data2));
		double beta1 = numerator / (1e-6+fart_norm(data1));
		double beta2 = numerator / fart_norm(data2);
		Similarity = (beta1 + beta2) / 2.0;
		return Similarity;
	}

}
/***/
cv::Mat FART::lateral_inhibition_index(FART obj, cv::Mat winnerMat, int index){
	cv::Mat Mk = (obj.W).rowRange(index, index + 1).clone();  //Best match vector
	cv::Mat lateral_inhb = cv::Mat::zeros(winnerMat.rows, winnerMat.rows, Mk.type());
	int M = obj.W.rows;
	cv::Mat Add = cv::Mat::zeros(1, 1,Mk.type());
	for (int i = 0; i < M; ++i){
		cv::Mat sub = (obj.W).rowRange(i, i + 1) - Mk;
		Add = Add + sub*sub.t();
	}
	double add = Add.at<double>(0, 0);
	double sigma = sqrt(add / M);

	for (int i = 0; i < winnerMat.rows; ++i){
		cv::Mat Mi = winnerMat.rowRange(i, i + 1).clone();
		for (int j = 0; j < winnerMat.rows; ++j){
			cv::Mat Mj = winnerMat.rowRange(j, j + 1).clone();
			double norm = cv::norm(Mi, Mj, cv::NORM_L1);
			lateral_inhb.at<double>(i, j) = (1/(sqrt(2*	CV_PI)*sigma))*exp(-(norm*norm) / (2 * sigma*sigma));
		}
	}
	return lateral_inhb;
}

cv::Mat FART::learn_index(cv::Mat data, cv::Mat lateral_inhb, cv::Mat winnerMat, double Thd, int similary_method){
	cv::Mat learn_cff = cv::Mat::zeros(winnerMat.rows, 1, winnerMat.type());
	double sum = 0.0;
	double inhibition_coeff = 0.0;
	double inhibitio;
	for (int i = 0; i < winnerMat.rows; ++i){
		cv::Mat W_Mki = winnerMat.rowRange(i, i + 1).clone();
		double Si = get_Similarity(data, W_Mki, similary_method);
		for (int j = 0; j < winnerMat.rows; ++j){
			if (i != j){
				double Kij = lateral_inhb.at<double>(i, j);
				cv::Mat W_Mkj = winnerMat.rowRange(j, j + 1).clone();
				double Sj = get_Similarity(data, W_Mkj, similary_method);
				if (Sj <= Thd) 
					inhibitio = 0;
				else
					inhibitio = Sj - Thd;
				inhibition_coeff = inhibition_coeff + Kij*inhibitio;
			}
		}
		learn_cff.at<double>(i, 0) = Si - inhibition_coeff;
		sum = sum + (Si - inhibition_coeff);
		inhibition_coeff = 0;
	}
	learn_cff = learn_cff / sum;
	return learn_cff;
}

void FART::Softlearn(FART &obj, cv::Mat &data, cv::Mat learn_index, cv::Mat I_Num, cv::Mat index){
	if (learn_index.rows != index.rows){
		std::cerr << " The number of learning coefficients is inconsistent with the number of winning neurons!" << std::endl;
		system("pause");
	}
	for (int i = 0; i < learn_index.rows; ++i){
		int bmu = index.at<uchar>(i, 0);
		int num = I_Num.at<uchar>(i, 0);
		double learn_ceoff = learn_index.at<double>(i, 0);
		cv::Mat W_old = obj.W.rowRange(bmu, bmu + 1).clone();
		cv::Mat W_new = (W_old*num + data*learn_ceoff) / (num + learn_ceoff);
		for (int j = 0; j < W_new.cols; ++j){
			obj.W.at<double>(bmu, j) = W_new.at<double>(0, j);
		}
	}
}

void FART::SoftFART_train(FART &obj, cv::Mat &data, int maxiter, double Thd,int similary_method){
	std::cout << "Trainnig FA..." << std::endl;

	//Data Information 
	int Sample_num = data.rows;
	int c = data.cols;
	obj.labels = cv::Mat::zeros(Sample_num, 1, CV_8U);

	//Normalization and Complement coding
	cv::Mat x = obj.complement_coder(data);

	//Initialization 
	if (obj.W.rows == 0){
		obj.W = cv::Mat::ones(1, 2 * c, CV_64F);
		obj.nCategorie = 1;
		obj.nClusters = 1;
	}
	obj.W_old = obj.W;

	//Learning
	obj.epoch = 0;
	int flag1 = 0, flag2 = 0, flag3 = 0;
	cv::Mat index_T, index_M, descend_M;
	cv::Mat Neuron_classiNum=cv::Mat::zeros(1,1,CV_8U);
	while (true){
		obj.epoch = obj.epoch + 1;

		for (int i = 0; i < Sample_num; ++i){    //loop over samples
			cv::Mat x_rows = x.rowRange(i, i + 1).clone();
			if (obj.T.rows == 0 || obj.M.rows == 0){
				activation_match(obj, x_rows);
			}
			cv::sortIdx(obj.T, index_T, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort activation function values in descending order     
			cv::sortIdx(obj.M, index_M, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort Vigilance values in descending order
			cv::sort(obj.M, descend_M, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			int large_Vigilance_num = 0;
			double Vigilance_M;
			
			for (int l = 0; l < descend_M.rows; ++l){
				Vigilance_M = descend_M.at<double>(l, 0);
				if (Vigilance_M >= obj.rho_up)
					large_Vigilance_num = large_Vigilance_num + 1;      //Numbers of nCategorie larger than up Vigilance values
			}
			cv::Mat winerIdex = cv::Mat::zeros(large_Vigilance_num, 1, CV_8U);
			cv::Mat winerMat = cv::Mat::zeros(large_Vigilance_num, obj.W.cols, obj.W.type());
			cv::Mat winerT = cv::Mat::zeros(large_Vigilance_num, 1, obj.W.type());
			cv::Mat winerM = winerT;
			int winerNUm = 0;
			bool mismatch_flag = true;  //mismatch flag 

			for (int j = 0; j < obj.nCategorie; ++j){
				int bmu = index_T.at<int>(j, 0);
				Vigilance_M = obj.M.at<double>(bmu, 0);
				if (Vigilance_M >= obj.rho_up || large_Vigilance_num>0){   //Vigilance test upper bound - pass 
					for (int k = 0; k < large_Vigilance_num; ++k){
						int Idex = index_M.at<int>(k, 0);
						cv::Mat W_old=(obj.W).rowRange(Idex, Idex + 1).clone();  // Winner Weight
						winerMat.rowRange(k, k + 1) = W_old;
						winerIdex.at<uchar>(k, 0) = Idex;
						winerT.at<double>(k, 0) = obj.T.at<double>(Idex);
						winerM.at<double>(k, 0) = obj.M.at<double>(Idex);
						Neuron_classiNum.at<uchar>(k, 0) = Neuron_classiNum.at<uchar>(k, 0) + 1;
					}
					cv::Mat winerMT = (winerM + winerT) / 2;
					cv::Point maxLoc;
					cv::minMaxLoc(winerMT, NULL, NULL, NULL, &maxLoc);
					int BestMach_Index = winerIdex.at<uchar>(maxLoc.y, 0);
					cv::Mat KIJ = lateral_inhibition_index(obj, winerMat, BestMach_Index);
					cv::Mat learnCoef = learn_index(x_rows, KIJ, winerMat, Thd, similary_method);
					Softlearn(obj, x_rows, learnCoef, Neuron_classiNum, winerIdex);

					mismatch_flag = false;  // set mismatch flag to false
					flag1 = flag1 + 1;
					break;
				}
			}

			if (mismatch_flag){
				obj.beta = 1;
				obj.nCategorie = obj.nCategorie + 1; // increment the number of categories
				obj.W.push_back(x_rows); // append new category 
				obj.nClusters = obj.nClusters + 1;  // increment the number of clusters
				cv::Mat num = cv::Mat::ones(1, 1, CV_8U);
				Neuron_classiNum.push_back(num);
				flag3 = flag3 + 1;
			}

			obj.T.release();  // clear activation vector
			obj.M.release();  // clear match vector

		}
		//show
		std::printf("Epoch: %d \nCategories: %d \nClusters: %d\n\n", obj.epoch, obj.nCategorie, obj.nClusters);

		// Stopping Conditions
		if (stopping_conditions(obj.W, obj.W_old, maxiter))
			break;

		obj.W_old.release();
		obj.W_old = obj.W;
	}

	//show
	std::cout << "Done!" << std::endl;
}

/******/

void FART::IFART_train(FART &obj, cv::Mat &data, int maxiter, int similary_method){

	std::cout << "Trainnig FA..." << std::endl;

	//Data Information 
	int Sample_num = data.rows;
	int c = data.cols;
	obj.labels = cv::Mat::zeros(Sample_num, 1, CV_8U);

	//Normalization and Complement coding
	cv::Mat x = obj.complement_coder(data);

	//Initialization 
	if (obj.W.rows == 0){
		obj.W = cv::Mat::ones(1,2*c,CV_64F);
		obj.nCategorie = 1;
		obj.nClusters = 1;
	}
	obj.W_old = obj.W;

	//Learning
	obj.epoch = 0;
	int flag1 =0, flag2 = 0, flag3 = 0;
	cv::Mat index_T, index_M, descend_M;
	while (true){
		obj.epoch = obj.epoch + 1;
		
		for (int i = 0; i < Sample_num; ++i){    //loop over samples
			cv::Mat x_rows = x.rowRange(i, i + 1).clone();  
			if (obj.T.rows == 0 || obj.M.rows == 0){
				activation_match(obj, x_rows);
			}
			cv::sortIdx(obj.T, index_T, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort activation function values in descending order     
			cv::sortIdx(obj.M, index_M, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort Vigilance values in descending order
			cv::sort(obj.M, descend_M, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
			int large_Vigilance_num=0;
			double Vigilance_M;
			for (int l = 0; l < descend_M.rows; ++l){
				Vigilance_M = descend_M.at<double>(l, 0);
				if (Vigilance_M >= obj.rho_up)
					large_Vigilance_num = large_Vigilance_num + 1;      //Numbers of nCategorie larger than up Vigilance values
			}

			bool mismatch_flag = true;  //mismatch flag 

			for (int j = 0; j < obj.nCategorie; ++j){
				int bmu = index_T.at<int>(j, 0);
				Vigilance_M = obj.M.at<double>(bmu, 0);
				if (Vigilance_M >= obj.rho_up || large_Vigilance_num>0){   //Vigilance test upper bound - pass    
					for (int k = 0; k < large_Vigilance_num; ++k){
						int index_Mk = index_M.at<int>(k, 0);
						cv::Mat W_Mk = (obj.W).rowRange(index_Mk, index_Mk + 1).clone();  // Winner Weight
						double Similar = get_Similarity(x_rows, W_Mk, similary_method);
						learn(obj, x_rows, index_Mk, Similar); //learning
					}

					mismatch_flag = false;  // set mismatch flag to false
				    flag1 = flag1 + 1;
					break;

				}
				else if (Vigilance_M >= obj.rho_low){
					obj.nCategorie = obj.nCategorie + 1;// increment the number of categories
					obj.W.push_back(x_rows); // append new category 
					mismatch_flag = false;  // set mismatch flag to false
				    flag2 = flag2 + 1;
					break;
				}
			}

			if (mismatch_flag){
				obj.beta = 1;
				obj.nCategorie = obj.nCategorie + 1; // increment the number of categories
				obj.W.push_back(x_rows); // append new category 
				obj.nClusters = obj.nClusters + 1;  // increment the number of clusters
				flag3 = flag3 + 1;
			}

			obj.T.release();  // clear activation vector
			obj.M.release();  // clear match vector
  
		} 
		//show
		std::printf("Epoch: %d \nCategories: %d \nClusters: %d\n\n", obj.epoch, obj.nCategorie, obj.nClusters);

		// Stopping Conditions
		if (stopping_conditions(obj.W, obj.W_old, maxiter))
			break;

		obj.W_old.release();
		obj.W_old = obj.W;
	}

	  //show
	std::cout << "Done!" << std::endl;
}

void FART::FART_train(FART &obj, cv::Mat &data, int maxiter, int similary_method){

	std::cout << "Trainnig FA..." << std::endl;

	//Data Information 
	int Sample_num = data.rows;
	int c = data.cols;

	//Normalization and Complement coding
	cv::Mat x = obj.complement_coder(data);

	//Initialization 
	if (obj.W.rows == 0){
		obj.W = cv::Mat::ones(1, 2 * c, CV_64F);
		obj.nCategorie = 1;
		obj.nClusters = 1;
	}
	obj.W_old = obj.W;

	//Learning
	obj.epoch = 0;
	cv::Mat index_T;
	while (true){
		obj.epoch = obj.epoch + 1;

		for (int i = 0; i < Sample_num; ++i){    //loop over samples
			cv::Mat x_rows = x.rowRange(i, i + 1).clone();
			if (obj.T.rows == 0 || obj.M.rows == 0){
				activation_match(obj, x_rows);
			}
			cv::sortIdx(obj.T, index_T, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort activation function values in descending order     

			bool mismatch_flag = true;  //mismatch flag 

			for (int j = 0; j < obj.nCategorie; ++j){
				int bmu = index_T.at<int>(j, 0);
				double Vigilance_M = obj.M.at<double>(bmu, 0);
				if (Vigilance_M >= obj.rho_up ){   //Vigilance test upper bound - pass    
	
					cv::Mat W_Mk = (obj.W).rowRange(bmu, bmu + 1).clone();  // Winner Weight
					double Similar = get_Similarity(x_rows, W_Mk, similary_method);
					learn(obj, x_rows, bmu, Similar); //learning
		
					mismatch_flag = false;  // set mismatch flag to false
			    	break;
				}
			}

			if (mismatch_flag){
				obj.beta = 1;
				obj.nCategorie = obj.nCategorie + 1; // increment the number of categories
				obj.W.push_back(x_rows); // append new category 
				obj.nClusters = obj.nClusters + 1;  // increment the number of clusters
			}

			obj.T.release();  // clear activation vector
			obj.M.release();  // clear match vector

		}
		//show
		std::printf("Epoch: %d \nCategories: %d \nClusters: %d\n\n", obj.epoch, obj.nCategorie, obj.nClusters);

		// Stopping Conditions
		if (stopping_conditions(obj.W, obj.W_old, maxiter))
			break;

		obj.W_old.release();
		obj.W_old = obj.W;
	}

	//show
	std::cout << "Done!" << std::endl;
}

void FART::test(cv::Mat &model, cv::Mat &test_data, double rho, double alpha, cv::Mat &lable){
	std::cout << "Testing..." << std::endl;
	//Data Information 
	int nCategories = model.rows;
	int Sample_num = test_data.rows;
	int c = test_data.cols;

	for (int i = 0; i < Sample_num; ++i){    //loop over samples
		cv::Mat x_rows = test_data.rowRange(i, i + 1).clone();
		cv::Mat T = cv::Mat::zeros(nCategories, 1, CV_64F);
		cv::Mat M = cv::Mat::zeros(nCategories, 1, CV_64F);

		cv::Mat W_TMP;
		for (int j = 0; j < 2*nCategories/3; ++j){
			W_TMP = model.rowRange(j, j + 1).clone();
			double numerator = fart_norm(fuzzy_min(x_rows, W_TMP));
			T.at<double>(j, 0) = numerator / (alpha + fart_norm(W_TMP));
			M.at<double>(j, 0) = numerator / fart_norm(x_rows);
		}

		cv::Mat index_T;
		cv::sortIdx(T, index_T, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); //Sort activation function values in descending order     
		bool  mismatch_flag = true;  // mismatch flag
		double Vigilance_M;
		for (int k = 0; k < 2*nCategories/3; ++k){
			int bmu = index_T.at<int>(k, 0);
			Vigilance_M = M.at<double>(bmu, 0);
			if (Vigilance_M >= rho){   //Vigilance test upper bound --- pass 
				lable.at<uchar>(i,0) = 1;
				mismatch_flag = false;  // mismatch flag
				break;
			}
		}

		if (mismatch_flag)
			lable.at<uchar>(i, 0) = 0;
	}
}