# 基于视觉注意机制的红外弱小目标检测
# Infrared-Dim-Target-Detection-Based-on-Human-Visual-Mechanism

### 若有疑问交流,可联系：wcw_cg@163.com.

### 若使用该项目的数据（图片），请注明出处，谢谢！    
### 数据链接：https://gas.graviti.cn/dataset/datawhale/MSIDT  或 关注公众号【OpenImage】获取。
       
        
        
### 1.环境配置     

### win+vs2013+opencv3.1.0+Matlab(2010以上版本)

### 2.程序文件      

#### Core_func：核心配置功能——文件读写、标签信息提取、ROI提取、评价指标计算、模型加载、模型匹配、背景抑制、matlab和opencv矩阵类的转换、画检测框等          

#### FART.hpp：FART和本文改进的Soft-FART类的定义    

#### FART.cpp：FART和本文改进的Soft-FART类的实现      

#### FastConerDetector.hpp：FAST角点检测类的定义（含本文设计的NMS）    

#### FastConerDetector.cpp：FAST角点检测实现（含本文设计的NMS）    
  
#### feature_extraction.hpp：特征提取的定义和实现    

#### MultiGray_Measure.hpp：多尺度灰度方差估计类的定义    

#### MultiGray_Measure.cpp：多尺度灰度方差估计类的实现     

#### Main：主程序运行       



### 3.模型文件       

#### SoftFART_Model.xml：使用Soft-FART训练的模型     
      
#### FART_Model.xml：使用FART训练的模型

### 如何运行？     

##### 近期会写！！！
