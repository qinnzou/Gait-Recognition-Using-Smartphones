# Deep Learning Based Gait Recognition Using Smartphones in the Wild

This is the source code of Deep learning based gait recogntion using smartphones in the wild. We provide the dataset and the pretrained model.

Zou Q, Wang Y, Zhao Y, Wang Q and Li Q, Deep learning based gait recogntion using smartphones in the wild, under review for IEEE Transactions on Information Forensics and Security, 2019.

Comparing with other biometrics, gait has advantages of being unobtrusive and difficult to conceal. Inertial sensors such as accelerometer and gyroscope are often used to capture gait dynamics. Nowadays, these inertial sensors have commonly been integrated in smartphones and widely used by average person, which makes it very convenient and inexpensive to collect gait data. In this paper, we study gait recognition using smartphones in the wild. Unlike traditional methods that often require the person to walk along a specified road and/or at a normal walking speed, the proposed method collects inertial gait data under a condition of unconstraint without knowing when, where, and how the user walks. To obtain a high performance of person identification and authentication, deep-learning techniques are presented to learn and model the gait biometrics from the walking data. Specifically, a hybrid deep neural network is proposed for robust gait feature representation, where features in the space domain and in the time domain are successively abstracted by a convolutional neural network and a recurrent neural network. In the experiments, two datasets collected by smartphones on a total of 118 subjects are used for evaluations. Experiments show that the proposed method achieves over 93.5% and 93.7% accuracy in person identification and authentication, respectively.

# Networks
## Network Architecture for Gait-extraction
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/net-seg.png)
### Network Architecture Details for Gait-extraction
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/extraction-details.png)

## Network Architecture for Identification
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/net-identification.png)

### CNN+LSTM
It is the network introduced in Fig. 4, which combines the above two networks. The whole network has to be trained from scratch.
### CNNfix+LSTM
It is also the network introduced in Fig. 4. When training, the parameters of CNN are fixed as that in the CNN model that has been trained independently, and the parameters of the LSTM and fully connected layer have to be trained from scratch.
### CNN+LSTMfix
It is also the network introduced in Fig. 4. When training, the parameters of LSTM are fixed as that in the LSTM model that has been trained independently, and the CNN and fully connection layer have to be trained from scratch.

## Network Architecture for Authentication
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/net-authentication.png)
### CNN+LSTM horizontal 
The ‘CNN+LSTM’ network, as have been introduced in Fig. 5, using horizontally aligned data pairs as the input. The weight parameters of CNN are unfixed in the training.
### CNN+LSTM vertical
The ‘CNN+LSTM’ network using vertically aligned data pairs as the input. The weight parameters of CNN are unfixed in the training.
### CNNfix+LSTM horizontal
The ‘CNNfix+LSTM’ network, as have been introduced in Fig. 5, using horizontally aligned data pairs as the input. The weight parameters of CNN are fixed in the training.
### CNNfix+LSTM vertical
The ‘CNNfix+LSTM’ network using vertically aligned data pairs as the input. The weight parameters of CNN are fixed in the training.

## Codes Download:
You can download these codes from the following link：

https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/tree/master/code

# Datasets
## Dataset for Identification & Authentication
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/datasets-for-identification%26authentication.png)
A number of 118 subjects are involved in the data collection. Among them, 20 subjects collect a larger amount of data in two days, with each has thousands of samples, and 98 subjects collect a smaller amount of data in one day, with each has hundreds of samples. Each data sample contains the 3-axis accelerometer data and the 3-axis gyroscope data. The sampling rate of all sensor data is 50 Hz. According to the different evaluation purposes, we construct six datasets based on the collected data. 
### Dataset #1
This dataset is collected on 118 subjects. Based on the step-segmentation algorithm introduced in Section III-B, the collected gait data can be annotated into steps. Following the findings that two-step data have a good performance in gait recognition [7], we collected gait samples by dividing the gait curve into two continuous steps. Meanwhile, we interpolate a single sample into a fixed length of 128 (using Linear Interpolation function). In order to enlarge the scale of the dataset, we make a one-step overlap between two neighboring samples for all subjects. In this way, a total number of 36,884 gait samples are collected. We use 33,104 samples for training, and the rest 3,740 for test. 
### Dataset #2
This dataset is collected on 20 subjects. We also divide the gait curve into two-step samples and interpolate them into the same length of 128. As each subject in this dataset has a much larger amount of data as compared to the that in Dataset #1, we do not make overlap between the samples. Finally, a total number of 49,275 samples are collected, in which 44,339 samples are used for training, and the rest 4,936 for test. 
### Dataset #3
This dataset is collected on the same 118 subjects as in Dataset #1. Different from Dataset #1, we divide the gait curve by using a fixed time length, instead of a step length. Exactly, we collect a sample with a time interval of 2.56 seconds. While the frequency of data collection is 50Hz, the length of each sample is also 128. Also, we make an overlap of 1.28 seconds to enlarge the dataset. A total number of 29,274 samples are collected, in which 26,283 samples are used for training, and the rest 2,991 for test.
### Dataset #4
This dataset is collected on 20 subjects. We also divide the gait curve in an interval of 2.56 seconds. We make no overlap between the samples. Finally, a total number of 39,314 samples are collected, in which 35,373 samples are used for training, and the rest 3,941 for test.
### Dataset #5
This dataset is used for authentication. It contains 74,142 authentication samples of 118 subjects, where the training set is constructed on 98 subjects and the test set is constructed on the other 20 subjects. There are 66,542 samples and 7,600 samples for training and test, respectively. Each authentication sample contains a pair of data sample that are from two different subjects or one same subject. The data sample consists of a 2-step acceleration and gyroscopic data, which are interpolated in the way as described in Dataset #1 and Dataset #2. The two data samples are horizontally aligned to create an authentication sample.
### Dataset #6
This dataset is also used for authentication. The authentication samples are constructed as the same as in Dataset #5. The only difference is that, in authentication sample construction, two data samples from two subjects are vertically aligned instead of horizontally aligned.


## Datasets for Gait-Extraction
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/datasets-for-gait-extraction.png) 
### Dataset #7
We took 577 samples from 10 subjects, with
data shape of 6×1024. 519 of them were used for training
and 58 were used for testing. Both the training and testing
datasets have data from these 10 subjects. There is no
overlap between the training sample and the test sample.
### Dataset #8
We took 1,354 samples from 118 subjects, with
data shape of 6×1024. In order to make the training and
test data come from different subjects, we use 1022 samples
from 20 subjects as training data and 332 samples from
other 98 subjects for testing. 
## Datasets Download:
You can download these datasets from the following link：

https://sites.google.com/site/qinzoucn/
or https://1drv.ms/f/s!AittnGm6vRKLyh3yWS7XaXfyUNQp

# Set up
## Requirements
PyTorch 0.4.0  
Python 3.6  
CUDA 8.0  
We run on the Intel Core Xeon E5-2630@2.3GHz, 64GB RAM and two GeForce GTX TITAN-X GPUs.

# Results
## Results for Gait-Extraction
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/results-for-extraction2.png) 

Here is four examples of walking data extraction results, where blue represents walking data, green represents non-walking data, and red represents unclassified data.
In dataset #7, our method achieved an accuracy of 90.22%,
which shows that our method is effective for segmentation of
walking data and non-walking data. Further, in the dataset #8
where the training data and the testing data are respectively
from different subjects, our method achieved an accuracy of
85.57%, which shows that our method has good robustness on
datasets that are not involved in training.

## Results for Identification
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/results-for-LSTMs.png) 

Performance of different LSTM networks. The classification experiments are conducted on 118 subjects. For each group of results, the left, middle, and right bars correspond to the results of the single-layer LSTM (SL-LSTM), the bi-directional LSTM (Bi-LSTM) and the double-layer LSTM (DL-LSTM), respectively.

![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/results-for-identification.png) 

Here is the classification results using Dataset #1 and Dataset #1. 

## Results for Authentication
![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/results-for-authentication-table.png) 


Here shows the authentication results obtained by
four deep-learning-based methods, i.e., LSTM, CNN, ‘CN-
N+LSTM’, and ‘CNNfix+LSTM’, and three traditional meth-
ods, i.e., EigenGait, Wavelet and Fourier. Note that, the Dataset #5 and Dataset #6 are constructed on the same 118 subjects and the same samples. The only difference is that, the input data have been aligned in two different manners. Exactly, the samples are aligned in horizontal for Dataset #5 and in vertical for Dataset #6. 

![image](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones/blob/master/images/results-for%20authentication.png) 


# Reference
```
@article{zou2019gait,
  title={Deep learning based gait recogntion using smartphones in the wild}
  author={Q. Zou and Y. Wang and Y. Zhao and Q. Wang and Q. Li},
  journal={arXiv preprint arXiv:1811.00338},
  year={2018},
}
```

# Copy Right:
This dataset was collected for academic research. It MUST NOT be used for commercial purposes. 
# Contact: 
For any problem about this dataset, please contact Dr. Qin Zou (qzou@whu.edu.cn).
