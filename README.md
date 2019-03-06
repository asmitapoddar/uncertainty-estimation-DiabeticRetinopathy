# Uncertainty Estimation of Neural Network for Detection of Diabetic Retinopathy
This project aims at disease detection from medical images, specifically, automated screening of Diabetic Retinopathy (DR), using the **PyTorch Framework**. Apart from achieving high accuracy in predicting the class of DR using Convolutional Neural Networks, we estimate the uncertainty of neural networks in making its prediction. We also generate visual explanation of the deep learning system to convey the pixels in the image that influences its decision. Together, these reveal the deep learning system’s competency and limits to the human, and in turn the human can know when to trust the deep learning system. Finally, we have created an end-to-end application which enables an end-user (such as a clinician) to obtain all the results on a dashboard. 

## Environment
Python 3.7

### Dependencies
 The dependencies are available in [requirements.txt](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/requirements.txt)
 
 ## Dataset
 The dataset used to train the network are Diabetic Retinopathy images from the Singapore national DR screening Program (SiDRP).
 There are 5 classes:
 0 - No DR,   
 1 - Mild,   
 2 - Moderate,   
 3 - Severe   
 4 - Proliferative DR
 
 ## Pipeline
 - Training [train.py](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/train.py): We trained a ResNet-18 Convolutional Neural Network model using our dataset of the DR images.
 - Prediction [prediction.py](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/prediction.py): We predict the class of the test DR image. We use an ensemble of classifiers, so we get multpile predictions for the test image. We use the average of the softmax layer values over each class to get our predicted class for an image.
 - Feature attribution [prediction.py](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/prediction.py)
 - Visualization [prediction.py](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/prediction.py)
 - Uncertainty Estimation [prediction.py](https://github.com/asmitapoddar/uncertainty-estimation-DR/blob/master/prediction.py)
 
 ## Usage
 Created a flask application (app.py)  
 
The deep learning system should give high confidence predictions when the predictions are likely to be correct and low confidence when the system is unsure.

