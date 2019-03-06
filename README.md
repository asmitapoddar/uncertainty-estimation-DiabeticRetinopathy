# Uncertainty Estimation of Neural Network for Detection of Diabetic Retinopathy
This project aims for disease detection from medical images, specifically, automated screening of Diabetic Retinopathy (DR). Apart from achieving high accuracy in predicting the class of DR, we estimate the uncertainty of neural networks in making its prediction. We also generate visual explanation of the deep learning system to convey the pixels in the image that influences its decision. Together, these reveal the deep learning system’s competency and limits to the human, and in turn the human can know when to trust the deep learning system.  using the PyTorch framework. 

We apply stochastic batch normalization to obtain uncertainty estimation on a deep neural network trained for detecting diabetic retinopathy. The deep learning system should give high confidence predictions when the predictions are likely to be correct and low confidence when the system is unsure.

## Environment
Python 3.7

## Dependencies
torch  
pandas   
numpy  
seaborn  
matplotlib  
scikit-learn  
scipy.misc  
glob  
PIL  
argparse

