# Uncertainty Estimation of Neural Network for Detection of Diabetic Retinopathy
This project aims at estimating uncertainty of neural networks for automated screening of Diabetic Retinopathy using the PyTorch framework. We also generate visual explanation of the deep learning system to convey the pixels in the image that influences its decision using Integrated Gradient method.  

We apply stochastic batch normalization to obtain uncertainty estimation on a deep neural network trained for detecting diabetic retinopathy. The deep learning system should give high confidence predictions when the predictions are likely to be correct and low confidence when the system is unsure.

## Environment
Python 3.6

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

